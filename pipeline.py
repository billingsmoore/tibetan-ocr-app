"""
Headless, two-stage split of BDRC's ``OCRPipeline`` for interactive line editing.

Upstream ``OCRPipeline.run_ocr()`` runs line detection, line extraction and text
recognition in a single call, which leaves no place to intervene. This module
cuts it at its natural seam -- the sorted list of ``Line`` objects -- so a UI can
show the detected lines, let a human correct them, and only then run recognition:

    page  = detect(image, "Models/Lines/PhotiLines.onnx")
    boxes = lines_to_boxes(page.lines)          # -> editable in a UI
    ...                                          # user edits/adds/deletes boxes
    lines = boxes_to_lines(boxes, page.lines)   # reconcile back to Line objects
    text  = ocr(page.image, lines, "OCRModels/Woodblock")

Note that detection *deskews* the page: ``DetectedPage.image`` is rotated by
``DetectedPage.angle`` relative to the input. All boxes are in the coordinate
space of that deskewed image, so a UI must display ``page.image`` -- not the
original upload -- or the overlay will not line up.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from uuid import UUID

import cv2
import numpy as np
import numpy.typing as npt

# pyewts ships as an sdist whose setup.py imports pkg_resources, which setuptools
# 81+ removed -- so it cannot be pip-installed in a modern build environment.
# It is pure Python and Apache-2.0, so we vendor it. See THIRD_PARTY_NOTICES.md.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import pyewts  # noqa: E402

from BDRC.Data import (
    BBox,
    CharsetEncoder,
    Encoding,
    Line,
    LineDetectionConfig,
    Platform,
)
from BDRC.image_dewarping import apply_global_tps, check_for_tps
from BDRC.Inference import LineDetection, OCRInference
from BDRC.line_detection import (
    build_line_data,
    build_raw_line_data,
    extract_line_images,
    filter_line_contours,
    rotate_from_angle,
    sort_lines_by_threshold2,
)
from BDRC.Utils import generate_guid, get_platform, import_local_model

DEFAULT_PATCH_SIZE = 512

# The detector pads its input up to a whole grid of patch_size tiles and runs the
# UNet on every tile, so cost is quantised by *tile count*, not pixel count. A
# 3500x912 page is a 7x2 grid (14 tiles, ~20s); at half scale it is 4x1 (4 tiles,
# ~5s) with identical lines found. MAX_DETECT_TILES caps that grid.
MAX_DETECT_TILES = 4

# Below this the mask gets too coarse to separate closely-set lines.
MIN_DETECT_SCALE = 0.35


@dataclass
class DetectedPage:
    """Result of the detection stage.

    Attributes:
        image: The deskewed page image (BGR). All line coordinates refer to this.
        angle: Rotation in degrees applied to the input to produce ``image``.
        lines: Detected lines in reading order (top to bottom).
    """

    image: npt.NDArray
    angle: float
    lines: List[Line]


# --------------------------------------------------------------------------- #
# Stage 1: detection
# --------------------------------------------------------------------------- #

_line_detectors: Dict[str, LineDetection] = {}


def _get_line_detector(
    model_path: str, patch_size: int, providers: Sequence[str] | None = None
) -> LineDetection:
    """Return a cached LineDetection session; ONNX sessions are costly to build.

    Keyed by providers too: a session built for CUDA cannot serve a CPU-only
    request, so the two are cached side by side.
    """
    key = f"{model_path}:{patch_size}:{','.join(providers) if providers else 'default'}"
    if key not in _line_detectors:
        config = LineDetectionConfig(model_file=model_path, patch_size=patch_size)
        _line_detectors[key] = LineDetection(
            get_platform(), config, providers=list(providers) if providers else None
        )
    return _line_detectors[key]


def line_session_providers(
    line_model: str, patch_size: int = DEFAULT_PATCH_SIZE, providers=None
) -> List[str]:
    """Execution providers the line-detection session is actually using.

    ONNX Runtime silently falls back to CPU when a requested provider cannot
    initialise, so the only way to know where inference ran is to ask the
    session after the fact.
    """
    detector = _get_line_detector(line_model, patch_size, providers)
    return list(detector._inference.get_providers())


def _detection_scale(width: int, height: int, patch_size: int) -> float:
    """Pick the largest scale <= 1.0 whose tile grid fits in MAX_DETECT_TILES."""
    scale = 1.0
    while scale > MIN_DETECT_SCALE:
        tiles_x = math.ceil(width * scale / patch_size)
        tiles_y = math.ceil(height * scale / patch_size)
        if tiles_x * tiles_y <= MAX_DETECT_TILES:
            return scale
        scale -= 0.05
    return MIN_DETECT_SCALE


def detect(
    image: npt.NDArray,
    line_model: str | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    merge_lines: bool = True,
    class_threshold: float = 0.9,
    fast: bool = True,
    providers: Sequence[str] | None = None,
) -> DetectedPage:
    """Detect text lines on a page, stopping before any cropping or recognition.

    With ``fast`` the mask is computed on a downscaled copy and the resulting
    boxes are projected back to full resolution. This is exact rather than
    approximate: ``rotate_from_angle`` rotates about the image centre with an
    unchanged output size, so scaling commutes with the deskew rotation. The
    returned image is always full resolution, because line crops feed a
    recogniser expecting ~100px-tall text and cropping from a downscaled page
    would halve the detail it sees.

    Args:
        image: Page image as a BGR numpy array (e.g. from ``cv2.imread``).
        line_model: Path to the PhotiLines ONNX model; defaults to the Hub copy.
        patch_size: Tile size the model was trained on.
        merge_lines: Group horizontally split fragments into single lines.
        class_threshold: Confidence cutoff for the line mask.
        fast: Downscale before detection to cut the tile count.

    Returns:
        A ``DetectedPage`` whose ``image`` is deskewed full-resolution and whose
        ``lines`` are sorted into reading order.

    Raises:
        ValueError: If the image is empty or no lines survive detection.
    """
    if image is None or image.size == 0:
        raise ValueError("input image is empty")

    if line_model is None:
        from models import line_model_path

        line_model = line_model_path()

    height, width = image.shape[:2]
    scale = _detection_scale(width, height, patch_size) if fast else 1.0

    if scale < 1.0:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image

    detector = _get_line_detector(line_model, patch_size, providers)
    line_mask = detector.predict(small, class_threshold=class_threshold)

    _, rot_mask, contours, angle = build_raw_line_data(small, line_mask)
    if not contours:
        raise ValueError("no lines detected on this page")

    contours = filter_line_contours(rot_mask, contours)
    if not contours:
        raise ValueError("no lines survived contour filtering")

    line_data = [build_line_data(c) for c in contours]
    sorted_lines, _ = sort_lines_by_threshold2(
        rot_mask, line_data, group_lines=merge_lines
    )

    # Deskew the full-resolution page with the angle measured on the small copy,
    # then lift the lines back into full-resolution coordinates.
    rot_img = rotate_from_angle(image, angle)
    lines = [_scale_line(line, 1.0 / scale) for line in sorted_lines] if scale < 1.0 \
        else list(sorted_lines)

    return DetectedPage(image=rot_img, angle=angle, lines=lines)


def dewarp(
    image: npt.NDArray,
    line_model: str | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    tps_threshold: float = 0.25,
    providers: Sequence[str] | None = None,
) -> Tuple[npt.NDArray, bool]:
    """Flatten a curved page so its text lines run straight.

    Photographed pecha pages bow toward the spine or the edges, which drags
    parts of a line outside any rectangle drawn around it. This fits a thin
    plate spline to the curvature of the most distorted detected line and warps
    the whole page by it.

    Kept as a separate preprocessing step rather than a flag on ``detect()``:
    dewarping changes the image, so detection simply runs afterwards on the
    result. It deliberately runs at full resolution -- a non-linear warp cannot
    be projected back from a downscaled copy the way a rotation can -- so it
    roughly doubles the cost of the detection step and is opt-in.

    Args:
        image: Page image as BGR.
        line_model: Path to the line ONNX model; defaults to the Hub copy.
        patch_size: Tile size the model was trained on.
        tps_threshold: Fraction of lines that must look curved before warping.

    Returns:
        ``(image, changed)`` -- the original image untouched when the page is
        already flat or too few lines were found to judge.
    """
    if image is None or image.size == 0:
        raise ValueError("input image is empty")

    if line_model is None:
        from models import line_model_path

        line_model = line_model_path()

    detector = _get_line_detector(line_model, patch_size, providers)
    mask = detector.predict(image)
    rot_img, rot_mask, contours, _ = build_raw_line_data(image, mask)
    contours = filter_line_contours(rot_mask, contours)

    if len(contours) < 2:  # too little evidence to model the curvature
        return image, False

    ratio, line_data = check_for_tps(rot_img, contours)
    if ratio <= tps_threshold:
        return image, False

    warped_image, _ = apply_global_tps(rot_img, rot_mask, line_data)
    return warped_image, True


def _scale_line(line: Line, factor: float) -> Line:
    """Project a Line into a coordinate space scaled by ``factor``."""
    contour = np.round(line.contour.astype(np.float64) * factor).astype(np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    return Line(
        guid=line.guid,
        contour=contour,
        bbox=BBox(x, y, w, h),
        center=(x + w // 2, y + h // 2),
    )


# --------------------------------------------------------------------------- #
# Box <-> Line reconciliation
# --------------------------------------------------------------------------- #


def lines_to_boxes(lines: Sequence[Line]) -> List[dict]:
    """Flatten Line objects into plain JSON-safe boxes for a UI."""
    return [
        {
            "id": str(line.guid),
            "x": int(line.bbox.x),
            "y": int(line.bbox.y),
            "w": int(line.bbox.w),
            "h": int(line.bbox.h),
        }
        for line in lines
    ]


def _rect_contour(x: int, y: int, w: int, h: int) -> npt.NDArray:
    """Build a 4-point contour matching a rectangle.

    ``extract_line_images`` crops through ``line.contour``, not ``line.bbox``, so
    a hand-drawn or hand-resized box has to be expressed as a contour.
    """
    return np.array(
        [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32
    )


def boxes_to_lines(
    boxes: Sequence[dict], original_lines: Sequence[Line] = ()
) -> List[Line]:
    """Turn UI boxes back into Line objects, preserving contours where possible.

    A detected contour hugs the actual glyph shape, which crops better than a
    plain rectangle on warped or slanted text. So a box whose geometry is
    unchanged keeps its original contour; only boxes the user actually moved,
    resized or created are downgraded to a rectangular contour.

    Args:
        boxes: Boxes from the UI, each with ``x``/``y``/``w``/``h`` and optional ``id``.
        original_lines: Lines from ``detect()``, used to look up untouched contours.

    Returns:
        Lines sorted top-to-bottom by vertical centre.
    """
    by_id = {str(line.guid): line for line in original_lines}
    # Geometry is the fallback identity: a UI that cannot round-trip an id (or
    # that mangles it) still gets contour preservation for untouched boxes.
    by_geom = {
        (line.bbox.x, line.bbox.y, line.bbox.w, line.bbox.h): line
        for line in original_lines
    }
    rebuilt: List[Line] = []

    for box in boxes:
        x, y = int(box["x"]), int(box["y"])
        w, h = int(box["w"]), int(box["h"])
        if w <= 0 or h <= 0:
            continue

        original = by_id.get(str(box.get("id")))
        unchanged = original is not None and (
            original.bbox.x == x
            and original.bbox.y == y
            and original.bbox.w == w
            and original.bbox.h == h
        )
        if unchanged:
            rebuilt.append(original)  # untouched: keep the precise contour
            continue

        by_geometry = by_geom.get((x, y, w, h))
        if by_geometry is not None:
            rebuilt.append(by_geometry)
            continue

        guid = original.guid if original is not None else generate_guid(23)
        rebuilt.append(
            Line(
                guid=guid,
                contour=_rect_contour(x, y, w, h),
                bbox=BBox(x, y, w, h),
                center=(x + w // 2, y + h // 2),
            )
        )

    # The user's boxes are authoritative, so re-sort by position rather than
    # re-running the detector's grouping heuristics.
    rebuilt.sort(key=lambda line: (line.center[1], line.center[0]))
    return rebuilt


# --------------------------------------------------------------------------- #
# Stage 2: recognition
# --------------------------------------------------------------------------- #

_ocr_engines: Dict[str, OCRInference] = {}
_converter = pyewts.pyewts()


def _get_ocr_engine(
    model_dir: str | None, providers: Sequence[str] | None = None
) -> OCRInference:
    """Return a cached OCRInference session, keyed by directory and providers."""
    if model_dir is None:
        from models import ocr_model_dir

        model_dir = ocr_model_dir()
    key = f"{model_dir}:{','.join(providers) if providers else 'default'}"
    if key not in _ocr_engines:
        model = import_local_model(model_dir)
        if model is None:
            raise ValueError(f"no model_config.json found in {model_dir!r}")
        _ocr_engines[key] = OCRInference(
            get_platform(), model.config, providers=list(providers) if providers else None
        )
    return _ocr_engines[key]


def crop_lines(
    page_image: npt.NDArray,
    lines: Sequence[Line],
    k_factor: float = 2.5,
    bbox_tolerance: float = 4.0,
) -> List[npt.NDArray]:
    """Crop line images from a deskewed page. Useful for previewing a bad box."""
    if not lines:
        return []
    return extract_line_images(page_image, list(lines), k_factor, bbox_tolerance)


def line_previews(
    page_image: npt.NDArray, lines: Sequence[Line], pad: int = 6
) -> List[npt.NDArray]:
    """Plain rectangular crops of each line, for showing to a person.

    Distinct from :func:`crop_lines`, which masks everything outside the line's
    contour -- correct for feeding the recogniser, but it leaves black wedges
    around slanted text that read as damage when a human looks at the strip.
    """
    height, width = page_image.shape[:2]
    previews = []
    for line in lines:
        box = line.bbox
        top, bottom = max(0, box.y - pad), min(height, box.y + box.h + pad)
        left, right = max(0, box.x - pad), min(width, box.x + box.w + pad)
        previews.append(page_image[top:bottom, left:right])
    return previews


def ocr(
    page_image: npt.NDArray,
    lines: Sequence[Line],
    ocr_model: str | None = None,
    k_factor: float = 2.5,
    bbox_tolerance: float = 4.0,
    target_encoding: Encoding = Encoding.Unicode,
    providers: Sequence[str] | None = None,
) -> List[str]:
    """Recognise text for each line, in the order given.

    Args:
        page_image: The deskewed page from ``detect()``.
        lines: Lines to recognise, typically from ``boxes_to_lines()``.
        ocr_model: Directory holding ``model_config.json`` and the ONNX model;
            defaults to the Hub copy of the default model.
        k_factor: Vertical scaling applied when cropping each line.
        bbox_tolerance: Height tolerance used during cropping.
        target_encoding: Unicode or Wylie output.

    Returns:
        One string per input line, aligned by index.
    """
    if not lines:
        return []

    engine = _get_ocr_engine(ocr_model, providers)
    line_images = crop_lines(page_image, lines, k_factor, bbox_tolerance)

    results: List[str] = []
    for line_image in line_images:
        text = engine.run(line_image).strip().replace("§", " ")

        encoder = engine.config.encoder
        if encoder == CharsetEncoder.Wylie and target_encoding == Encoding.Unicode:
            text = _converter.toUnicode(text)
        elif encoder == CharsetEncoder.Stack and target_encoding == Encoding.Wylie:
            text = _converter.toWylie(text)

        results.append(text)

    return results


def available_ocr_models() -> List[str]:
    """Names of OCR models this app can fetch from the Hub."""
    from models import OCR_MODEL_REPOS

    return sorted(OCR_MODEL_REPOS)
