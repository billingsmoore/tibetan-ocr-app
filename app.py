"""
Interactive Tibetan page OCR: detect lines, correct them by hand, then transcribe.

The point of this app is the pause in the middle. Line detection is good but not
perfect -- it invents boxes on margin smudges and occasionally splits or misses a
line -- and a wrong box silently produces wrong text. So detection and
recognition run as two separate steps with an editable box overlay between them.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from gradio_image_annotation import image_annotator

import export
import models
import pipeline

try:  # Only present on Hugging Face ZeroGPU Spaces.
    import spaces

    on_gpu = spaces.GPU(duration=120)
except ImportError:  # pragma: no cover - local and CPU deployments

    def on_gpu(fn):
        """No-op stand-in so the same code runs off-Space."""
        return fn


BOX_COLOR = (37, 150, 190)
MAX_UPLOAD_PIXELS = 40_000_000  # ~40MP; bigger inputs are downscaled on ingest


# --------------------------------------------------------------------------- #
# Conversions between the annotator's box dicts and pipeline boxes
# --------------------------------------------------------------------------- #


def _to_annotator_boxes(boxes: List[dict]) -> List[dict]:
    """pipeline x/y/w/h -> annotator xmin/ymin/xmax/ymax."""
    return [
        {
            "xmin": b["x"],
            "ymin": b["y"],
            "xmax": b["x"] + b["w"],
            "ymax": b["y"] + b["h"],
            "label": str(i + 1),
            "color": BOX_COLOR,
        }
        for i, b in enumerate(boxes)
    ]


def _from_annotator_boxes(boxes: List[dict], ids: List[str]) -> List[dict]:
    """annotator xmin/ymin/xmax/ymax -> pipeline x/y/w/h.

    The annotator drops any identity we attach, so a box is matched back to its
    original detected line by its label, which we set to the 1-based index. Boxes
    the user drew have no usable label and fall through with id=None, which
    ``boxes_to_lines`` treats as new.
    """
    result = []
    for box in boxes:
        x0, x1 = sorted((int(box["xmin"]), int(box["xmax"])))
        y0, y1 = sorted((int(box["ymin"]), int(box["ymax"])))

        box_id = None
        label = str(box.get("label", "")).strip()
        if label.isdigit() and 1 <= int(label) <= len(ids):
            box_id = ids[int(label) - 1]

        result.append({"id": box_id, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return result


def _clamp_upload(image: np.ndarray) -> np.ndarray:
    """Downscale absurdly large uploads so a single request can't exhaust RAM."""
    height, width = image.shape[:2]
    if height * width <= MAX_UPLOAD_PIXELS:
        return image
    factor = (MAX_UPLOAD_PIXELS / (height * width)) ** 0.5
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# GPU-side work
#
# On ZeroGPU a device is attached only for the duration of an @spaces.GPU call,
# so the ONNX sessions -- which bind to the device when they are built -- have to
# be created inside these functions rather than at import. Model *downloads* are
# resolved by the callers beforehand, so no GPU time is spent on network I/O.
# --------------------------------------------------------------------------- #


@on_gpu
def _detect_on_gpu(bgr: np.ndarray, flatten: bool, line_model: str):
    """Optionally flatten the page, then detect its lines."""
    changed = None
    if flatten:
        bgr, changed = pipeline.dewarp(bgr, line_model=line_model)
    page = pipeline.detect(bgr, line_model=line_model)
    return page.image, page.angle, page.lines, changed


@on_gpu
def _ocr_on_gpu(image: np.ndarray, lines, ocr_model: str):
    """Recognise the given lines."""
    return pipeline.ocr(image, lines, ocr_model=ocr_model)


# --------------------------------------------------------------------------- #
# Step handlers
# --------------------------------------------------------------------------- #


def detect_lines(
    annotation: Optional[Dict[str, Any]],
    flatten: bool,
    progress=gr.Progress(),
):
    """Run line detection and hand back an editable overlay."""
    if not annotation or annotation.get("image") is None:
        raise gr.Error("Upload a page image first.")

    progress(0.1, desc="Preparing page")
    rgb = _clamp_upload(np.asarray(annotation["image"]))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    progress(0.2, desc="Fetching model" if flatten else "Preparing")
    line_model = models.line_model_path()  # outside the GPU call: no device time

    progress(0.4, desc="Detecting lines")
    try:
        image, angle, lines, changed = _detect_on_gpu(bgr, flatten, line_model)
    except ValueError as exc:
        raise gr.Error(f"Line detection failed: {exc}")

    note = ""
    if changed is not None:
        note = " Page was flattened." if changed else " Page looked flat already."

    boxes = pipeline.lines_to_boxes(lines)
    state = {"image": image, "lines": lines, "ids": [b["id"] for b in boxes]}

    progress(1.0, desc="Done")
    return (
        {
            "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "boxes": _to_annotator_boxes(boxes),
        },
        state,
        f"Found {len(boxes)} lines, deskewed by {angle:.2f}°.{note} "
        "Correct the boxes, then transcribe.",
    )


def run_ocr(
    annotation: Optional[Dict[str, Any]],
    state: Optional[Dict[str, Any]],
    model_name: str,
    progress=gr.Progress(),
):
    """Transcribe the lines exactly as the user has left them."""
    if not state:
        raise gr.Error("Detect lines first.")
    if not annotation or not annotation.get("boxes"):
        raise gr.Error("No boxes to transcribe -- detect lines or draw some.")

    progress(0.1, desc="Reading boxes")
    boxes = _from_annotator_boxes(annotation["boxes"], state["ids"])
    lines = pipeline.boxes_to_lines(boxes, state["lines"])
    if not lines:
        raise gr.Error("Every box was empty; nothing to transcribe.")

    ocr_model = models.ocr_model_dir(model_name)  # outside the GPU call
    progress(0.3, desc=f"Transcribing {len(lines)} lines")
    texts = _ocr_on_gpu(state["image"], lines, ocr_model)

    progress(1.0, desc="Done")
    return "\n".join(texts), f"Transcribed {len(texts)} lines. Edit freely below."


def build_download(text: str, fmt: str):
    """Write the edited transcription to a temp file for download."""
    if not text.strip():
        raise gr.Error("Nothing to download yet.")
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{fmt}", prefix="transcription_"
    )
    tmp.close()
    return export.export(text, fmt, tmp.name)


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

INTRO = """
# Tibetan Page Transcription

**1.** Upload a page &nbsp;→&nbsp; **2.** Detect lines &nbsp;→&nbsp; **3.** Fix the
boxes &nbsp;→&nbsp; **4.** Transcribe &nbsp;→&nbsp; **5.** Edit and download.

Drag to draw a missing line, drag its handles to resize, select and press
Delete to remove one. Detection often boxes a smudge in the margin -- deleting
those before transcribing keeps the output clean.
"""

with gr.Blocks(title="Tibetan Page Transcription") as demo:
    page_state = gr.State()

    gr.Markdown(INTRO)

    with gr.Row():
        detect_btn = gr.Button("Detect lines", variant="primary")
        ocr_btn = gr.Button("Transcribe these lines", variant="primary")
        flatten_cb = gr.Checkbox(
            label="Flatten curved page",
            info="For photographed pages that bow. Slower.",
            value=False,
            scale=0,
        )
        model_dd = gr.Dropdown(
            choices=pipeline.available_ocr_models(),
            value=models.DEFAULT_OCR_MODEL,
            label="OCR model",
            scale=0,
        )

    status = gr.Markdown("Upload a page image to begin.")

    with gr.Row():
        annotator = image_annotator(
            label="Page (detected lines)",
            image_type="numpy",
            sources=["upload", "clipboard"],
            box_min_size=5,
            handle_size=10,
            box_thickness=2,
            disable_edit_boxes=True,  # boxes carry index labels; keep them stable
            show_remove_button=True,  # deleting a spurious box is a core action
            show_download_button=False,
            show_share_button=False,
            scale=3,
        )
        with gr.Column(scale=2):
            transcription = gr.Textbox(
                label="Transcription (editable)",
                lines=22,
                max_lines=40,
                buttons=["copy"],
                placeholder="Transcribed text appears here, one row per line.",
            )
            with gr.Row():
                fmt = gr.Radio(
                    choices=["txt", "docx", "pdf"], value="txt", label="Format", scale=2
                )
                download_btn = gr.Button("Prepare download", scale=1)
            download_file = gr.File(label="Download", interactive=False)

    detect_btn.click(
        detect_lines,
        inputs=[annotator, flatten_cb],
        outputs=[annotator, page_state, status],
    )
    ocr_btn.click(
        run_ocr,
        inputs=[annotator, page_state, model_dd],
        outputs=[transcription, status],
    )
    download_btn.click(
        build_download, inputs=[transcription, fmt], outputs=[download_file]
    )


if __name__ == "__main__":
    # Warm the Hub cache so the first user doesn't pay the model download.
    try:
        models.prefetch()
    except Exception as exc:  # network hiccup shouldn't stop the app booting
        print(f"[warn] model prefetch failed, will fetch on demand: {exc}")

    demo.queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860))
    )
