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
import translate

try:  # Only present on Hugging Face ZeroGPU Spaces.
    import spaces

    HAS_ZEROGPU = True
except ImportError:  # pragma: no cover - local and CPU deployments
    spaces = None
    HAS_ZEROGPU = False

# ZeroGPU quota is finite and shared. When it runs out the call raises rather
# than queueing, so every GPU entry point has an undecorated twin that reruns
# the same work pinned to CPU -- slower, but the app keeps working.
CPU_PROVIDERS = ["CPUExecutionProvider"]

_QUOTA_HINTS = ("quota", "exceeded", "gpu task aborted", "no gpu is currently available")


def _is_quota_error(exc: BaseException) -> bool:
    """True when a GPU call failed for capacity reasons rather than a real bug."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _QUOTA_HINTS)


def _with_cpu_fallback(gpu_fn, cpu_fn, gpu_extra=(None,), cpu_extra=(CPU_PROVIDERS,)):
    """Try the GPU variant; on quota exhaustion rerun on CPU.

    ``gpu_extra``/``cpu_extra`` are appended to the call, which is how the ONNX
    paths pin their execution providers. Torch-based work passes empty tuples
    and picks its own device.

    Returns ``(result, used_cpu)`` so the UI can say which path ran.
    """

    def run(*args):
        if not HAS_ZEROGPU:
            # No GPU in this environment at all -- CPU is simply the normal path,
            # with default providers, so behaviour off-Space is unchanged.
            return cpu_fn(*args, *(None,) * len(gpu_extra)), False
        try:
            return gpu_fn(*args, *gpu_extra), False
        except Exception as exc:  # noqa: BLE001 - re-raised unless it's quota
            if not _is_quota_error(exc):
                raise
            print(f"[zerogpu] quota exhausted, falling back to CPU: {exc}")
            return cpu_fn(*args, *cpu_extra), True

    return run


BOX_COLOR = (37, 150, 190)
MAX_UPLOAD_PIXELS = 40_000_000  # ~40MP; bigger inputs are downscaled on ingest

INTERLINEAR = "Interlinear (image + Tibetan + English)"
TRANSCRIPTION_ONLY = "Transcription only"
TRANSLATION_ONLY = "Translation only"


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


def _detect_work(bgr: np.ndarray, flatten: bool, line_model: str, providers):
    """Optionally flatten the page, then detect its lines."""
    changed = None
    if flatten:
        bgr, changed = pipeline.dewarp(bgr, line_model=line_model, providers=providers)
    page = pipeline.detect(bgr, line_model=line_model, providers=providers)
    active = pipeline.line_session_providers(line_model, providers=providers)
    return page.image, page.angle, page.lines, changed, active


def _ocr_work(image: np.ndarray, lines, ocr_model: str, providers):
    """Recognise the given lines."""
    return pipeline.ocr(image, lines, ocr_model=ocr_model, providers=providers)


def _translate_work(texts, device):
    """Translate Tibetan lines to English."""
    return translate.translate_batch(texts, device=device)


# Detection and recognition are ONNX, and the session reports
# CPUExecutionProvider on this Space even inside a GPU allocation -- CUDA never
# initialises for them. Wrapping them in @spaces.GPU therefore reserved quota to
# do CPU work, which is what starved translation. They now run on CPU directly.
ONNX_PROVIDERS = CPU_PROVIDERS if HAS_ZEROGPU else None

if HAS_ZEROGPU:
    # Translation is torch, genuinely uses the device, and satisfies ZeroGPU's
    # requirement that a Space declare at least one GPU function.
    _translate_gpu = spaces.GPU(duration=60)(_translate_work)
else:  # pragma: no cover - the dispatcher never calls this off-Space
    _translate_gpu = None


def run_detect(*args):
    return _detect_work(*args, ONNX_PROVIDERS), False


def run_ocr_work(*args):
    return _ocr_work(*args, ONNX_PROVIDERS), False


# On fallback the device is pinned rather than autodetected: see translate.py.
run_translate = _with_cpu_fallback(
    _translate_gpu, _translate_work, gpu_extra=(None,), cpu_extra=("cpu",)
)


def _translate_many(texts: List[str], progress) -> tuple:
    """Translate several lines, reporting progress where it is slow.

    On the GPU everything goes in one call: it finishes quickly, and each call
    reserves quota up front so a batch is much cheaper than one call per line.
    On CPU a line takes tens of seconds, so there the loop runs one at a time
    and reports which line it is on -- otherwise the button looks dead.
    """
    if HAS_ZEROGPU:
        try:
            progress(0.15, desc=f"Translating {len(texts)} lines on GPU")
            return _translate_gpu(texts, None), False
        except Exception as exc:  # noqa: BLE001 - re-raised unless it's quota
            if not _is_quota_error(exc):
                raise
            print(f"[zerogpu] quota exhausted, falling back to CPU: {exc}")

    device = "cpu" if HAS_ZEROGPU else None
    results: List[str] = []
    for index, text in enumerate(texts):
        progress(
            index / max(len(texts), 1),
            desc=f"Translating line {index + 1} of {len(texts)} on CPU",
        )
        results.append(translate.translate_batch([text], device=device)[0])
    return results, HAS_ZEROGPU


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
        (image, angle, lines, changed, active), used_cpu = run_detect(
            bgr, flatten, line_model
        )
    except ValueError as exc:
        raise gr.Error(f"Line detection failed: {exc}")

    note = ""
    if changed is not None:
        note = " Page was flattened." if changed else " Page looked flat already."
    if used_cpu:
        note += " (GPU quota spent — ran on CPU, slower.)"
    elif HAS_ZEROGPU:
        on = "GPU" if any("CUDA" in p or "Tensorrt" in p for p in active) else "CPU"
        note += f" (ran on {on}: {active[0]})"

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
    texts, used_cpu = run_ocr_work(state["image"], lines, ocr_model)

    # Crop the same lines again on the CPU so each text row can show the strip of
    # page it came from. Cropping is cheap next to recognition, and this keeps
    # large image data out of the GPU call's return value.
    progress(0.9, desc="Preparing line previews")
    crops = pipeline.line_previews(state["image"], lines)
    rows = [
        {"text": text, "target": "", "crop": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)}
        for text, crop in zip(texts, crops)
    ]

    progress(1.0, desc="Done")
    note = " (GPU quota spent — ran on CPU, slower.)" if used_cpu else ""
    return rows, list(texts), f"Transcribed {len(texts)} lines.{note} Edit freely below."


SAMPLE_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "examples", "sample_page.png")


def load_sample():
    """Put the synthetic sample page into the annotator."""
    image = cv2.imread(SAMPLE_PAGE)
    if image is None:
        raise gr.Error("Sample page is missing from this deployment.")
    return {"image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "boxes": []}


def translate_line(text: str, index: int, translations: Optional[List[str]]):
    """Translate one line and store the result alongside the others."""
    if not (text or "").strip():
        raise gr.Error("Nothing to translate on this line.")

    # Immediate acknowledgement; a line takes tens of seconds on CPU.
    yield gr.skip(), gr.skip(), f"Translating line {index + 1}…"

    try:
        results, used_cpu = run_translate([text])
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
        raise gr.Error(f"Translation failed: {exc}")

    english = results[0] if results else ""
    updated = list(translations or [])
    while len(updated) <= index:
        updated.append("")
    updated[index] = english

    note = " (GPU quota spent — ran on CPU.)" if used_cpu else ""
    yield english, updated, f"Translated line {index + 1}.{note}"


def translate_all(
    edits: Optional[List[str]],
    rows: Optional[List[dict]],
    translations: Optional[List[str]],
    progress=gr.Progress(),
):
    """Translate every line that doesn't have a translation yet.

    Lines already translated -- by the per-line button, or by hand -- are left
    alone, so this fills gaps rather than overwriting work. The untranslated
    lines go in a single batched call: on ZeroGPU each call reserves quota up
    front, so one batch costs far less than one call per line.
    """
    if not rows:
        raise gr.Error("Transcribe the page first.")

    texts = list(edits or [row.get("text", "") for row in rows])
    while len(texts) < len(rows):
        texts.append(rows[len(texts)].get("text", ""))

    current = list(translations or [])
    while len(current) < len(texts):
        current.append("")

    pending = [i for i, text in enumerate(texts) if text.strip() and not current[i].strip()]
    if not pending:
        yield rows, texts, current, "Every line already has a translation."
        return

    # Say something before the first line is done: on CPU each takes tens of
    # seconds, and without this the click looks like it did nothing. gr.skip()
    # leaves the row state alone so this does not re-render the whole list.
    yield (
        gr.skip(),
        gr.skip(),
        gr.skip(),
        f"Translating {len(pending)} lines… this can take a while on CPU.",
    )

    try:
        results, used_cpu = _translate_many([texts[i] for i in pending], progress)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as-is
        raise gr.Error(f"Translation failed: {exc}")

    for slot, index in enumerate(pending):
        if slot < len(results):
            current[index] = results[slot]

    refreshed = [
        {**row, "text": texts[i], "target": current[i]} for i, row in enumerate(rows)
    ]

    progress(1.0, desc="Done")
    note = " (GPU quota spent — ran on CPU.)" if used_cpu else ""
    yield refreshed, texts, current, f"Translated {len(pending)} lines.{note}"


def build_download(
    edits: Optional[List[str]],
    translations: Optional[List[str]],
    rows: Optional[List[dict]],
    content: str,
    fmt: str,
):
    """Write the chosen content to a temp file for download."""
    if content == INTERLINEAR:
        if not rows:
            raise gr.Error("Transcribe the page first.")
        items = [
            {
                "crop": row.get("crop"),
                "source": (edits or [])[i] if i < len(edits or []) else row.get("text", ""),
                "target": (translations or [])[i] if i < len(translations or []) else "",
            }
            for i, row in enumerate(rows)
        ]
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf", prefix="interlinear_"
        )
        tmp.close()
        return export.to_interlinear_pdf(items, tmp.name)

    if content == TRANSLATION_ONLY:
        lines, prefix = translations or [], "translation_"
    else:
        lines, prefix = edits or [], "transcription_"

    text = "\n".join(lines)
    if not text.strip():
        raise gr.Error("Nothing to download yet.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}", prefix=prefix)
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

No page to hand? **Load sample page** gives you a short, clean one that runs
through the whole pipeline quickly.
"""

# A pecha page is roughly 4:1, so its displayed height is width/4 -- the page
# needs the full column width to be legible, and a fixed height alone would do
# nothing. The drag handle lets you trade height against the rest of the page.
CSS = """
#page-pane { resize: vertical; overflow: auto; min-height: 520px; flex: 0 0 auto; }
#page-pane img { image-rendering: -webkit-optimize-contrast; }

/* The overlay boxes are positioned against the rendered image, so any resize of
   this pane drags them along with it. Mounting the transcription rows below
   forces exactly that reflow, and the default transition animates the squeeze --
   which reads as the boxes mysteriously shrinking away after transcribing.
   Pin the pane and let it change size only when the drag handle is used. */
#page-pane, #page-pane * { transition: none !important; animation: none !important; }

/* Each text field sits directly under the strip of page it came from, so the
   correspondence needs no clicking to discover. */
.line-strip img { object-fit: contain; width: 100%; background: transparent; }
.line-strip { margin-bottom: 2px; opacity: 0.85; }
.line-strip:hover { opacity: 1; }
.translation-field textarea { font-style: italic; opacity: 0.9; }
"""

CREDITS = """
### Acknowledgements

This app is a thin interactive layer over models and code created by others.

| | |
|---|---|
| **OCR pipeline & models** | [Buddhist Digital Resource Center](https://www.bdrc.io) — forked from [tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app) (MIT). Line segmentation by [BDRC/PhotiLines](https://huggingface.co/BDRC/PhotiLines), recognition by [BDRC/Woodblock](https://huggingface.co/BDRC/Woodblock). Models trained on transcriptions from BDRC, [ALL](https://asianlegacylibrary.org/), [Adarsha](https://adarshah.org/) and [NorbuKetaka](http://purl.bdrc.io/resource/PR1ER1). |
| **Translation model** | [billingsmoore/mlotsawa-ground-base](https://huggingface.co/billingsmoore/mlotsawa-ground-base), a Tibetan→English seq2seq model from the [MLotsawa](https://github.com/billingsmoore/MLotsawa) project |
| **Box editor** | [gradio-image-annotation](https://github.com/edgarGracia/gradio_image_annotator) by Edgar Gracia (MIT) |
| **Wylie ↔ Unicode** | [pyewts](https://github.com/Esukhia/pyewts) by the Esukhia development team (Apache-2.0) |
| **Tibetan font** | Tibetan Machine Uni, embedded in exported PDFs (GPL with font exception) |
| **Framework & libraries** | [Gradio](https://gradio.app) (Apache-2.0), [ONNX Runtime](https://onnxruntime.ai) (MIT), [OpenCV](https://opencv.org) (Apache-2.0), [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) (Apache-2.0), [thin-plate-spline](https://pypi.org/project/thin-plate-spline/) (MIT), [Transformers](https://github.com/huggingface/transformers) (Apache-2.0), [PyTorch](https://pytorch.org) (BSD), [python-docx](https://github.com/python-openxml/python-docx) (MIT), [ReportLab](https://www.reportlab.com) (BSD) |

Full terms in [THIRD_PARTY_NOTICES.md](https://github.com/billingsmoore/tibetan-ocr-app/blob/interactive-ui/THIRD_PARTY_NOTICES.md).
The heavy lifting here is BDRC's; please credit them in any work that uses this.
"""

with gr.Blocks(title="Tibetan Page Transcription") as demo:
    page_state = gr.State()

    gr.Markdown(INTRO)

    with gr.Row():
        sample_btn = gr.Button("Load sample page")
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

    annotator = image_annotator(
        label="Page (detected lines) — drag the bottom edge to resize",
        elem_id="page-pane",
        image_type="numpy",
        sources=["upload", "clipboard"],
        box_min_size=5,
        handle_size=10,
        box_thickness=2,
        disable_edit_boxes=True,  # boxes carry index labels; keep them stable
        show_remove_button=True,  # deleting a spurious box is a core action
        show_download_button=False,
        show_share_button=False,
        height=560,
    )

    with gr.Row():
        gr.Markdown("### Transcription", container=False)
        translate_all_btn = gr.Button("Translate all", size="sm", scale=0)

    # Two separate states on purpose. `rows` triggers the re-render below, so it
    # must only change when OCR runs; `edits` holds what you type. If typing fed
    # back into a render input, every keystroke would rebuild the components and
    # steal focus mid-word.
    rows_state = gr.State([])
    edits_state = gr.State([])
    trans_state = gr.State([])

    @gr.render(inputs=rows_state)
    def render_transcription(rows):
        if not rows:
            gr.Markdown("_Detect lines and transcribe to see the text here._")
            return

        for index, row in enumerate(rows):
            with gr.Group():
                gr.Image(
                    value=row["crop"],
                    show_label=False,
                    interactive=False,
                    container=False,
                    height=54,
                    elem_classes=["line-strip"],
                )
                with gr.Row():
                    box = gr.Textbox(
                        value=row["text"],
                        show_label=False,
                        container=False,
                        lines=1,
                        max_lines=4,
                        autoscroll=False,
                        # Explicit: inside gr.render Gradio does not infer
                        # interactivity from usage, so a Textbox given a value
                        # renders read-only unless told otherwise.
                        interactive=True,
                        scale=9,
                    )
                    translate_btn = gr.Button("Translate", size="sm", scale=1)
                english = gr.Textbox(
                    # Read back from the row so a re-render (Translate all)
                    # keeps translations already produced.
                    value=row.get("target", ""),
                    show_label=False,
                    container=False,
                    lines=1,
                    max_lines=4,
                    autoscroll=False,
                    interactive=True,
                    placeholder="English translation",
                    elem_classes=["translation-field"],
                )

            def save(new_text, edits, idx=index):
                edits = list(edits or [])
                while len(edits) <= idx:
                    edits.append("")
                edits[idx] = new_text
                return edits

            def save_translation(new_text, translations, idx=index):
                translations = list(translations or [])
                while len(translations) <= idx:
                    translations.append("")
                translations[idx] = new_text
                return translations

            box.change(save, inputs=[box, edits_state], outputs=[edits_state])
            english.change(
                save_translation, inputs=[english, trans_state], outputs=[trans_state]
            )
            # `yield from` keeps this a generator *function*, which is what
            # Gradio checks for; a lambda returning a generator would not stream.
            def do_translate(text, translations, idx=index):
                yield from translate_line(text, idx, translations)

            translate_btn.click(
                do_translate,
                inputs=[box, trans_state],
                outputs=[english, trans_state, status],
            )

    with gr.Row():
        content = gr.Radio(
            choices=[INTERLINEAR, TRANSCRIPTION_ONLY, TRANSLATION_ONLY],
            value=TRANSCRIPTION_ONLY,
            label="Content",
            scale=3,
        )
        fmt = gr.Radio(
            choices=["txt", "docx", "pdf"], value="txt", label="Format", scale=2
        )
        download_btn = gr.Button("Prepare download", scale=1)
        download_file = gr.File(label="Download", interactive=False, scale=2)

    def _format_visibility(choice):
        # Interlinear embeds the line strips, so it only makes sense as a PDF.
        return gr.update(visible=choice != INTERLINEAR)

    content.change(_format_visibility, inputs=[content], outputs=[fmt])

    sample_btn.click(load_sample, outputs=[annotator])
    detect_btn.click(
        detect_lines,
        inputs=[annotator, flatten_cb],
        outputs=[annotator, page_state, status],
    )
    ocr_btn.click(
        run_ocr,
        inputs=[annotator, page_state, model_dd],
        outputs=[rows_state, edits_state, status],
    )
    translate_all_btn.click(
        translate_all,
        inputs=[edits_state, rows_state, trans_state],
        outputs=[rows_state, edits_state, trans_state, status],
    )
    download_btn.click(
        build_download,
        inputs=[edits_state, trans_state, rows_state, content, fmt],
        outputs=[download_file],
    )

    gr.Markdown(CREDITS)


if __name__ == "__main__":
    # Warm the Hub cache so the first user doesn't pay the model download.
    try:
        models.prefetch()
    except Exception as exc:  # network hiccup shouldn't stop the app booting
        print(f"[warn] model prefetch failed, will fetch on demand: {exc}")

    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        css=CSS,  # Gradio 6 takes css here, not on Blocks
    )
