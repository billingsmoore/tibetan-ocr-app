---
title: Tibetan Page Transcription
emoji: 📜
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.22.0
app_file: app.py
pinned: false
license: mit
models:
  - BDRC/PhotiLines
  - BDRC/Woodblock
  - billingsmoore/mlotsawa-ground-base
---

# Tibetan Page Transcription

Upload a page of Tibetan text, **correct the detected lines by hand**, then
transcribe and edit the result.

Line detection is good but not perfect: it boxes smudges in the margin, and
occasionally splits or misses a line. A wrong box produces wrong text silently,
so this app deliberately splits the pipeline in two and puts an editable overlay
between the halves.

1. **Upload** a page image.
2. **Detect lines** — boxes appear over the deskewed page. Tick *Flatten curved
   page* first if the photo bows.
3. **Correct them** — drag to draw a missing line, use the handles to resize,
   select and press Delete to remove a spurious one.
4. **Transcribe** — only the boxes you confirmed are recognised.
5. **Translate** any line into English, and edit that too.
6. **Download** the transcription, the translation, or an interlinear PDF
   pairing each line's image with both.

## Layout

```
app.py           Gradio UI and step handlers
pipeline.py      detect() / dewarp() / ocr() — the interactive split
models.py        lazy model fetching from the Hub
translate.py     Tibetan -> English seq2seq translation
export.py        txt, docx, pdf and interlinear-pdf writers
Config.py        string -> enum tables
BDRC/            upstream inference code (6 modules, unmodified except as noted)
Assets/Fonts/    TibMachUni, embedded into generated PDFs
vendor/pyewts/   vendored EWTS <-> Unicode converter
examples/        a real pecha page, plus a synthetic sample and its generator
```

## How it works

A fork of BDRC's [tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app),
reduced to its inference core. Upstream's `OCRPipeline.run_ocr()` runs detection,
cropping and recognition in one call, leaving nowhere to intervene; `pipeline.py`
cuts it at its natural seam — the sorted list of `Line` objects — into `detect()`
and `ocr()`.

Three details make the round-trip work:

- **Cropping uses contours, not boxes.** `extract_line_images` crops through
  `line.contour`, which hugs the glyph shape and handles slanted text better
  than a rectangle. So a box you *didn't* touch keeps its original contour;
  only boxes you moved, resized or drew become rectangles.
- **Detection runs downscaled, cropping runs full-size.** The detector pads its
  input to a whole grid of 512px tiles, so cost is quantised by tile count, not
  pixels. A 3500×912 page is 14 tiles; at half scale it is 4, a ~3.5× speedup
  that finds the same lines. Boxes are then projected back to full resolution,
  because the recogniser wants ~100px-tall text. This is exact, not approximate:
  deskew rotates about the image centre at unchanged size, so scaling commutes
  with it.
- **Dewarping is a separate preprocessing step,** not a flag inside `detect()`.
  A non-linear warp cannot be projected back from a downscaled copy the way a
  rotation can, so `dewarp()` runs at full resolution and detection simply runs
  afterwards on its output. It only fires when a line's vertical deviation
  exceeds its own height.

### Changes to upstream code

`BDRC/` is upstream's, with two deliberate edits:

- `Data.py`, `Utils.py` — Qt imports made optional, so the inference code runs
  headless without PySide6.
- `image_dewarping.py` — fixed `run_tps`, which called `npt.NDArray(...)` as if
  it were a constructor (it is a typing alias, so every call raised) and applied
  its corner scaling twice. Dewarping could not have worked upstream; `run_ocr`
  swallowed the exception into a generic "Line processing failed". On a page
  bowed by 90px this now removes 85% of the distortion.

## Models

No weights are stored in this repository. They are fetched from the Hub on first
use and cached:

| Purpose | Model |
|---|---|
| Line segmentation | [BDRC/PhotiLines](https://huggingface.co/BDRC/PhotiLines) |
| OCR | [BDRC/Woodblock](https://huggingface.co/BDRC/Woodblock) |
| Translation | [billingsmoore/mlotsawa-ground-base](https://huggingface.co/billingsmoore/mlotsawa-ground-base) |

`BDRC/Woodblock` is currently the only BDRC OCR repo whose `model_config.json`
carries every key the config reader needs — the others omit `add_blank`, and
guessing that flag wrong corrupts CTC decoding into plausible-looking garbage
rather than raising. Add them to `OCR_MODEL_REPOS` in `models.py` once their
configs are complete.

## Running locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py          # http://localhost:7860
```

`verify_pipeline.py` is a smoke test: it runs the full chain on
`examples/I1ER9510006.jpg` and writes an overlay image plus the boxes as JSON.

**Load sample page** in the app uses `examples/sample_page.png`, a synthetic
folio of the refuge and bodhicitta verse. It exists because the real example is
a dense woodblock print that is slow to work through: the sample detects in ~6s
and transcribes in ~2s against ~13s and ~10s, and its short lines translate
quickly too. Regenerate or edit it with `python examples/make_sample.py`.

The text is deliberately ubiquitous, so output is easy to check by eye. The
recogniser reads it near-perfectly despite being trained on woodblock prints:
three of five lines exact, one missing only the ornamental `༄༅།` head mark, one
differing by a single near-identical diacritic.

## Licensing

MIT, inherited from upstream. Vendored and downloaded components carry their own
terms — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
