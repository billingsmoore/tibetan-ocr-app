# Third-party notices

## tibetan-ocr-app (this repository)

Forked from [buda-base/tibetan-ocr-app](https://github.com/buda-base/tibetan-ocr-app),
© 2024 Buddhist Digital Resource Center, MIT License. The `BDRC/` package and
`Config.py` are upstream code; see `LICENSE`. Modifications are listed in the
README.

## pyewts (vendored, `vendor/pyewts/`)

[pyewts](https://github.com/Esukhia/pyewts) by the Esukhia development team,
Apache License 2.0. Used for EWTS (Wylie) ↔ Unicode conversion.

Vendored rather than installed from PyPI because its `setup.py` imports
`pkg_resources` — removed in setuptools 81 — so it cannot be built in a current
pip environment. The source is unmodified.

## Tibetan Machine Uni (`Assets/Fonts/TibMachUni-1.901b.ttf`)

Tibetan Machine Uni font, released under the GNU General Public License with
font exception. Embedded in generated PDFs so Tibetan renders on systems without
a Tibetan font installed.

## Models (downloaded at runtime, not redistributed here)

Fetched from the Hugging Face Hub on first use:

- [BDRC/PhotiLines](https://huggingface.co/BDRC/PhotiLines) — line segmentation, CC-BY-4.0
- [BDRC/Woodblock](https://huggingface.co/BDRC/Woodblock) — Tibetan OCR

Both © Buddhist Digital Resource Center. This repository ships no model weights.
