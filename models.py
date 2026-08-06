"""
Lazy model resolution against the Hugging Face Hub.

Nothing in this repository ships model weights. The line-segmentation and OCR
models are pulled from the Hub the first time they are actually needed and then
served from the local Hub cache, which keeps the Space repo small and avoids
redistributing BDRC's weights ourselves.

Set ``HF_HOME`` to control where the cache lives. On a Space the default
(``~/.cache/huggingface``) is fine but ephemeral -- models are re-fetched after a
cold start, which is why :func:`prefetch` exists.
"""

from __future__ import annotations

import os
from typing import Dict

from huggingface_hub import hf_hub_download

# Line segmentation. v1 is a single self-contained .onnx; v2 splits its weights
# into an .onnx.data sidecar, which complicates caching for no measured gain.
LINE_MODEL_REPO = "BDRC/PhotiLines"
LINE_MODEL_FILE = "PhotiLines.onnx"

# OCR models, keyed by the label shown in the UI. Each repo holds a
# model_config.json plus the ONNX file that config names.
#
# Only Woodblock is listed because it is the sole BDRC repo whose config has
# every key ``read_ocr_model_config`` requires. CombinedBetsug_Wylie_{E,C}_v1,
# HTR-Base and GoogleBooks_E_v1 all omit ``add_blank`` (and ``version``), and
# guessing that flag wrong corrupts CTC decoding into plausible-looking garbage
# rather than raising -- so they stay out until their configs are complete.
OCR_MODEL_REPOS: Dict[str, str] = {
    "Woodblock": "BDRC/Woodblock",
}

DEFAULT_OCR_MODEL = "Woodblock"


def line_model_path() -> str:
    """Return a local path to the line-segmentation ONNX, downloading if needed."""
    return hf_hub_download(repo_id=LINE_MODEL_REPO, filename=LINE_MODEL_FILE)


def ocr_model_dir(name: str = DEFAULT_OCR_MODEL) -> str:
    """Return a local directory holding an OCR model's config and weights.

    ``import_local_model`` expects a directory containing ``model_config.json``
    alongside the ONNX file the config names. Both files come from the same Hub
    snapshot, so they land in the same cache directory.

    Raises:
        KeyError: If ``name`` is not a known OCR model.
    """
    if name not in OCR_MODEL_REPOS:
        raise KeyError(f"unknown OCR model {name!r}; known: {sorted(OCR_MODEL_REPOS)}")

    repo = OCR_MODEL_REPOS[name]
    config_path = hf_hub_download(repo_id=repo, filename="model_config.json")

    # Pull the weights the config points at, so they sit beside it in the cache.
    import json

    with open(config_path, encoding="utf-8") as fh:
        onnx_name = json.load(fh)["onnx-model"]
    hf_hub_download(repo_id=repo, filename=onnx_name)

    return os.path.dirname(config_path)


def prefetch(ocr_models: tuple = (DEFAULT_OCR_MODEL,)) -> None:
    """Warm the cache at startup so the first request doesn't pay the download."""
    line_model_path()
    for name in ocr_models:
        ocr_model_dir(name)
