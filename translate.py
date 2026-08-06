"""
Tibetan -> English translation with billingsmoore/mlotsawa-ground-base.

Follows the local backend in SimpleTranslationUI: generation calls the model
directly with its own ``task_specific_params["translation_bo_to_en"]`` settings
rather than going through ``pipeline("translation", ...)``, which was removed in
transformers >= 5 and raises "Unknown task translation".

The model is a plain seq2seq translator, not an instruction-following LLM, so
there is no prompt to configure. It loads lazily on first use and is cached for
the process; torch is imported inside the functions so app startup does not pay
for it.
"""

from __future__ import annotations

from typing import List

MODEL_ID = "billingsmoore/mlotsawa-ground-base"
PREFIX = "translate Tibetan to English: "

_model = None
_tokenizer = None


def _load():
    """Load and cache the translation model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"[INFO] Loading translation model: {MODEL_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
        _model.eval()
    return _model, _tokenizer


def prefetch() -> None:
    """Warm the cache at startup so the first translation isn't a download."""
    _load()


def translate_batch(texts: List[str], device: str | None = None) -> List[str]:
    """Translate Tibetan strings to English, preserving input order.

    ``device`` must be given explicitly when running outside an ``@spaces.GPU``
    call on a ZeroGPU Space. Autodetection is not safe there: ZeroGPU's emulation
    reports ``torch.cuda.is_available()`` as True in the main process so that
    libraries configure themselves for GPU, but actually touching CUDA outside an
    allocation trips its low-level init guard. Passing "cpu" keeps the fallback
    path away from CUDA entirely.
    """
    if not texts:
        return []

    import torch

    model, tokenizer = _load()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(
        [PREFIX + text for text in texts],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=300, num_beams=4, early_stopping=True
        )

    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
