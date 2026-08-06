"""
Render a synthetic pecha page for demos and testing.

The real example page is a dense woodblock print: seven long lines that take a
while to transcribe and longer to translate. This produces a clean, short page
instead, so the whole pipeline can be exercised quickly.

The text is the four-line refuge and bodhicitta verse plus the mani mantra --
short, ubiquitous, and well represented in translation training data, so the
output is easy to sanity-check by eye.

Tibetan needs complex-script shaping to stack its consonants, so the font is
loaded with the Raqm layout engine. Without it the glyphs come out unstacked
and the page is nonsense.

    python examples/make_sample.py [output.png]
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Assets", "Fonts", "TibMachUni-1.901b.ttf",
)

LINES = [
    "༄༅། །སངས་རྒྱས་ཆོས་དང་ཚོགས་ཀྱི་མཆོག་རྣམས་ལ།",
    "བྱང་ཆུབ་བར་དུ་བདག་ནི་སྐྱབས་སུ་མཆི།",
    "བདག་གིས་སྦྱིན་སོགས་བགྱིས་པའི་བསོད་ནམས་ཀྱིས།",
    "འགྲོ་ལ་ཕན་ཕྱིར་སངས་རྒྱས་འགྲུབ་པར་ཤོག",
    "ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ།",
]

PAGE = (1600, 520)
MARGIN_X = 90
FONT_SIZE = 46
LINE_GAP = 92
PAPER = (232, 223, 198)
INK = (38, 30, 24)


def render(path: str = "examples/sample_page.png") -> str:
    """Draw the sample page and write it to ``path``."""
    try:
        font = ImageFont.truetype(FONT, FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    except Exception:  # pragma: no cover - Pillow built without Raqm
        raise SystemExit(
            "Pillow lacks Raqm layout support, so Tibetan will not shape correctly.\n"
            "Install libraqm (e.g. apt install libraqm0) and reinstall Pillow."
        )

    image = Image.new("RGB", PAGE, PAPER)

    # A little paper grain, so the page is not implausibly flat for a model
    # trained on scans of physical pages.
    grain = np.random.default_rng(7).normal(0, 3.5, (PAGE[1], PAGE[0], 1))
    image = Image.fromarray(
        np.clip(np.asarray(image, dtype=np.float32) + grain, 0, 255).astype(np.uint8)
    )

    draw = ImageDraw.Draw(image)
    top = (PAGE[1] - (len(LINES) - 1) * LINE_GAP - FONT_SIZE) // 2
    for index, line in enumerate(LINES):
        draw.text(
            (MARGIN_X, top + index * LINE_GAP),
            line,
            font=font,
            fill=INK,
            language="bo",
        )

    # Margin rules, as on a printed pecha folio.
    draw.line([(46, 30), (46, PAGE[1] - 30)], fill=(150, 60, 55), width=3)
    draw.line(
        [(PAGE[0] - 46, 30), (PAGE[0] - 46, PAGE[1] - 30)], fill=(150, 60, 55), width=3
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path)
    return path


if __name__ == "__main__":
    out = render(sys.argv[1] if len(sys.argv) > 1 else "examples/sample_page.png")
    print(f"wrote {out}")
