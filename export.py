"""
Export an edited transcription to txt, docx or pdf.

Tibetan needs care in two places. It has no spaces, so naive word-wrapping never
finds a break point and lines overflow; instead we break after the tsheg (U+0F0B)
or shad (U+0F0D) that already mark syllable and clause boundaries. And most
system fonts have no Tibetan glyphs, so the PDF embeds TibMachUni, which ships
with the upstream app.
"""

from __future__ import annotations

import os
from typing import List, Sequence

TIBETAN_FONT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Assets", "Fonts", "TibMachUni-1.901b.ttf"
)
FONT_NAME = "TibMachUni"

# Tsheg (syllable dot) and shad (clause bar) are the natural break points.
BREAK_AFTER = "་།"


def wrap_tibetan(text: str, limit: int) -> List[str]:
    """Wrap ``text`` to ``limit`` characters, breaking only after tsheg/shad.

    Falls back to a hard break when a single run of characters exceeds the limit
    with no break point in it, so output is never silently clipped.
    """
    if limit <= 0:
        return [text]

    lines: List[str] = []
    current = ""

    for char in text:
        current += char
        if len(current) >= limit:
            cut = max(current.rfind(c) for c in BREAK_AFTER)
            if cut > 0:
                lines.append(current[: cut + 1])
                current = current[cut + 1 :]
            else:
                lines.append(current)
                current = ""

    if current:
        lines.append(current)
    return lines


def to_txt(lines: Sequence[str], path: str) -> str:
    """Write the transcription as UTF-8 plain text, one detected line per row."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def to_docx(lines: Sequence[str], path: str) -> str:
    """Write a .docx with each line as its own paragraph in a Tibetan font."""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    document = Document()
    for text in lines:
        paragraph = document.add_paragraph()
        run = paragraph.add_run(text)
        run.font.name = FONT_NAME
        run.font.size = Pt(14)
        # Word picks the font for Tibetan off the complex-script slot, not the
        # latin one, so set both or the glyphs fall back to a boxes font.
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        rfonts.set(qn("w:cs"), FONT_NAME)
        rfonts.set(qn("w:eastAsia"), FONT_NAME)

    document.save(path)
    return path


def to_pdf(lines: Sequence[str], path: str, font_size: int = 14) -> str:
    """Write a PDF with TibMachUni embedded so it renders anywhere."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as pdfcanvas

    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, TIBETAN_FONT))

    page_width, page_height = A4
    margin = 40
    leading = font_size * 1.9  # Tibetan stacks above and below the baseline
    usable = page_width - 2 * margin

    pdf = pdfcanvas.Canvas(path, pagesize=A4)
    pdf.setFont(FONT_NAME, font_size)
    y = page_height - margin

    # Estimate how many glyphs fit, then wrap on real break points.
    sample = pdfmetrics.stringWidth("ག", FONT_NAME, font_size) or font_size * 0.5
    limit = max(int(usable / sample), 10)

    for text in lines:
        for chunk in wrap_tibetan(text, limit) or [""]:
            if y < margin + leading:
                pdf.showPage()
                pdf.setFont(FONT_NAME, font_size)
                y = page_height - margin
            pdf.drawString(margin, y, chunk)
            y -= leading

    pdf.save()
    return path


WRITERS = {"txt": to_txt, "docx": to_docx, "pdf": to_pdf}


def export(text: str, fmt: str, path: str) -> str:
    """Write ``text`` (newline-separated lines) to ``path`` in ``fmt``."""
    if fmt not in WRITERS:
        raise ValueError(f"unknown format {fmt!r}; known: {sorted(WRITERS)}")
    return WRITERS[fmt](text.split("\n"), path)
