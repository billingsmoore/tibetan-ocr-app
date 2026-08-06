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


# A whole pecha line is around 20:1. Drawn across one page width that leaves the
# script about five points tall -- present but unreadable. Splitting the strip
# into stacked segments trades vertical space for legibility; this is the
# rendered height each segment aims for.
SEGMENT_TARGET_HEIGHT = 46
MAX_SEGMENTS = 3


def _trim_dark_border(array):
    """Drop the black margin deskewing leaves around a crop.

    Rotating the page fills the corners with black, and a slanted line's crop
    inherits bands of it. Left in, those bands dominate the strip's height and
    push the script down to nothing once it is scaled to page width.
    """
    if array is None or getattr(array, "size", 0) == 0 or array.ndim != 3:
        return array

    lit = array.mean(axis=2) > 12
    rows, cols = lit.any(axis=1), lit.any(axis=0)
    if not rows.any() or not cols.any():
        return array

    top, bottom = int(rows.argmax()), len(rows) - int(rows[::-1].argmax())
    left, right = int(cols.argmax()), len(cols) - int(cols[::-1].argmax())
    return array[top:bottom, left:right]


def _segment_strip(image, usable: float):
    """Cut a wide line strip into pieces that stay legible at page width."""
    from math import ceil

    if image.width <= 0 or image.height <= 0:
        return []

    natural = usable * image.height / image.width
    parts = max(1, min(MAX_SEGMENTS, ceil(SEGMENT_TARGET_HEIGHT / max(natural, 1e-6))))
    if parts == 1:
        return [image]

    step = ceil(image.width / parts)
    return [
        image.crop((start, 0, min(start + step, image.width), image.height))
        for start in range(0, image.width, step)
    ]


def to_interlinear_pdf(items: Sequence[dict], path: str, font_size: int = 13) -> str:
    """Write a PDF of line strips, each with its transcription and translation.

    ``items`` are dicts with ``crop`` (an RGB numpy array of the page strip the
    line was read from), ``source`` (Tibetan) and ``target`` (English). Laying
    the three out together mirrors the editing UI, so the PDF can be checked
    against the original without going back to the page.
    """
    from io import BytesIO

    from PIL import Image
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas as pdfcanvas

    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, TIBETAN_FONT))

    page_width, page_height = A4
    margin = 42
    usable = page_width - 2 * margin
    tibetan_leading = font_size * 1.95
    english_size = font_size - 2
    english_leading = english_size * 1.45

    pdf = pdfcanvas.Canvas(path, pagesize=A4)
    pdf.setTitle("Transcription and translation")
    y = page_height - margin

    tibetan_glyph = pdfmetrics.stringWidth("ག", FONT_NAME, font_size) or font_size * 0.5
    tibetan_limit = max(int(usable / tibetan_glyph), 10)

    for number, item in enumerate(items, start=1):
        crop = item.get("crop")
        source = (item.get("source") or "").strip()
        target = (item.get("target") or "").strip()

        source_lines = wrap_tibetan(source, tibetan_limit) if source else []
        english_lines = _wrap_plain(target, usable, english_size) if target else []

        segments = []
        crop = _trim_dark_border(crop)
        if crop is not None and getattr(crop, "size", 0):
            for piece in _segment_strip(Image.fromarray(crop), usable):
                height = usable * piece.height / piece.width
                buffer = BytesIO()
                piece.save(buffer, format="PNG")
                buffer.seek(0)
                segments.append((ImageReader(buffer), height))

        block = (
            14
            + sum(height + 3 for _, height in segments)
            + 6
            + tibetan_leading * len(source_lines)
            + (6 + english_leading * len(english_lines) if english_lines else 0)
            + 18
        )
        # Start a fresh page rather than split an entry, unless it is taller than
        # a page on its own -- then it has to flow regardless.
        if y - block < margin and y < page_height - margin - 1:
            pdf.showPage()
            y = page_height - margin

        pdf.setFont("Helvetica-Bold", 8)
        pdf.setFillGray(0.5)
        pdf.drawString(margin, y - 9, f"{number}")
        pdf.setFillGray(0)
        y -= 14

        for reader, height in segments:
            pdf.drawImage(
                reader, margin, y - height, width=usable, height=height, mask="auto"
            )
            y -= height + 3

        y -= 6
        pdf.setFont(FONT_NAME, font_size)
        for chunk in source_lines:
            pdf.drawString(margin, y - font_size, chunk)
            y -= tibetan_leading

        if english_lines:
            y -= 6
            pdf.setFont("Helvetica", english_size)
            pdf.setFillGray(0.3)
            for chunk in english_lines:
                pdf.drawString(margin, y - english_size, chunk)
                y -= english_leading
            pdf.setFillGray(0)

        y -= 10
        pdf.setStrokeGray(0.85)
        pdf.setLineWidth(0.5)
        pdf.line(margin, y, page_width - margin, y)
        y -= 8

    pdf.save()
    return path


def _wrap_plain(text: str, width: float, size: int) -> List[str]:
    """Wrap Latin text on spaces to fit ``width`` points."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, "Helvetica", size) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


WRITERS = {"txt": to_txt, "docx": to_docx, "pdf": to_pdf}


def export(text: str, fmt: str, path: str) -> str:
    """Write ``text`` (newline-separated lines) to ``path`` in ``fmt``."""
    if fmt not in WRITERS:
        raise ValueError(f"unknown format {fmt!r}; known: {sorted(WRITERS)}")
    return WRITERS[fmt](text.split("\n"), path)
