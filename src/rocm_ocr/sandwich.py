"""PyMuPDF Sandwich PDF Generator for ROCm OCR microservice.

Embeds OCR-recognized text bounding boxes as an invisible text layer
(render_mode=3) into existing PDF documents, producing searchable PDF
documents while preserving 100% of the original visual content.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Final

import pymupdf

from rocm_ocr.models import PageResult

# Standard PDF font (Base-14 Helvetica) - no external TTF files needed
DEFAULT_STANDARD_FONT: Final[str] = "helv"


def _insert_ocr_box(
    page: pymupdf.Page,
    text: str,
    bbox: list[float],
    fontname: str = DEFAULT_STANDARD_FONT,
) -> None:
    """Inject an OCR text box as an invisible text layer, centered in bounding box.

    Args:
        page: Target PyMuPDF page.
        text: Recognized text string.
        bbox: Bounding box [x0, y0, x1, y1] in points.
        fontname: Standard PDF font name (default: 'helv').
    """
    clean_text = text.strip()
    if not clean_text or len(bbox) != 4:
        return

    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        return

    box_height = y1 - y0
    box_width = x1 - x0
    fontsize = max(3.0, box_height * 0.85)

    # Perfectly centered horizontally in the bounding box
    try:
        f = pymupdf.Font(fontname)
        text_w = f.text_length(clean_text, fontsize=fontsize)
        x_start = x0 + (box_width - text_w) / 2.0
    except (RuntimeError, ValueError, AttributeError):
        x_start = x0

    baseline_y = y1 - (box_height * 0.15)
    point = pymupdf.Point(x_start, baseline_y)

    rot = page.rotation
    if rot != 0:
        point = point * page.derotation_matrix

    page.insert_text(
        point,
        clean_text,
        fontsize=fontsize,
        render_mode=3,
        fontname=fontname,
        rotate=rot,
    )


def generate_sandwich_pdf(
    pdf_bytes: bytes,
    pages: Sequence[PageResult],
) -> bytes:
    """Inject an invisible OCR text layer into a PDF document using OCR results.

    Args:
        pdf_bytes: Raw binary content of the source PDF.
        pages: Sequence of PageResult objects containing detected text bounding boxes.

    Returns:
        The generated searchable sandwich PDF bytes.
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages_by_num = {p.page_number: p for p in pages}

        for page_idx in range(len(doc)):
            if page_idx not in pages_by_num:
                continue

            page_result = pages_by_num[page_idx]
            page = doc[page_idx]

            for box in page_result.boxes:
                _insert_ocr_box(
                    page,
                    text=box.text,
                    bbox=box.bbox,
                    fontname=DEFAULT_STANDARD_FONT,
                )

        out_buf = io.BytesIO()
        doc.save(out_buf, garbage=4, deflate=True)
        return out_buf.getvalue()
    finally:
        doc.close()
