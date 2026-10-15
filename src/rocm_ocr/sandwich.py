"""PyMuPDF Sandwich PDF Generator for ROCm OCR microservice.

Embeds OCR-recognized text bounding boxes as an invisible text layer
(render_mode=3) into existing PDF documents, producing searchable PDF
documents while preserving 100% of the original visual content.
Eliminates character drift using rotation-aware horizontal scaling matrix morphs,
detects born-digital documents, and cleans pre-existing text layers from scans.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Final

import pymupdf

from rocm_ocr.models import PageResult

# Standard PDF font (Base-14 Helvetica) - no external TTF files needed
DEFAULT_STANDARD_FONT: Final[str] = "helv"


def _page_has_fullpage_scan(page: pymupdf.Page) -> bool:
    """Check if page contains a raster image covering >= 70% of its area."""
    page_area = page.rect.width * page.rect.height
    for img in page.get_images():
        for r in page.get_image_rects(img[0]):
            if (r.width * r.height) >= 0.70 * page_area:
                return True
    return False


def _count_visible_chars(page: pymupdf.Page) -> int:
    """Count visible text characters on a page (excluding invisible OCR)."""
    return sum(
        len(span.get("chars", ()))
        for span in page.get_texttrace()
        if span.get("type") != 3
    )


def is_born_digital_pdf(pdf_bytes: bytes) -> tuple[bool, int, int]:
    """Check if a PDF document is born-digital with native vector text.

    Args:
        pdf_bytes: Raw binary content of the PDF.

    Returns:
        Tuple of (is_born_digital: bool, total_text_chars: int, page_count: int).
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = len(doc)
        if total_pages == 0:
            return False, 0, 0

        total_visible = sum(_count_visible_chars(page) for page in doc)
        fullpage_scan_pages = sum(1 for page in doc if _page_has_fullpage_scan(page))

        # Born-digital: has significant visible text and no or minority
        # fullpage scan pages
        if total_visible >= 50 and fullpage_scan_pages == 0:
            return True, total_visible, total_pages
        if total_visible >= 100 and fullpage_scan_pages < total_pages / 2:
            return True, total_visible, total_pages

        return False, total_visible, total_pages
    finally:
        doc.close()


def _strip_existing_text(page: pymupdf.Page) -> None:
    """Remove any existing text layer from a scanned page, preserving images."""
    if page.get_text().strip():
        page.add_redact_annot(page.rect)
        page.apply_redactions(images=0)


def _insert_ocr_box(
    page: pymupdf.Page,
    text: str,
    bbox: list[float],
    fontname: str = DEFAULT_STANDARD_FONT,
) -> None:
    """Inject an OCR text box as an invisible text layer with rotation-aware
    morph scaling.

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
    baseline_y = y1 - (box_height * 0.15)
    point = pymupdf.Point(x0, baseline_y)

    rot = page.rotation
    pt_morph = point
    if rot != 0:
        pt_morph = pt_morph * page.derotation_matrix

    # Compute horizontal scaling factor to eliminate character drift across words/lines
    morph = None
    try:
        f = pymupdf.Font(fontname)
        natural_w = f.text_length(clean_text, fontsize=fontsize)
        if natural_w > 0:
            scale_x = box_width / natural_w
            # Clamp to sensible ratio (0.5 to 2.5) to avoid extreme distortion
            if 0.5 <= scale_x <= 2.5:
                if rot in (90, 270):
                    morph = (pt_morph, pymupdf.Matrix(1.0, scale_x))
                else:
                    morph = (pt_morph, pymupdf.Matrix(scale_x, 1.0))
    except Exception:
        pass

    try:
        page.insert_text(
            pt_morph if rot != 0 else point,
            clean_text,
            fontsize=fontsize,
            render_mode=3,
            fontname=fontname,
            rotate=rot,
            morph=morph,
        )
    except Exception:
        page.insert_text(
            pt_morph if rot != 0 else point,
            clean_text,
            fontsize=fontsize,
            render_mode=3,
            fontname=DEFAULT_STANDARD_FONT,
            rotate=rot,
        )


def generate_sandwich_pdf(
    pdf_bytes: bytes,
    pages: Sequence[PageResult],
) -> bytes:
    """Inject an invisible OCR text layer into a PDF document using OCR results.

    Strips any pre-existing text layer from the page to prevent duplicate text,
    then embeds the new OCR text boxes with horizontal scaling.

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

            # Strip existing text layer before inserting new OCR boxes
            _strip_existing_text(page)

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
