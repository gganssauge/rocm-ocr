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
from pymupdf.mupdf import (
    PDF_REDACT_IMAGE_NONE,
    PDF_REDACT_LINE_ART_NONE,
    PDF_REDACT_TEXT_REMOVE_INVISIBLE,
    PdfRedactOptions,
    pdf_apply_redaction,
)

from rocm_ocr.models import PageResult
from rocm_ocr.parser import (
    UnsupportedFormatError,
    count_visible_pdf_chars,
    pdf_page_needs_ocr,
    validate_pdf_document,
)

# Standard PDF font (Base-14 Helvetica) - no external TTF files needed
DEFAULT_STANDARD_FONT: Final[str] = "helv"


def is_born_digital_pdf(pdf_bytes: bytes) -> tuple[bool, int, int]:
    """Check if a PDF document is born-digital with native vector text.

    Args:
        pdf_bytes: Raw binary content of the PDF.

    Returns:
        Tuple of (is_born_digital: bool, total_text_chars: int, page_count: int).
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise UnsupportedFormatError(f"Failed to parse PDF document: {exc}") from exc
    try:
        validate_pdf_document(doc)
        total_pages = len(doc)
        if total_pages == 0:
            return False, 0, 0

        total_visible = sum(count_visible_pdf_chars(page) for page in doc)
        all_native = all(not pdf_page_needs_ocr(page) for page in doc)
        return all_native, total_visible, total_pages
    finally:
        doc.close()


def _strip_existing_text(page: pymupdf.Page) -> None:
    """Remove invisible OCR text without modifying visible page content."""
    if any(span["type"] == 3 for span in page.get_texttrace()):
        annotation = page.add_redact_annot(page.rect, fill=None, cross_out=False)
        options = PdfRedactOptions()
        options.black_boxes = 0
        options.image_method = PDF_REDACT_IMAGE_NONE
        options.line_art = PDF_REDACT_LINE_ART_NONE
        options.text = PDF_REDACT_TEXT_REMOVE_INVISIBLE
        # Applying all page redactions would also commit the user's annotations.
        if not pdf_apply_redaction(annotation.this, options):
            raise RuntimeError("Failed to remove the existing invisible OCR layer")


def _compute_morph(
    fontname: str,
    clean_text: str,
    fontsize: float,
    box_width: float,
    pt_morph: pymupdf.Point,
    rot: int,
) -> tuple[pymupdf.Point, pymupdf.Matrix] | None:
    """Compute horizontal scaling factor to eliminate character drift."""
    try:
        f = pymupdf.Font(fontname)
        natural_w = f.text_length(clean_text, fontsize=fontsize)
        if natural_w <= 0:
            return None
        scale_x = box_width / natural_w
        if not (0.5 <= scale_x <= 2.5):
            return None
        if rot in (90, 270):
            return (pt_morph, pymupdf.Matrix(1.0, scale_x))
        return (pt_morph, pymupdf.Matrix(scale_x, 1.0))
    except Exception:
        return None


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
    if not clean_text or len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return

    x0, y0, x1, y1 = bbox
    box_height = y1 - y0
    box_width = x1 - x0
    fontsize = max(3.0, box_height * 0.85)
    baseline_y = y1 - (box_height * 0.15)
    point = pymupdf.Point(x0, baseline_y)

    rot = page.rotation
    pt_morph = point * page.derotation_matrix if rot != 0 else point
    morph = _compute_morph(fontname, clean_text, fontsize, box_width, pt_morph, rot)

    try:
        page.insert_text(
            pt_morph,
            clean_text,
            fontsize=fontsize,
            render_mode=3,
            fontname=fontname,
            rotate=rot,
            morph=morph,
        )
    except Exception:
        page.insert_text(
            pt_morph,
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

    Strips pre-existing invisible OCR text to prevent duplicates, preserving
    native visible content, then embeds new OCR text boxes with horizontal scaling.

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
