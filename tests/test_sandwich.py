"""Tests for the sandwich PDF generator module."""

import io

import pymupdf

from rocm_ocr.models import BoundingBox, PageResult
from rocm_ocr.sandwich import generate_sandwich_pdf


def _create_minimal_pdf() -> bytes:
    """Create a minimal 1-page PDF document in memory."""
    doc = pymupdf.open()
    page = doc.new_page(width=595.0, height=842.0)
    # Add a visual shape/line to ensure non-empty PDF stream
    page.draw_rect(
        pymupdf.Rect(50, 50, 100, 100), color=(0, 0, 0), fill=(0.9, 0.9, 0.9)
    )
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_generate_sandwich_pdf_injects_invisible_text():
    """Verify that generate_sandwich_pdf injects searchable text."""
    pdf_bytes = _create_minimal_pdf()

    pages = [
        PageResult(
            page_number=0,
            width=595.0,
            height=842.0,
            boxes=[
                BoundingBox(
                    text="Sandwich Invoice #12345",
                    bbox=[50.0, 100.0, 250.0, 120.0],
                    confidence=0.99,
                ),
                BoundingBox(
                    text="Betrag: 42,50 EUR",
                    bbox=[50.0, 130.0, 200.0, 150.0],
                    confidence=0.98,
                ),
            ],
        )
    ]

    result_bytes = generate_sandwich_pdf(pdf_bytes, pages)
    assert len(result_bytes) > 0
    assert result_bytes.startswith(b"%PDF")

    # Verify searchable text via PyMuPDF
    res_doc = pymupdf.open(stream=result_bytes, filetype="pdf")
    page_text = res_doc[0].get_text()
    res_doc.close()

    assert "Sandwich Invoice #12345" in page_text
    assert "Betrag: 42,50 EUR" in page_text


def test_generate_sandwich_pdf_handles_rotation():
    """Verify that rotated pages place text properly with derotation matrix."""
    doc = pymupdf.open()
    page = doc.new_page(width=595.0, height=842.0)
    page.set_rotation(90)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    pdf_bytes = buf.getvalue()

    pages = [
        PageResult(
            page_number=0,
            width=842.0,
            height=595.0,
            boxes=[
                BoundingBox(
                    text="Rotated Text Content",
                    bbox=[50.0, 100.0, 250.0, 120.0],
                    confidence=0.95,
                )
            ],
        )
    ]

    result_bytes = generate_sandwich_pdf(pdf_bytes, pages)
    res_doc = pymupdf.open(stream=result_bytes, filetype="pdf")
    assert res_doc[0].rotation == 90
    assert "Rotated Text Content" in res_doc[0].get_text()
    res_doc.close()


def test_generate_sandwich_pdf_empty_boxes():
    """Verify graceful handling when page has empty boxes."""
    pdf_bytes = _create_minimal_pdf()
    pages = [
        PageResult(
            page_number=0,
            width=595.0,
            height=842.0,
            boxes=[
                BoundingBox(text="", bbox=[0.0, 0.0, 10.0, 10.0], confidence=0.0),
                BoundingBox(
                    text="Invalid", bbox=[50.0, 50.0, 40.0, 40.0], confidence=0.0
                ),
            ],
        )
    ]
    result_bytes = generate_sandwich_pdf(pdf_bytes, pages)
    assert len(result_bytes) > 0
