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


def test_is_born_digital_pdf_detection():
    """Verify is_born_digital_pdf accurately identifies digital PDFs vs scans."""
    from rocm_ocr.sandwich import is_born_digital_pdf

    # 1. Born digital PDF with native text
    doc_digital = pymupdf.open()
    page = doc_digital.new_page(width=595.0, height=842.0)
    page.insert_text(
        pymupdf.Point(50, 100),
        "Dies ist ein digitales Dokument mit ausreichend Text, um als "
        "born-digital eingestuft zu werden.",
        fontsize=12,
    )
    buf_digital = io.BytesIO()
    doc_digital.save(buf_digital)
    doc_digital.close()

    is_bd, chars, pages = is_born_digital_pdf(buf_digital.getvalue())
    assert is_bd is True
    assert chars > 50
    assert pages == 1

    # 2. Scanned PDF (image covering full page)
    doc_scan = pymupdf.open()
    page_scan = doc_scan.new_page(width=595.0, height=842.0)
    # Insert full page image
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    page_scan.insert_image(page_scan.rect, pixmap=pix)
    buf_scan = io.BytesIO()
    doc_scan.save(buf_scan)
    doc_scan.close()

    is_bd_scan, _, _ = is_born_digital_pdf(buf_scan.getvalue())
    assert is_bd_scan is False


def test_generate_sandwich_pdf_strips_old_text():
    """Verify generate_sandwich_pdf removes pre-existing text before adding
    new OCR layer.
    """
    # Create PDF with old text
    doc = pymupdf.open()
    page = doc.new_page(width=595.0, height=842.0)
    page.insert_text(pymupdf.Point(50, 100), "Alte schlechte OCR", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    pages = [
        PageResult(
            page_number=0,
            width=595.0,
            height=842.0,
            boxes=[
                BoundingBox(
                    text="Neue ROCm-OCR",
                    bbox=[50.0, 100.0, 200.0, 120.0],
                    confidence=0.99,
                )
            ],
        )
    ]

    sandwich_bytes = generate_sandwich_pdf(buf.getvalue(), pages)
    res_doc = pymupdf.open(stream=sandwich_bytes, filetype="pdf")
    res_text = res_doc[0].get_text()
    res_doc.close()

    assert "Alte schlechte OCR" not in res_text
    assert "Neue ROCm-OCR" in res_text
