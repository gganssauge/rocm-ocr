"""Tests for the sandwich PDF generator module."""

import io
from unittest.mock import MagicMock

import pymupdf
import pytest

from rocm_ocr.models import BoundingBox, PageResult
from rocm_ocr.sandwich import generate_sandwich_pdf, is_born_digital_pdf


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


@pytest.mark.parametrize("operation", ["inspect", "generate"])
def test_pdf_is_closed_when_processing_fails(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Release native PDF resources even when validation or text insertion fails."""
    data = _create_minimal_pdf()
    doc = pymupdf.open(stream=data, filetype="pdf")
    monkeypatch.setattr("rocm_ocr.sandwich.pymupdf.open", MagicMock(return_value=doc))
    failure = MagicMock(side_effect=RuntimeError("processing failed"))
    if operation == "inspect":
        monkeypatch.setattr("rocm_ocr.sandwich.validate_pdf_document", failure)
        with pytest.raises(RuntimeError, match="processing failed"):
            is_born_digital_pdf(data)
    else:
        monkeypatch.setattr("rocm_ocr.sandwich._strip_existing_text", failure)
        pages = [PageResult(page_number=0, width=595, height=842)]
        with pytest.raises(RuntimeError, match="processing failed"):
            generate_sandwich_pdf(data, pages)
    assert doc.is_closed


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
    page.insert_text(
        pymupdf.Point(50, 100), "Alte schlechte OCR", fontsize=12, render_mode=3
    )
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


def test_sandwich_preserves_visible_content_while_replacing_invisible_ocr() -> None:
    """Keep vector graphics, images and overlapping visible text pixel-identical."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=200)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.clear_with(240)
        page.insert_image(page.rect, pixmap=pix)
        page.draw_rect(pymupdf.Rect(20, 20, 80, 80), color=(1, 0, 0), fill=(1, 0, 0))
        page.insert_text((20, 110), "Visible title")
        page.insert_text((20, 110), "Old OCR", render_mode=3)
        original_pixels = page.get_pixmap().samples
        data = doc.tobytes()
    result = generate_sandwich_pdf(
        data,
        [
            PageResult(
                page_number=0,
                width=200,
                height=200,
                boxes=[
                    BoundingBox(
                        text="New OCR",
                        bbox=[20, 100, 100, 115],
                        confidence=0.99,
                    )
                ],
            )
        ],
    )
    with pymupdf.open(stream=result, filetype="pdf") as doc:
        assert doc[0].get_pixmap().samples == original_pixels
        text = doc[0].get_text()
        assert "Visible title" in text
        assert "Old OCR" not in text
        assert "New OCR" in text


def test_sandwich_does_not_apply_existing_redaction_annotations() -> None:
    """Uncommitted user redactions must remain annotations, not alter the PDF."""
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 40), "Visible title")
        page.insert_text((20, 100), "Old OCR", render_mode=3)
        page.add_redact_annot(pymupdf.Rect(10, 20, 120, 50), fill=(0, 0, 0))
        original_pixels = page.get_pixmap().samples
        data = doc.tobytes()
    result = generate_sandwich_pdf(
        data, [PageResult(page_number=0, width=200, height=200, boxes=[])]
    )
    with pymupdf.open(stream=result, filetype="pdf") as doc:
        assert doc[0].get_pixmap().samples == original_pixels
        assert len(list(doc[0].annots())) == 1
        assert "Visible title" in doc[0].get_text()
        assert "Old OCR" not in doc[0].get_text()


def test_is_born_digital_empty_pdf():
    """Verify empty PDF returns (False, 0, 0)."""
    from rocm_ocr.sandwich import is_born_digital_pdf

    empty_pdf_bytes = (
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
        b"xref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000068 00000 n \n"
        b"trailer\n<< /Root 1 0 R /Size 3 >>\nstartxref\n120\n%%EOF\n"
    )
    is_bd, chars, pages = is_born_digital_pdf(empty_pdf_bytes)
    assert is_bd is False
    assert chars == 0
    assert pages == 0


def test_insert_ocr_box_fallback_on_invalid_font():
    """Verify _insert_ocr_box handles invalid font without crashing."""
    from rocm_ocr.sandwich import _insert_ocr_box

    doc = pymupdf.open()
    page = doc.new_page(width=595.0, height=842.0)
    _insert_ocr_box(
        page,
        text="Test",
        bbox=[50.0, 50.0, 100.0, 70.0],
        fontname="nonexistent_font_fallback",
    )
    doc.close()


def test_is_born_digital_mixed_multipage_pdf():
    """Never skip scanned pages merely because native text is the majority."""
    from rocm_ocr.sandwich import is_born_digital_pdf

    doc = pymupdf.open()
    # Page 0: Digital text
    p0 = doc.new_page(width=595.0, height=842.0)
    p0.insert_text(
        pymupdf.Point(50, 100),
        "Erste Seite mit ausreichend digitalem Vektor-Text für die Erkennung.",
        fontsize=12,
    )
    # Page 1: Digital text
    p1 = doc.new_page(width=595.0, height=842.0)
    p1.insert_text(
        pymupdf.Point(50, 100),
        "Zweite Seite mit weiterem ausführlichen Textinhalt im Dokument.",
        fontsize=12,
    )
    # Page 2: Full-page scan image
    p2 = doc.new_page(width=595.0, height=842.0)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    p2.insert_image(p2.rect, pixmap=pix)

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    is_bd, chars, pages = is_born_digital_pdf(buf.getvalue())
    assert is_bd is False
    assert pages == 3
    assert chars >= 100


def test_generate_sandwich_skips_unmapped_pages():
    """Verify pages not in OCR results are preserved without modifications."""
    doc = pymupdf.open()
    doc.new_page(width=595.0, height=842.0)  # Page 0
    doc.new_page(width=595.0, height=842.0)  # Page 1
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    # Only provide boxes for Page 1
    pages = [
        PageResult(
            page_number=1,
            width=595.0,
            height=842.0,
            boxes=[
                BoundingBox(
                    text="Seite 2 Text",
                    bbox=[50.0, 100.0, 200.0, 120.0],
                    confidence=0.99,
                )
            ],
        )
    ]
    res_bytes = generate_sandwich_pdf(buf.getvalue(), pages)
    res_doc = pymupdf.open(stream=res_bytes, filetype="pdf")
    assert len(res_doc) == 2
    assert res_doc[0].get_text().strip() == ""
    assert "Seite 2 Text" in res_doc[1].get_text()
    res_doc.close()


def test_generate_sandwich_rotated_90_morph():
    """Verify morph scaling works on 90-degree rotated pages."""
    doc = pymupdf.open()
    page = doc.new_page(width=595.0, height=842.0)
    page.set_rotation(90)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    pages = [
        PageResult(
            page_number=0,
            width=842.0,
            height=595.0,
            boxes=[
                BoundingBox(
                    text="Rotated 90 Deg Scaled Text",
                    bbox=[50.0, 100.0, 300.0, 120.0],
                    confidence=0.99,
                )
            ],
        )
    ]
    res_bytes = generate_sandwich_pdf(buf.getvalue(), pages)
    res_doc = pymupdf.open(stream=res_bytes, filetype="pdf")
    assert "Rotated 90 Deg Scaled Text" in res_doc[0].get_text()
    res_doc.close()
