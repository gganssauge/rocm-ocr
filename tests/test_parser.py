"""Unit tests for document parsing and rendering."""

import io

import pymupdf
import pytest
from PIL import Image

from rocm_ocr.parser import UnsupportedFormatError, parse_document


def _create_sample_image(img_format: str) -> bytes:
    """Generate sample image bytes in memory."""
    img = Image.new("RGB", (100, 50), color="blue")
    buf = io.BytesIO()
    img.save(buf, format=img_format)
    return buf.getvalue()


def _create_sample_pdf() -> bytes:
    """Generate a 2-page sample PDF in memory."""
    doc = pymupdf.open()
    page1 = doc.new_page(width=300, height=400)
    page1.insert_text((50, 50), "Page One")
    page2 = doc.new_page(width=300, height=400)
    page2.insert_text((50, 50), "Page Two")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_parse_png_image() -> None:
    """Test parsing a valid PNG image."""
    data = _create_sample_image("PNG")
    pages = parse_document(data, filename="sample.png")
    assert len(pages) == 1
    assert pages[0].page_number == 0
    assert pages[0].width == 100.0
    assert pages[0].height == 50.0
    assert pages[0].scale == 1.0


def test_parse_jpeg_image() -> None:
    """Test parsing a valid JPEG image."""
    data = _create_sample_image("JPEG")
    pages = parse_document(data, filename="sample.jpg")
    assert len(pages) == 1
    assert pages[0].page_number == 0
    assert pages[0].scale == 1.0


def test_parse_pdf_document() -> None:
    """Test parsing a multi-page PDF document."""
    data = _create_sample_pdf()
    pages = parse_document(data, filename="doc.pdf")
    assert len(pages) == 2
    assert pages[0].page_number == 0
    assert pages[0].width == 300.0
    assert pages[0].height == 400.0
    assert pages[0].scale == 2.0
    assert pages[1].page_number == 1


def test_parse_empty_content_raises() -> None:
    """Test that empty payload raises UnsupportedFormatError."""
    with pytest.raises(UnsupportedFormatError, match="empty"):
        parse_document(b"")


def test_parse_corrupt_content_raises() -> None:
    """Test that corrupt payload raises UnsupportedFormatError."""
    with pytest.raises(UnsupportedFormatError, match="Unrecognized"):
        parse_document(b"not a valid file")
