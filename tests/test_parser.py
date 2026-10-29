"""Unit tests for document parsing and rendering."""

import io
import weakref

import pymupdf
import pytest
from PIL import Image

from rocm_ocr.parser import (
    DocumentTooLargeError,
    UnsupportedFormatError,
    parse_document,
)


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


def test_pdf_page_limit_is_100() -> None:
    """Reject the 101st page, including PDFs with tiny rendering dimensions."""
    with pymupdf.open() as doc:
        for _ in range(101):
            doc.new_page(width=1, height=1)
        data = doc.tobytes()

    with pytest.raises(DocumentTooLargeError, match="100 page limit"):
        parse_document(data, filename="too-many-pages.pdf")


@pytest.mark.parametrize("file_format", ["pdf", "png"])
def test_rendered_pages_over_20_million_pixels_are_rejected(
    file_format: str,
) -> None:
    """Apply the pixel budget to PDF rendering and raster image decoding."""
    if file_format == "pdf":
        with pymupdf.open() as doc:
            doc.new_page(width=2500, height=2000.1)
            data = doc.tobytes()
    else:
        with Image.new("1", (5000, 4001)) as image:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()

    with pytest.raises(DocumentTooLargeError, match="20000000 pixel limit"):
        parse_document(data, filename=f"too-large.{file_format}")


def test_pdf_iteration_does_not_retain_completed_page_images() -> None:
    """Keep only the current raster page alive during sequential consumption."""
    from rocm_ocr.parser import iter_document_pages

    pages = iter_document_pages(_create_sample_pdf(), filename="doc.pdf")
    first = next(pages)
    previous_image = weakref.ref(first.image)
    del first
    second = next(pages)

    assert second.page_number == 1
    assert previous_image() is None
    pages.close()


def test_pdf_with_exactly_100_pages_is_accepted() -> None:
    """Treat the page-count limit as inclusive."""
    from rocm_ocr.parser import iter_document_pages

    with pymupdf.open() as doc:
        for _ in range(100):
            doc.new_page(width=1, height=1)
        data = doc.tobytes()
    count = 0
    for page in iter_document_pages(data, filename="100-pages.pdf"):
        page.image.close()
        count += 1
    assert count == 100


@pytest.mark.parametrize("file_format", ["pdf", "png"])
def test_rendered_pages_at_exactly_20_million_pixels_are_accepted(
    file_format: str,
) -> None:
    """Treat the pixel budget as inclusive at the actual rendering resolution."""
    if file_format == "pdf":
        with pymupdf.open() as doc:
            doc.new_page(width=2500, height=2000)
            data = doc.tobytes()
    else:
        with Image.new("1", (5000, 4000)) as image:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
    pages = parse_document(data, filename=f"at-limit.{file_format}")
    try:
        assert pages[0].image.size == (5000, 4000)
    finally:
        pages[0].image.close()


@pytest.mark.parametrize("width", [10_000, 20_000])
def test_image_decompression_bombs_raise_document_limit_errors(width: int) -> None:
    """Map Pillow's warning and error paths to the same explicit size error."""
    with Image.new("1", (width, 10_000)) as image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
    with pytest.raises(DocumentTooLargeError, match="20000000 pixel limit"):
        parse_document(buf.getvalue(), filename="large.png")
