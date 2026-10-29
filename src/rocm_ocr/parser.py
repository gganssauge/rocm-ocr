"""Document rendering and format parsing for PDFs and raster images."""

import io
import warnings
from collections.abc import Generator, Iterator
from dataclasses import dataclass

import pymupdf
from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "TIFF", "WEBP"}
PDF_RENDER_ZOOM = 2.0  # Render at 2x resolution (144 DPI) for accurate OCR
MAX_PDF_PAGES = 100
MAX_PAGE_PIXELS = 20_000_000


class UnsupportedFormatError(ValueError):
    """Raised when the uploaded document format is unsupported."""


class DocumentTooLargeError(ValueError):
    """Raised when a document exceeds the page or rasterization budget."""


@dataclass(frozen=True)
class RenderedPage:
    """Represents a rendered document page prepared for OCR inference."""

    page_number: int
    image: Image.Image
    width: float
    height: float
    scale: float


def count_visible_pdf_chars(page: pymupdf.Page) -> int:
    """Count native visible text, excluding invisible OCR and transparent spans."""
    return sum(
        len(span.get("chars", ()))
        for span in page.get_texttrace()
        if span["type"] != 3 and span["opacity"] > 0
    )


def pdf_page_needs_ocr(page: pymupdf.Page) -> bool:
    """Require OCR unless a page has sufficient native text and no full-page scan."""
    if count_visible_pdf_chars(page) < 50:
        return True
    page_area = page.rect.width * page.rect.height
    return any(
        rect.width * rect.height >= 0.70 * page_area
        for image in page.get_images()
        for rect in page.get_image_rects(image[0])
    )


def validate_pdf_document(doc: pymupdf.Document) -> None:
    """Reject PDFs exceeding the supported document resource budget."""
    if len(doc) > MAX_PDF_PAGES:
        raise DocumentTooLargeError("PDF exceeds the 100 page limit")
    matrix = pymupdf.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    for page in doc:
        rect = (page.rect * matrix).irect
        if rect.width * rect.height > MAX_PAGE_PIXELS:
            raise DocumentTooLargeError(
                "Rendered PDF page exceeds the 20000000 pixel limit"
            )


def _parse_pdf_pages(file_bytes: bytes, ocr_only: bool) -> Iterator[RenderedPage]:
    """Render each PDF page to an RGB image with scale tracking."""
    matrix = pymupdf.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        validate_pdf_document(doc)
        for page_idx, page in enumerate(doc):
            if ocr_only and not pdf_page_needs_ocr(page):
                continue
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield RenderedPage(
                page_number=page_idx,
                image=img,
                width=float(page.rect.width),
                height=float(page.rect.height),
                scale=PDF_RENDER_ZOOM,
            )


def _parse_image_file(file_bytes: bytes) -> list[RenderedPage]:
    """Parse a single raster image file into a RenderedPage."""
    try:
        with (
            warnings.catch_warnings(
                action="error", category=Image.DecompressionBombWarning
            ),
            Image.open(io.BytesIO(file_bytes)) as raw_img,
        ):
            img_format = (raw_img.format or "").upper()
            if img_format not in SUPPORTED_IMAGE_FORMATS:
                raise UnsupportedFormatError(
                    f"Unsupported image format: {img_format}. "
                    f"Supported formats: {sorted(SUPPORTED_IMAGE_FORMATS)}"
                )
            if raw_img.width * raw_img.height > MAX_PAGE_PIXELS:
                raise DocumentTooLargeError("Image exceeds the 20000000 pixel limit")
            rgb_img = raw_img.convert("RGB")
            return [
                RenderedPage(
                    page_number=0,
                    image=rgb_img,
                    width=float(rgb_img.width),
                    height=float(rgb_img.height),
                    scale=1.0,
                )
            ]
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise DocumentTooLargeError("Image exceeds the 20000000 pixel limit") from exc
    except UnidentifiedImageError as exc:
        raise UnsupportedFormatError(
            "Unrecognized file format or corrupted payload"
        ) from exc


def iter_document_pages(
    file_bytes: bytes, filename: str = "", *, ocr_only: bool = False
) -> Generator[RenderedPage, None, None]:
    """Render binary PDF/image content, using filename as a format hint.

    With ocr_only, skip native PDF pages. Callers must close yielded images.
    """
    if not file_bytes:
        raise UnsupportedFormatError("Uploaded document content is empty")

    is_pdf = file_bytes.startswith(b"%PDF") or filename.lower().endswith(".pdf")
    if is_pdf:
        try:
            yield from _parse_pdf_pages(file_bytes, ocr_only)
        except DocumentTooLargeError:
            raise
        except Exception as exc:
            raise UnsupportedFormatError(
                f"Failed to parse PDF document: {exc}"
            ) from exc

    else:
        yield from _parse_image_file(file_bytes)


def parse_document(file_bytes: bytes, filename: str = "") -> list[RenderedPage]:
    """Collect rendered pages for callers requiring a reusable list.

    Use iter_document_pages for bounded-memory sequential inference.
    """
    return list(iter_document_pages(file_bytes, filename))
