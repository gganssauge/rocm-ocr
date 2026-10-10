"""Document rendering and format parsing for PDFs and raster images."""

import io
from dataclasses import dataclass

import fitz  # PyMuPDF
from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_FORMATS = {"PNG", "JPEG", "TIFF", "WEBP"}
PDF_RENDER_ZOOM = 2.0  # Render at 2x resolution (144 DPI) for accurate OCR


class UnsupportedFormatError(ValueError):
    """Raised when the uploaded document format is unsupported."""


@dataclass(frozen=True)
class RenderedPage:
    """Represents a rendered document page prepared for OCR inference."""

    page_number: int
    image: Image.Image
    width: float
    height: float
    scale: float


def _parse_pdf_pages(file_bytes: bytes) -> list[RenderedPage]:
    """Render each PDF page to an RGB image with scale tracking."""
    pages: list[RenderedPage] = []
    matrix = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(
                RenderedPage(
                    page_number=page_idx,
                    image=img,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    scale=PDF_RENDER_ZOOM,
                )
            )
    return pages


def _parse_image_file(file_bytes: bytes) -> list[RenderedPage]:
    """Parse a single raster image file into a RenderedPage."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as raw_img:
            img_format = (raw_img.format or "").upper()
            if img_format not in SUPPORTED_IMAGE_FORMATS:
                raise UnsupportedFormatError(
                    f"Unsupported image format: {img_format}. "
                    f"Supported formats: {sorted(SUPPORTED_IMAGE_FORMATS)}"
                )
            rgb_img = raw_img.convert("RGB")
            width, height = float(rgb_img.width), float(rgb_img.height)
            return [
                RenderedPage(
                    page_number=0,
                    image=rgb_img,
                    width=width,
                    height=height,
                    scale=1.0,
                )
            ]
    except UnidentifiedImageError as exc:
        raise UnsupportedFormatError(
            "Unrecognized file format or corrupted payload"
        ) from exc


def parse_document(file_bytes: bytes, filename: str = "") -> list[RenderedPage]:
    """Parse document bytes into a sequence of rendered pages.

    Args:
        file_bytes: Binary document content.
        filename: Optional filename used for hint inspection.

    Returns:
        List of RenderedPage objects ready for OCR inference.
    """
    if not file_bytes:
        raise UnsupportedFormatError("Uploaded document content is empty")

    is_pdf = file_bytes.startswith(b"%PDF") or filename.lower().endswith(".pdf")
    if is_pdf:
        try:
            return _parse_pdf_pages(file_bytes)
        except Exception as exc:
            raise UnsupportedFormatError(
                f"Failed to parse PDF document: {exc}"
            ) from exc

    return _parse_image_file(file_bytes)
