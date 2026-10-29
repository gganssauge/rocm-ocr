"""Integration and API endpoint tests for the ROCm OCR microservice."""

import asyncio
import io
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from rocm_ocr.engine import ROCM_PROVIDER, GpuInferenceError
from rocm_ocr.main import _process_inference_pages, app
from rocm_ocr.parser import RenderedPage


def _create_test_image_bytes() -> bytes:
    """Create sample PNG bytes for upload testing."""
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("inference_fails", [False, True])
def test_inference_closes_images_and_page_iterator(
    monkeypatch: pytest.MonkeyPatch, inference_fails: bool
) -> None:
    """Close images and the parser generator on both successful and failed OCR."""
    image = Image.new("RGB", (10, 20))
    iterator_closed = False

    def pages() -> Generator[RenderedPage, None, None]:
        nonlocal iterator_closed
        try:
            yield RenderedPage(3, image, 10.0, 20.0, 1.0)
        finally:
            iterator_closed = True

    monkeypatch.setattr(
        "rocm_ocr.main.iter_document_pages", MagicMock(return_value=pages())
    )
    engine = MagicMock()
    engine.infer_page.return_value = []
    if inference_fails:
        engine.infer_page.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="inference failed"):
            _process_inference_pages(engine, b"document", "test.pdf")
    else:
        result = _process_inference_pages(engine, b"document", "test.pdf")
        assert [page.model_dump() for page in result] == [
            {"page_number": 3, "width": 10.0, "height": 20.0, "boxes": []}
        ]
    assert iterator_closed
    with pytest.raises(ValueError, match="closed"):
        image.getpixel((0, 0))


@pytest.fixture
async def client(native_rapid: MagicMock):
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as test_client,
        app.router.lifespan_context(app),
    ):
        yield test_client


@pytest.mark.asyncio
async def test_healthz_endpoint(client: AsyncClient) -> None:
    """Verify /healthz returns 200 and ok status."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_endpoint(client: AsyncClient) -> None:
    """Verify /readyz returns 200 and active provider when ready."""
    response = await client.get("/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "provider" in data


@pytest.mark.asyncio
async def test_readyz_endpoint_not_ready(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify /readyz returns 503 when engine is not ready."""
    mock_engine = MagicMock()
    mock_engine.is_ready.return_value = False
    monkeypatch.setattr(app.state, "engine", mock_engine)

    response = await client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["detail"] == "OCR engine or execution provider not ready"


@pytest.mark.asyncio
async def test_setup_endpoint(client: AsyncClient) -> None:
    """Verify /ocr/v1/setup updates concurrency dynamically."""
    response = await client.post("/ocr/v1/setup", json={"concurrency": 4})
    assert response.status_code == 200
    assert response.json() == {
        "status": "updated",
        "concurrency": 4,
        "active_tasks": 0,
    }


@pytest.mark.asyncio
async def test_setup_endpoint_validation_error(client: AsyncClient) -> None:
    """Verify /ocr/v1/setup validates input concurrency >= 1."""
    response = await client.post("/ocr/v1/setup", json={"concurrency": 0})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_success(client: AsyncClient) -> None:
    """Verify /ocr/v1/predict parses image and returns OCR response."""
    img_bytes = _create_test_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}

    response = await client.post("/ocr/v1/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "execution_provider" in data
    assert "inference_time_ms" in data
    assert "pages" in data
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 0


@pytest.mark.asyncio
async def test_predict_endpoint_bad_file(client: AsyncClient) -> None:
    """Verify /ocr/v1/predict returns 400 on invalid document format."""
    files = {"file": ("test.txt", b"plain text is not supported", "text/plain")}
    response = await client.post("/ocr/v1/predict", files=files)
    assert response.status_code == 400
    assert (
        "Unsupported" in response.json()["detail"]
        or "Unrecognized" in response.json()["detail"]
    )


@pytest.mark.asyncio
async def test_predict_endpoint_gpu_error_fail_fast(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify /ocr/v1/predict transparently returns 503 on GPU error."""
    mock_engine = MagicMock()
    mock_engine.active_provider = ROCM_PROVIDER
    mock_engine.infer_page.side_effect = GpuInferenceError(
        detail="ROCm GPU inference failed",
        provider_attempted=ROCM_PROVIDER,
        error_message="HIP out of memory error on device gfx1101",
    )
    monkeypatch.setattr(app.state, "engine", mock_engine)

    mock_exit = MagicMock()
    monkeypatch.setattr("rocm_ocr.main._fatal_exit", mock_exit)

    img_bytes = _create_test_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}

    response = await client.post("/ocr/v1/predict", files=files)
    assert response.status_code == 503
    assert response.json() == {
        "detail": "ROCm GPU inference failed",
        "provider_attempted": "ROCMExecutionProvider",
        "error_message": "HIP out of memory error on device gfx1101",
    }
    await asyncio.sleep(0.6)
    mock_exit.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_scalar_endpoint(client: AsyncClient) -> None:
    """Verify /scalar serves the interactive documentation."""
    response = await client.get("/scalar")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "ROCm OCR Microservice - API Documentation" in response.text
    assert "Scalar.createApiReference" in response.text


@pytest.mark.asyncio
async def test_scalar_endpoint_with_local_assets(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify /scalar uses local static URLs when assets exist in static directory."""
    from rocm_ocr import main

    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    (tmp_path / "scalar.js").write_text("console.log('scalar');", encoding="utf-8")
    (tmp_path / "favicon.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    response = await client.get("/scalar")
    assert response.status_code == 200
    assert "/static/scalar.js" in response.text
    assert "/static/favicon.png" in response.text
    assert "https://cdn.jsdelivr.net" not in response.text
    assert "https://fastapi.tiangolo.com" not in response.text


@pytest.mark.asyncio
async def test_docs_redirect_endpoint(client: AsyncClient) -> None:
    """Verify /docs redirects to the /scalar documentation."""
    response = await client.get("/docs", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers.get("location") == "/scalar"


@pytest.mark.asyncio
async def test_static_asset_serving(client: AsyncClient) -> None:
    """Verify StaticFiles serves assets placed in STATIC_DIR."""
    from rocm_ocr.main import STATIC_DIR

    test_file = STATIC_DIR / "test_asset.txt"
    test_file.write_text("static-content-ok", encoding="utf-8")
    try:
        response = await client.get("/static/test_asset.txt")
        assert response.status_code == 200
        assert response.text == "static-content-ok"
    finally:
        test_file.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_sandwich_endpoint_success(client: AsyncClient, monkeypatch) -> None:
    """Verify /ocr/v1/sandwich returns a valid PDF with invisible text layer."""
    import pymupdf

    from rocm_ocr.models import BoundingBox

    def _create_minimal_pdf() -> bytes:
        doc = pymupdf.open()
        page = doc.new_page(width=595.0, height=842.0)
        page.draw_rect(pymupdf.Rect(50, 50, 100, 100), color=(0, 0, 0))
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()

    mock_boxes = [
        BoundingBox(
            text="Test Injected Title", bbox=[50.0, 50.0, 200.0, 70.0], confidence=0.99
        )
    ]
    monkeypatch.setattr(
        app.state.engine, "infer_page", MagicMock(return_value=mock_boxes)
    )

    pdf_bytes = _create_minimal_pdf()
    files = {"file": ("test_doc.pdf", pdf_bytes, "application/pdf")}
    response = await client.post("/ocr/v1/sandwich", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "X-Extracted-Chars" in response.headers
    assert response.headers["X-Total-Pages"] == "1"

    res_doc = pymupdf.open(stream=response.content, filetype="pdf")
    assert "Test Injected Title" in res_doc[0].get_text()
    res_doc.close()


@pytest.mark.asyncio
async def test_sandwich_endpoint_rejects_non_pdf(client: AsyncClient) -> None:
    """Verify /ocr/v1/sandwich rejects non-PDF payloads with 400 Bad Request."""
    img_bytes = _create_test_image_bytes()
    files = {"file": ("test_doc.png", img_bytes, "image/png")}
    response = await client.post("/ocr/v1/sandwich", files=files)
    assert response.status_code == 400
    assert "requires a PDF" in response.json()["detail"]


@pytest.mark.asyncio
async def test_sandwich_endpoint_born_digital_passthrough(client: AsyncClient) -> None:
    """Verify /ocr/v1/sandwich returns born-digital PDF untouched with
    X-Born-Digital header.
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595.0, height=842.0)
    page.insert_text(
        pymupdf.Point(50, 100),
        "Präziser digitaler Text im Dokument für Born-Digital-Erkennung.",
        fontsize=12,
    )
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    pdf_bytes = buf.getvalue()

    files = {"file": ("digital.pdf", pdf_bytes, "application/pdf")}
    response = await client.post("/ocr/v1/sandwich", files=files)

    assert response.status_code == 200
    assert response.headers.get("x-born-digital") == "true"
    assert response.headers.get("x-skipped-reason") == "born-digital"
    assert response.content == pdf_bytes


@pytest.mark.asyncio
async def test_openapi_version_matches_app_version(client: AsyncClient) -> None:
    """Verify OpenAPI schema reports the correct application version."""
    from rocm_ocr.main import APP_VERSION

    response = await client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert data["info"]["version"] == APP_VERSION
    assert APP_VERSION == "0.2.2"


@pytest.mark.asyncio
async def test_auth_disabled_by_default(client: AsyncClient) -> None:
    """Verify endpoints accept requests without auth when api_key is empty."""
    from rocm_ocr.config import settings

    assert settings.api_key == ""
    response = await client.post("/ocr/v1/setup", json={"concurrency": 2})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_auth_enforced_when_api_key_configured(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify 401 returned when api_key is configured and token is invalid."""
    from rocm_ocr.config import settings

    monkeypatch.setattr(settings, "api_key", "super-secret-token")

    # Missing token
    res_missing = await client.post("/ocr/v1/setup", json={"concurrency": 2})
    assert res_missing.status_code == 401
    assert res_missing.json()["detail"] == "Invalid or missing API key"
    assert res_missing.headers.get("www-authenticate") == "Bearer"

    # Wrong token
    res_wrong = await client.post(
        "/ocr/v1/setup",
        json={"concurrency": 2},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert res_wrong.status_code == 401

    # Valid token
    res_valid = await client.post(
        "/ocr/v1/setup",
        json={"concurrency": 2},
        headers={"Authorization": "Bearer super-secret-token"},
    )
    assert res_valid.status_code == 200
    assert res_valid.json()["status"] == "updated"


@pytest.mark.asyncio
async def test_health_endpoints_bypass_auth_when_api_key_configured(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify /healthz and /readyz remain accessible without auth for probes."""
    from rocm_ocr.config import settings

    monkeypatch.setattr(settings, "api_key", "super-secret-token")

    res_health = await client.get("/healthz")
    assert res_health.status_code == 200

    res_ready = await client.get("/readyz")
    assert res_ready.status_code == 200


@pytest.mark.asyncio
async def test_healthz_responsive_during_inference(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify /healthz probe responds immediately even while OCR inference
    is executing.
    """
    import asyncio
    import time

    def slow_inference(*args, **kwargs):
        time.sleep(0.3)
        return []

    monkeypatch.setattr("rocm_ocr.main._process_inference_pages", slow_inference)

    pdf_bytes = b"%PDF-1.4 sample content"
    files = {"file": ("test.pdf", pdf_bytes, "application/pdf")}

    # Launch slow predict in the background
    predict_task = asyncio.create_task(client.post("/ocr/v1/predict", files=files))

    # Wait briefly to ensure inference thread is running
    await asyncio.sleep(0.05)

    # Health check should respond immediately while predict is still running
    health_start = time.perf_counter()
    health_res = await client.get("/healthz")
    health_elapsed = time.perf_counter() - health_start

    assert health_res.status_code == 200
    assert health_elapsed < 0.2  # Responded without waiting for the 0.3s sleep

    predict_res = await predict_task
    assert predict_res.status_code == 200
