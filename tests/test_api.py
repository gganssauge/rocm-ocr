"""Integration and API endpoint tests for the ROCm OCR microservice."""

import io
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from rocm_ocr.engine import ROCM_PROVIDER, GpuInferenceError
from rocm_ocr.main import app


def _create_test_image_bytes() -> bytes:
    """Create sample PNG bytes for upload testing."""
    img = Image.new("RGB", (100, 100), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
async def client():
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
async def test_readyz_endpoint_not_ready(client: AsyncClient) -> None:
    """Verify /readyz returns 503 when engine is not ready."""
    mock_engine = MagicMock()
    mock_engine.is_ready.return_value = False
    app.state.engine = mock_engine

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
async def test_predict_endpoint_gpu_error_fail_fast(client: AsyncClient) -> None:
    """Verify /ocr/v1/predict transparently returns 503 on GPU error."""
    mock_engine = MagicMock()
    mock_engine.active_provider = ROCM_PROVIDER
    mock_engine.infer_page.side_effect = GpuInferenceError(
        detail="ROCm GPU inference failed",
        provider_attempted=ROCM_PROVIDER,
        error_message="HIP out of memory error on device gfx1101",
    )
    app.state.engine = mock_engine

    img_bytes = _create_test_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}

    response = await client.post("/ocr/v1/predict", files=files)
    assert response.status_code == 503
    assert response.json() == {
        "detail": "ROCm GPU inference failed",
        "provider_attempted": "ROCMExecutionProvider",
        "error_message": "HIP out of memory error on device gfx1101",
    }
