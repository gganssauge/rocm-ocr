"""Tests for asgi-correlation-id middleware integration and logging filters."""

import logging
import uuid
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from rocm_ocr.logging import HealthCheckFilter
from rocm_ocr.main import app


@pytest.fixture
async def client(native_rapid: MagicMock):
    """Create async test client with lifespan context."""
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as c,
        app.router.lifespan_context(app),
    ):
        yield c


@pytest.mark.asyncio
async def test_correlation_id_generated_when_missing(client: AsyncClient) -> None:
    """Verify that a valid UUID4 correlation ID is generated when no header is sent."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    cid = response.headers.get("X-Correlation-ID")
    assert cid is not None
    # Validate it is a valid UUID
    uuid_obj = uuid.UUID(cid)
    assert uuid_obj.version == 4


@pytest.mark.asyncio
async def test_correlation_id_preserved_from_request_header(
    client: AsyncClient,
) -> None:
    """Verify that incoming X-Correlation-ID header is preserved."""
    test_cid = str(uuid.uuid4())
    response = await client.get("/healthz", headers={"X-Correlation-ID": test_cid})
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == test_cid


@pytest.mark.asyncio
async def test_custom_correlation_id_preserved(client: AsyncClient) -> None:
    """Verify that any custom string correlation ID is preserved."""
    custom_cid = "service-alpha-trace-987654"
    response = await client.get("/healthz", headers={"X-Correlation-ID": custom_cid})
    assert response.status_code == 200
    assert response.headers.get("X-Correlation-ID") == custom_cid


def test_health_check_filter() -> None:
    """Verify that HealthCheckFilter filters 200 OK probes but keeps errors."""
    flt = HealthCheckFilter()

    rec_healthz_200 = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "test.py",
        1,
        '127.0.0.1:123 - "GET /healthz HTTP/1.1" 200 OK',
        (),
        None,
    )
    assert flt.filter(rec_healthz_200) is False

    rec_readyz_200 = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "test.py",
        1,
        '127.0.0.1:123 - "GET /readyz HTTP/1.1" 200 OK',
        (),
        None,
    )
    assert flt.filter(rec_readyz_200) is False

    rec_healthz_500 = logging.LogRecord(
        "uvicorn.access",
        logging.ERROR,
        "test.py",
        1,
        '127.0.0.1:123 - "GET /healthz HTTP/1.1" 500 Internal',
        (),
        None,
    )
    assert flt.filter(rec_healthz_500) is True

    rec_predict_200 = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "test.py",
        1,
        '127.0.0.1:123 - "POST /ocr/v1/predict HTTP/1.1" 200 OK',
        (),
        None,
    )
    assert flt.filter(rec_predict_200) is True


def test_correlation_access_formatter() -> None:
    """Verify that CorrelationAccessFormatter renders correlation_id in access log."""
    from rocm_ocr.logging import CorrelationAccessFormatter

    formatter = CorrelationAccessFormatter(use_colors=False)
    rec = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "main.py",
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "POST", "/ocr/v1/predict", "1.1", 200),
        None,
    )
    rec.correlation_id = "test-cid-abc"
    result = formatter.format(rec)
    assert "[test-cid-abc]" in result
    assert "POST /ocr/v1/predict HTTP/1.1" in result
