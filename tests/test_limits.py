"""HTTP regression tests for document resource limits."""

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pymupdf
import pytest
from httpx import ASGITransport, AsyncClient

from rocm_ocr.main import app


@pytest.fixture
async def client(native_rapid: MagicMock) -> AsyncIterator[AsyncClient]:
    """Keep the real API and parser while replacing native GPU inference."""
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http_client,
    ):
        yield http_client


def _native_pdf(page_count: int = 1, width: float = 300, height: float = 400) -> bytes:
    with pymupdf.open() as doc:
        for _ in range(page_count):
            page = doc.new_page(width=width, height=height)
            page.insert_text(
                (5, 20),
                "A native document with enough visible text to skip OCR analysis.",
                fontsize=4,
            )
        return doc.tobytes()


@pytest.mark.parametrize("endpoint", ["/ocr/v1/predict", "/ocr/v1/sandwich"])
async def test_requests_over_50_mib_are_rejected(endpoint: str) -> None:
    """Reject oversized requests before multipart parsing or engine access."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            endpoint,
            content=b"",
            headers={"Content-Length": "52428801"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "HTTP request exceeds the 50 MiB limit"}


@pytest.mark.parametrize("endpoint", ["/ocr/v1/predict", "/ocr/v1/sandwich"])
@pytest.mark.parametrize("limit", ["page-count", "pixels"])
async def test_pdf_limits_apply_even_to_native_documents(
    client: AsyncClient, endpoint: str, limit: str
) -> None:
    """Do not bypass document budgets when born-digital PDFs skip inference."""
    data = (
        _native_pdf(page_count=101)
        if limit == "page-count"
        else _native_pdf(width=2500, height=2000.1)
    )
    response = await client.post(
        endpoint, files={"file": ("document.pdf", data, "application/pdf")}
    )
    assert response.status_code == 413
    expected_detail = (
        "PDF exceeds the 100 page limit"
        if limit == "page-count"
        else "Rendered PDF page exceeds the 20000000 pixel limit"
    )
    assert response.json() == {"detail": expected_detail}


@pytest.mark.parametrize("declared_length", [None, "1"])
async def test_streamed_requests_cannot_bypass_size_limit(
    declared_length: str | None,
) -> None:
    """Count received multipart bytes even without a truthful Content-Length."""

    async def multipart_body() -> AsyncIterator[bytes]:
        yield (
            b"--upload\r\n"
            b'Content-Disposition: form-data; name="file"; filename="large.pdf"\r\n'
            b"Content-Type: application/pdf\r\n\r\n"
        )
        chunk = b"x" * 1_048_576
        for _ in range(50):
            yield chunk
        yield b"\r\n--upload--\r\n"

    headers = {"Content-Type": "multipart/form-data; boundary=upload"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/ocr/v1/predict", content=multipart_body(), headers=headers
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "HTTP request exceeds the 50 MiB limit"}


async def test_healthcheck_remains_responsive_during_pdf_preflight(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run native PDF inspection off-loop while holding an inference slot."""
    entered = threading.Event()
    release = threading.Event()
    original_trace = pymupdf.Page.get_texttrace

    def delayed_trace(page: pymupdf.Page) -> list[dict[str, Any]]:
        entered.set()
        release.wait(timeout=2)
        return original_trace(page)

    monkeypatch.setattr(pymupdf.Page, "get_texttrace", delayed_trace)
    request = asyncio.create_task(
        client.post(
            "/ocr/v1/sandwich",
            files={"file": ("native.pdf", _native_pdf(), "application/pdf")},
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        health = await asyncio.wait_for(client.get("/healthz"), timeout=0.5)
        assert health.status_code == 200
        assert not request.done()
        setup = await client.post("/ocr/v1/setup", json={"concurrency": 1})
        assert setup.json()["active_tasks"] == 1
    finally:
        release.set()
        result = await request

    assert result.status_code == 200
    assert result.headers["x-born-digital"] == "true"


@pytest.mark.parametrize("endpoint", ["/ocr/v1/predict", "/ocr/v1/sandwich"])
async def test_malformed_pdf_returns_a_client_error(
    client: AsyncClient, endpoint: str
) -> None:
    """Keep parse failures distinct from exceeded resource budgets."""
    response = await client.post(
        endpoint,
        files={"file": ("invalid.pdf", b"not a PDF", "application/pdf")},
    )
    assert response.status_code == 400
    assert "Failed to parse PDF document" in response.json()["detail"]


@pytest.mark.parametrize("declared_length", [None, "52428800"])
async def test_requests_at_exactly_50_mib_are_accepted(
    client: AsyncClient, declared_length: str | None
) -> None:
    """Include the exact boundary, with and without a Content-Length header."""

    async def json_body() -> AsyncIterator[bytes]:
        value = b'{"concurrency":1}'
        yield value
        remaining = 52_428_800 - len(value)
        chunk = b" " * 1_048_576
        while remaining:
            part = chunk[:remaining]
            yield part
            remaining -= len(part)

    headers = {"Content-Type": "application/json"}
    if declared_length is not None:
        headers["Content-Length"] = declared_length
    response = await client.post("/ocr/v1/setup", content=json_body(), headers=headers)
    assert response.status_code == 200
    assert response.json()["concurrency"] == 1


async def test_mixed_pdf_gets_searchable_scan_without_duplicate_native_text(
    client: AsyncClient, native_rapid: MagicMock
) -> None:
    """Preserve native pages and OCR the minority scan page with correct metadata."""
    native_rapid.return_value = (
        [([[10, 10], [200, 10], [200, 30], [10, 30]], "Recognized scan", 0.99)],
        None,
    )
    with pymupdf.open(stream=_native_pdf(page_count=2), filetype="pdf") as doc:
        original_pixels = doc[0].get_pixmap().samples
        scan = doc.new_page(width=300, height=400)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.clear_with(255)
        scan.insert_image(scan.rect, pixmap=pix)
        data = doc.tobytes()
    result = await client.post(
        "/ocr/v1/sandwich", files={"file": ("mixed.pdf", data, "application/pdf")}
    )
    assert result.status_code == 200
    assert result.headers["x-total-pages"] == "3"
    assert "x-born-digital" not in result.headers
    with pymupdf.open(stream=result.content, filetype="pdf") as doc:
        assert doc[0].get_pixmap().samples == original_pixels
        assert doc[0].get_text().count("A native document") == 1
        assert "Recognized scan" in doc[2].get_text()


async def test_cancelled_request_holds_slot_until_native_worker_finishes(
    client: AsyncClient, native_rapid: MagicMock
) -> None:
    """Repeated cancellation must not permit overlapping inference at capacity 1."""
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()
    lock = threading.Lock()
    calls = 0

    def blocked_native(image: object) -> tuple[None, None]:
        nonlocal calls
        with lock:
            calls += 1
            if calls == 1:
                entered.set()
            else:
                second_entered.set()
        release.wait(timeout=3)
        return None, None

    native_rapid.side_effect = blocked_native

    async def predict() -> object:
        return await client.post(
            "/ocr/v1/predict",
            files={"file": ("native.pdf", _native_pdf(), "application/pdf")},
        )

    first = asyncio.create_task(predict())
    second: asyncio.Task[object] | None = None
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        first.cancel()
        await asyncio.sleep(0)
        second = asyncio.create_task(predict())
        first.cancel()
        await asyncio.sleep(0.05)
        assert not second_entered.is_set()
        state = await client.post("/ocr/v1/setup", json={"concurrency": 1})
        assert state.json()["active_tasks"] == 1
    finally:
        release.set()
        tasks = [first] if second is None else [first, second]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)


async def test_cpu_fallback_returns_http_503_not_success(
    client: AsyncClient, native_rapid: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness and prediction reject fallback in the classification stage."""
    monkeypatch.setattr("rocm_ocr.main._fatal_exit", MagicMock())
    native_rapid.text_cls.infer.session.get_providers.return_value = [
        "CPUExecutionProvider"
    ]
    ready = await client.get("/readyz")
    assert ready.status_code == 503
    result = await client.post(
        "/ocr/v1/predict",
        files={"file": ("document.pdf", _native_pdf(), "application/pdf")},
    )
    assert result.status_code == 503
    assert result.json()["provider_attempted"] == "ROCMExecutionProvider"
    assert "Classification uses" in result.json()["error_message"]
    await asyncio.sleep(0.6)
