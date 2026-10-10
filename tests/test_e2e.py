"""End-to-End (E2E) integration tests for the ROCm OCR microservice."""

import asyncio
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from rocm_ocr.main import app

PDF_FIXTURE_PATH = Path("2025-08-25 Vodafone Vodafone Aktivierungscode.pdf")
TXT_FIXTURE_PATH = Path("2025-08-25 Vodafone Vodafone Aktivierungscode.txt")


@pytest.fixture
async def e2e_client():
    """Create async test client with lifespan context for E2E tests."""
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        yield client


@pytest.mark.skipif(
    not PDF_FIXTURE_PATH.is_file(),
    reason="Vodafone test fixture PDF not present",
)
@pytest.mark.asyncio
async def test_e2e_predict_vodafone_pdf(e2e_client: AsyncClient) -> None:
    """Verify end-to-end OCR inference on the Vodafone activation code PDF."""
    pdf_bytes = PDF_FIXTURE_PATH.read_bytes()

    files = {"file": (PDF_FIXTURE_PATH.name, pdf_bytes, "application/pdf")}
    response = await e2e_client.post("/ocr/v1/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "execution_provider" in data
    assert data["inference_time_ms"] > 0
    assert len(data["pages"]) == 2

    first_page = data["pages"][0]
    assert first_page["page_number"] == 0
    assert first_page["width"] > 0
    assert first_page["height"] > 0

    all_texts = [b["text"] for b in first_page["boxes"]]
    combined_text = " ".join(all_texts).lower()

    # Verify key tokens present in the document
    assert "vodafone" in combined_text
    assert "aktivierungscode" in combined_text

    # Verify bounding box validity
    for box in first_page["boxes"]:
        x0, y0, x1, y1 = box["bbox"]
        assert 0 <= x0 <= x1 <= first_page["width"] + 10.0
        assert 0 <= y0 <= y1 <= first_page["height"] + 10.0
        assert 0.0 <= box["confidence"] <= 1.0


GERMAN_INVOICE_PDF = Path("tests/fixtures/sample_german_invoice.pdf")
GERMAN_INVOICE_GT = Path("tests/fixtures/sample_german_invoice.gt.txt")


@pytest.mark.asyncio
async def test_e2e_predict_german_invoice(e2e_client: AsyncClient) -> None:
    """Verify end-to-end OCR and German umlaut extraction on neutral invoice."""
    assert GERMAN_INVOICE_PDF.is_file(), f"Missing fixture {GERMAN_INVOICE_PDF}"
    pdf_bytes = GERMAN_INVOICE_PDF.read_bytes()

    files = {"file": (GERMAN_INVOICE_PDF.name, pdf_bytes, "application/pdf")}
    response = await e2e_client.post("/ocr/v1/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["inference_time_ms"] > 0
    assert len(data["pages"]) == 1

    first_page = data["pages"][0]
    all_text = " ".join(b["text"] for b in first_page["boxes"])

    # Verify extraction of German special characters and key fields
    assert "Musterstraße" in all_text
    assert "Fälligkeitsdatum" in all_text
    assert "München" in all_text
    assert "Geschäftsführer" in all_text
    assert "Rechnung Nr. 2026-0815" in all_text


@pytest.mark.asyncio
async def test_e2e_setup_and_concurrent_predict(e2e_client: AsyncClient) -> None:
    """Verify runtime concurrency adjustment and concurrent document inference."""
    assert GERMAN_INVOICE_PDF.is_file()
    pdf_bytes = GERMAN_INVOICE_PDF.read_bytes()

    setup_resp = await e2e_client.post("/ocr/v1/setup", json={"concurrency": 2})
    assert setup_resp.status_code == 200
    assert setup_resp.json()["concurrency"] == 2

    async def _send_predict() -> int:
        files = {"file": (GERMAN_INVOICE_PDF.name, pdf_bytes, "application/pdf")}
        resp = await e2e_client.post("/ocr/v1/predict", files=files)
        return resp.status_code

    results = await asyncio.gather(_send_predict(), _send_predict())
    assert results == [200, 200]

    # Reset back to default concurrency
    await e2e_client.post("/ocr/v1/setup", json={"concurrency": 1})
