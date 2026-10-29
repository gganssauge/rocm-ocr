"""End-to-End (E2E) integration tests for the ROCm OCR microservice."""

import asyncio
from pathlib import Path
from threading import Barrier

import pytest
from httpx import ASGITransport, AsyncClient, Response

from rocm_ocr.engine import ROCM_PROVIDER, OcrEngine
from rocm_ocr.main import app
from rocm_ocr.models import BoundingBox
from rocm_ocr.parser import RenderedPage

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
    assert "München" in all_text
    assert "Fälligkeitsdatum" in all_text
    assert "Umsatzsteuer" in all_text
    assert "RE-2026-0042" in all_text
    assert "Kurfürstendamm" in all_text


BTD_PDF_PATH = Path("tests/fixtures/bundestag_drucksache_sample.pdf")


@pytest.mark.asyncio
async def test_e2e_predict_bundestag_drucksache(e2e_client: AsyncClient) -> None:
    """Verify end-to-end OCR on official Bundestag publication."""
    assert BTD_PDF_PATH.is_file(), f"Missing fixture {BTD_PDF_PATH}"
    pdf_bytes = BTD_PDF_PATH.read_bytes()

    files = {"file": (BTD_PDF_PATH.name, pdf_bytes, "application/pdf")}
    response = await e2e_client.post("/ocr/v1/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["inference_time_ms"] > 0
    assert len(data["pages"]) == 1

    first_page = data["pages"][0]
    all_text = " ".join(b["text"] for b in first_page["boxes"])

    # Verify official header and key contents from Bundestag publication
    assert "Deutscher Bundestag" in all_text
    assert "Drucksache 20/100" in all_text
    assert "20. Wahlperiode" in all_text
    assert "Existenzminimum" in all_text


@pytest.mark.asyncio
async def test_e2e_setup_and_concurrent_predict(
    e2e_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require overlapping OCR workers and unchanged results from real GPU inference."""
    assert GERMAN_INVOICE_PDF.is_file()
    pdf_bytes = GERMAN_INVOICE_PDF.read_bytes()

    setup_resp = await e2e_client.post("/ocr/v1/setup", json={"concurrency": 2})
    assert setup_resp.status_code == 200
    assert setup_resp.json()["concurrency"] == 2

    workers = Barrier(2)
    original_infer = OcrEngine.infer_page

    def concurrent_infer(engine: OcrEngine, page: RenderedPage) -> list[BoundingBox]:
        workers.wait(timeout=5)
        return original_infer(engine, page)

    async def _send_predict() -> Response:
        files = {"file": (GERMAN_INVOICE_PDF.name, pdf_bytes, "application/pdf")}
        return await e2e_client.post("/ocr/v1/predict", files=files)

    try:
        with monkeypatch.context() as context:
            context.setattr(OcrEngine, "infer_page", concurrent_infer)
            responses = await asyncio.gather(_send_predict(), _send_predict())
        assert [response.status_code for response in responses] == [200, 200]

        baseline = await _send_predict()
        assert baseline.status_code == 200
        expected_pages = baseline.json()["pages"]
        assert "RE-2026-0042" in " ".join(
            box["text"] for page in expected_pages for box in page["boxes"]
        )
        for response in responses:
            assert response.json()["execution_provider"] == ROCM_PROVIDER
            assert response.json()["pages"] == expected_pages
    finally:
        await e2e_client.post("/ocr/v1/setup", json={"concurrency": 1})
