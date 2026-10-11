"""FastAPI application for the ROCm OCR microservice."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from scalar_fastapi import get_scalar_api_reference

from rocm_ocr.concurrency import DynamicConcurrencyLimiter
from rocm_ocr.config import settings
from rocm_ocr.engine import GpuInferenceError, OcrEngine
from rocm_ocr.logging import setup_logging
from rocm_ocr.models import (
    ErrorResponse,
    HealthResponse,
    PageResult,
    PredictResponse,
    ReadyResponse,
    SetupRequest,
    SetupResponse,
)
from rocm_ocr.parser import UnsupportedFormatError, parse_document

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize application state, concurrency limiter, and OCR engine."""
    setup_logging()
    app.state.limiter = DynamicConcurrencyLimiter(settings.ocr_concurrency)
    app.state.engine = OcrEngine(settings)
    yield


app = FastAPI(
    title="ROCm OCR Microservice",
    description="High-performance OCR inference accelerated by AMD ROCm",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/scalar", include_in_schema=False)
async def scalar_html() -> HTMLResponse:
    """Serve interactive Scalar API documentation (air-gapped if assets present)."""
    has_local_js = (STATIC_DIR / "scalar.js").is_file()
    has_local_icon = (STATIC_DIR / "favicon.png").is_file()
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="ROCm OCR Microservice - API Documentation",
        scalar_js_url=(
            "/static/scalar.js"
            if has_local_js
            else "https://cdn.jsdelivr.net/npm/@scalar/api-reference"
        ),
        scalar_favicon_url=(
            "/static/favicon.png"
            if has_local_icon
            else "https://fastapi.tiangolo.com/img/favicon.png"
        ),
        with_default_fonts=False,
        telemetry=False,
        custom_css=(
            ":root { "
            "--scalar-font: system-ui, -apple-system, BlinkMacSystemFont, "
            "'Segoe UI', Roboto, Helvetica, Arial, sans-serif; "
            "--scalar-font-code: ui-monospace, SFMono-Regular, Menlo, Monaco, "
            "Consolas, monospace; "
            "}"
        ),
    )


@app.get("/docs", include_in_schema=False)
async def redirect_docs() -> RedirectResponse:
    """Redirect /docs to the self-contained /scalar API documentation."""
    return RedirectResponse(
        url="/scalar", status_code=status.HTTP_307_TEMPORARY_REDIRECT
    )


app.add_middleware(
    CorrelationIdMiddleware,
    header_name="X-Correlation-ID",
    validator=None,
)


@app.exception_handler(GpuInferenceError)
async def gpu_inference_exception_handler(
    request: object, exc: GpuInferenceError
) -> JSONResponse:
    """Return transparent 503 payload when GPU or driver error occurs."""
    content = ErrorResponse(
        detail=exc.detail,
        provider_attempted=exc.provider_attempted,
        error_message=exc.error_message,
    ).model_dump()
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=content
    )


@app.get("/healthz")
async def healthz() -> HealthResponse:
    """Liveness probe returning HTTP 200 when server is listening."""
    return HealthResponse(status="ok")


@app.get("/readyz")
async def readyz() -> ReadyResponse:
    """Readiness probe verifying engine and execution provider health."""
    engine: OcrEngine = app.state.engine
    if not engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR engine or execution provider not ready",
        )
    return ReadyResponse(status="ready", provider=engine.active_provider)


@app.post("/ocr/v1/setup")
async def setup(request: SetupRequest) -> SetupResponse:
    """Dynamically adjust concurrency limits for OCR inference."""
    limiter: DynamicConcurrencyLimiter = app.state.limiter
    await limiter.update_concurrency(request.concurrency)
    return SetupResponse(
        status="updated",
        concurrency=limiter.capacity,
        active_tasks=limiter.active_tasks,
    )


def _process_inference_pages(
    engine: OcrEngine, file_bytes: bytes, filename: str
) -> list[PageResult]:
    """Parse document pages and run OCR inference on each rendered page."""
    try:
        rendered_pages = parse_document(file_bytes, filename=filename)
    except UnsupportedFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    page_results: list[PageResult] = []
    for page in rendered_pages:
        boxes = engine.infer_page(page)
        page_results.append(
            PageResult(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                boxes=boxes,
            )
        )
    return page_results


@app.post("/ocr/v1/predict")
async def predict(file: Annotated[UploadFile, File()]) -> PredictResponse:
    """Execute OCR inference on an uploaded PDF or image file."""
    file_bytes = await file.read()
    filename = file.filename or ""

    limiter: DynamicConcurrencyLimiter = app.state.limiter
    engine: OcrEngine = app.state.engine

    start_time = time.perf_counter()
    async with limiter:
        pages = _process_inference_pages(engine, file_bytes, filename)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return PredictResponse(
        execution_provider=engine.active_provider,
        inference_time_ms=elapsed_ms,
        pages=pages,
    )
