"""FastAPI application for the ROCm OCR microservice."""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from rocm_ocr.concurrency import DynamicConcurrencyLimiter
from rocm_ocr.config import settings
from rocm_ocr.engine import GpuInferenceError, OcrEngine
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize application state, concurrency limiter, and OCR engine."""
    app.state.limiter = DynamicConcurrencyLimiter(settings.ocr_concurrency)
    app.state.engine = OcrEngine(settings)
    yield


app = FastAPI(
    title="ROCm OCR Microservice",
    description="High-performance OCR inference accelerated by AMD ROCm",
    version="0.1.0",
    lifespan=lifespan,
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


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness probe returning HTTP 200 when server is listening."""
    return HealthResponse(status="ok")


@app.get("/readyz", response_model=ReadyResponse)
async def readyz() -> ReadyResponse:
    """Readiness probe verifying engine and execution provider health."""
    engine: OcrEngine = app.state.engine
    if not engine.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR engine or execution provider not ready",
        )
    return ReadyResponse(status="ready", provider=engine.active_provider)


@app.post("/ocr/v1/setup", response_model=SetupResponse)
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


@app.post("/ocr/v1/predict", response_model=PredictResponse)
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
