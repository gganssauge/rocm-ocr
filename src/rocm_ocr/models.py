"""Data models and API schemas for the ROCm OCR service."""

from pydantic import BaseModel, Field


class SetupRequest(BaseModel):
    """Request payload to dynamically adjust concurrency limits."""

    concurrency: int = Field(
        gt=0,
        description="Target maximum concurrent inference tasks (must be >= 1)",
    )


class SetupResponse(BaseModel):
    """Response returned when concurrency has been updated."""

    status: str = Field(default="updated", description="Operation status string")
    concurrency: int = Field(description="Active maximum concurrency limit")
    active_tasks: int = Field(description="Currently active tasks executing OCR")


class BoundingBox(BaseModel):
    """Detected text bounding box and transcription result."""

    text: str = Field(description="Recognized text string")
    bbox: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box [x0, y0, x1, y1] (top-left to bottom-right)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )


class PageResult(BaseModel):
    """OCR results for a single rendered document page or image."""

    page_number: int = Field(description="Zero-based index of the page")
    width: float = Field(description="Page width in points or pixels")
    height: float = Field(description="Page height in points or pixels")
    boxes: list[BoundingBox] = Field(
        default_factory=list,
        description="List of detected text bounding boxes",
    )


class PredictResponse(BaseModel):
    """Response returned upon successful OCR execution."""

    execution_provider: str = Field(
        description="Active execution provider (e.g. ROCMExecutionProvider)",
    )
    inference_time_ms: float = Field(
        description="Total duration of inference in milliseconds",
    )
    pages: list[PageResult] = Field(
        default_factory=list,
        description="Per-page OCR extraction results",
    )


class ErrorResponse(BaseModel):
    """Structured error payload for hardware or driver failures."""

    detail: str = Field(description="High-level description of failure")
    provider_attempted: str = Field(
        description="Execution provider that failed (e.g. ROCMExecutionProvider)",
    )
    error_message: str = Field(description="Underlying driver or runtime error message")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="ok", description="Server liveness status")


class ReadyResponse(BaseModel):
    """Readiness check response schema."""

    status: str = Field(default="ready", description="Engine readiness status")
    provider: str = Field(description="Loaded execution provider")
