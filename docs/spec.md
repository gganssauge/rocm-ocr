# ROCm OCR Microservice Specification

- **Project:** `rocm-ocr`
- **Purpose:** Standalone, high-performance OCR inference microservice accelerated by AMD ROCm GPUs.
- **Engine:** RapidOCR running PP-OCRv4 models (detection, angle classification, text recognition) via ONNX Runtime with AMD `ROCMExecutionProvider`.
- **Target Hardware:** AMD Radeon RX 7800 XT (Navi 32 / RDNA3 / `gfx1101`) & AMD ROCm 7.x (`/dev/kfd`, `/dev/dri`).

---

## 1. Architectural Scope

`rocm-ocr` is strictly a domain-agnostic inference service. It has no knowledge of downstream consumers (such as Paperless-ngx, document management systems, or archivers).

### Core Responsibilities

1. Accept PDF files or image streams via HTTP.
2. Render pages (for PDFs) and run PP-OCRv4 detection, direction classification, and text recognition on the AMD GPU.
3. Return structured JSON with token/line bounding boxes, recognized text, confidence scores, and execution metadata (active execution provider, inference duration).
4. Provide a dynamic `/setup` endpoint to tune runtime concurrency (e.g., during bulk reindexing) without requiring container restarts.
5. Provide transparent error reporting (fail-fast on GPU/kernel crashes; no silent or unannounced fallbacks).

---

## 2. API Contract

The service exposes HTTP endpoints via FastAPI.

### 2.0 Authentication (Bearer Token)

The service supports optional pre-shared key (PSK) authentication via standard HTTP Bearer tokens.

- **Enforcement:** If `API_KEY` is set to a non-empty string, all operational endpoints under `/ocr/v1/*` (`/setup`, `/predict`, `/sandwich`) require `Authorization: Bearer <API_KEY>`.
- **Timing Protection:** Token comparison uses constant-time validation (`secrets.compare_digest`).
- **Unauthenticated Probes:** Health probes (`/healthz`, `/readyz`) and API documentation (`/scalar`, `/docs`, `/openapi.json`) remain accessible without credentials.
- **Unauthorized Response (HTTP 401):**

  ```json
  {
    "detail": "Invalid or missing API key"
  }
  ```

- **Development/Testing:** If `API_KEY` is unset or empty, authentication is disabled.

### 2.1 `POST /ocr/v1/setup`

Dynamically adjusts runtime parameters such as maximum concurrency.

- **Initial State:** `concurrency = 1` on service startup.
- **Request:**

  ```json
  {
    "concurrency": 4
  }
  ```

- **Response (HTTP 200):**

  ```json
  {
    "status": "updated",
    "concurrency": 4,
    "active_tasks": 0
  }
  ```

- **Semantics:** Concurrency is managed via an `asyncio.Semaphore` (or token pool). Changing this value takes effect immediately for incoming requests.

### 2.2 `POST /ocr/v1/predict`

Executes OCR inference on an uploaded document or image.

- **Request:**
  - Content-Type: `multipart/form-data`
  - Body: `file` (Binary content of PDF, PNG, JPEG, TIFF, or WebP)
- **Response (HTTP 200 - Success):**

  ```json
  {
    "execution_provider": "ROCMExecutionProvider",
    "inference_time_ms": 284.5,
    "pages": [
      {
        "page_number": 0,
        "width": 595.0,
        "height": 842.0,
        "boxes": [
          {
            "text": "Invoice Number: 2026-0042",
            "bbox": [72.0, 100.5, 250.0, 115.0],
            "confidence": 0.985
          }
        ]
      }
    ]
  }
  ```

- **Bounding Box Convention:**
  `[x0, y0, x1, y1]` in points (for PDFs) or pixels (for images), where `(x0, y0)` is top-left and `(x1, y1)` is bottom-right.
- **Response (HTTP 503 / 500 - Hardware/Driver Error):**
  Transparent error reporting without silent CPU fallback:

  ```json
  {
    "detail": "ROCm GPU inference failed",
    "provider_attempted": "ROCMExecutionProvider",
    "error_message": "HIP out of memory error on device gfx1101"
  }
  ```

### 2.3 `POST /ocr/v1/sandwich`

Executes OCR inference on an uploaded PDF and returns a searchable sandwich PDF with an invisible text layer (`render_mode=3`).

- **Request:**
  - Content-Type: `multipart/form-data`
  - Body: `file` (Binary content of PDF)
- **Response (HTTP 200 - Success):**
  - Content-Type: `application/pdf`
  - Headers:
    - `X-Inference-Time-Ms`: Total inference duration in milliseconds
    - `X-Execution-Provider`: Active execution provider (`ROCMExecutionProvider`)
    - `X-Total-Pages`: Number of processed pages
    - `X-Extracted-Chars`: Total number of recognized characters
- **Response (HTTP 400 - Bad Request):**
  - Returned if the uploaded file is not a valid PDF document.

### 2.4 `GET /healthz` & `GET /readyz`

- `GET /healthz`: Returns HTTP 200 as soon as the HTTP server is listening.
- `GET /readyz`: Verifies that ONNX Runtime and the configured Execution Provider (`ROCMExecutionProvider` or fallback) are loaded and capable of executing inference.

### 2.5 `GET /scalar` & `GET /docs`

- `GET /scalar`: Interactive API documentation powered by Scalar (`scalar-fastapi`). In container deployments, frontend assets (`scalar.js`, `favicon.png`) are pre-baked at build time to provide fully air-gapped operation without external CDN dependencies.
- `GET /docs`: Redirects (HTTP 307) to `/scalar`.

### 2.6 Document Resource Limits

The following inclusive limits apply to uploaded documents:

| Resource | Maximum |
| :--- | :--- |
| Complete HTTP request body, including all multipart fields and framing | 50 MiB (52,428,800 bytes) |
| PDF pages | 100 |
| Pixels per rendered PDF page or decoded raster image | 20,000,000 |

- HTTP body bytes are counted as they arrive, including requests without
  `Content-Length`. An oversized declared length is rejected before multipart parsing.
- PDF pixel dimensions are checked at the actual rendering resolution (144 DPI),
  rounding outwards to integer pixels. Image dimensions are checked before RGB decoding.
- PDF limits also apply to born-digital documents passed through by `/sandwich`.
- PDF pages are rendered and processed sequentially, not retained as a complete
  rasterized document. Reading uploaded files into memory and the PDF preflight
  inspection share the inference concurrency limiter.
- PDF inspection runs outside the event loop so that health checks remain responsive.
- Oversized input returns HTTP 413 with `{"detail": "<limit description>"}`.
  Malformed input continues to return HTTP 400. Documents previously accepted above
  these limits must be split or downscaled by the caller.

These are per-request limits, not rate limits or sandboxing for native parsers.
Production deployments should also restrict access and bound aggregate upload traffic.

---

## 3. Environment Variables & Defaults

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | Required for RDNA3 consumer cards (e.g. RX 7800 XT / `gfx1101`). |
| `OCR_CONCURRENCY` | `1` | Default initial concurrency (overridable at runtime via `/setup`). |
| `OCR_MODEL_DIR` | `/models` | Path containing PP-OCRv4 ONNX model files. |
| `LOG_LEVEL` | `INFO` | Standard logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `PORT` | `8000` | HTTP listening port. |
| `API_KEY` | `""` | Optional pre-shared Bearer token protecting `/ocr/v1/*` endpoints. |

---

## 4. Model Artifacts

- PP-OCRv4 ONNX models (det, cls, rec) are baked into the Docker image at build time under `/models`.
- Allows completely offline, air-gapped operation in Kubernetes clusters.
- Can be overridden at runtime by mounting custom models to `OCR_MODEL_DIR`.

---

## 5. Engineering Standards & Quality Gates

- **Language & Runtime:** Python 3.12 (standard version with verified `onnxruntime-rocm` prebuilt wheels).
- **Package Management:** Exclusively Astral `uv` (`uv venv`, `uv sync`, `uv run`).
- **Code Standards:**
  - Cognitive Complexity $\le 15$ per function/method (SonarCloud standard).
  - Maximum function length: 30 lines.
  - Type annotations on all public functions.
  - Docstrings explaining parameters and return values.
- **Testing:** Unit tests with mocks for HTTP endpoints, concurrency limiter, and OCR engine.
