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

### 2.3 `GET /healthz` & `GET /readyz`

- `GET /healthz`: Returns HTTP 200 as soon as the HTTP server is listening.
- `GET /readyz`: Verifies that ONNX Runtime and the configured Execution Provider (`ROCMExecutionProvider` or fallback) are loaded and capable of executing inference.

---

## 3. Environment Variables & Defaults

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | Required for RDNA3 consumer cards (e.g. RX 7800 XT / `gfx1101`). |
| `OCR_CONCURRENCY` | `1` | Default initial concurrency (overridable at runtime via `/setup`). |
| `OCR_MODEL_DIR` | `/models` | Path containing PP-OCRv4 ONNX model files. |
| `LOG_LEVEL` | `INFO` | Standard logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `PORT` | `8000` | HTTP listening port. |

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
