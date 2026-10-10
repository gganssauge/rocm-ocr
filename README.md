# rocm-ocr

Standalone, high-performance OCR inference microservice accelerated by AMD ROCm GPUs using RapidOCR and ONNX Runtime.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![AMD ROCm](https://img.shields.io/badge/ROCm-7.1%20%7C%20RDNA3-red.svg)](https://rocm.docs.amd.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

`rocm-ocr` executes PP-OCRv4 and PP-OCRv3 models (text detection, angle classification, and recognition) accelerated on AMD ROCm consumer and datacenter GPUs (specifically tuned for AMD Radeon RX 7800 XT / RDNA3 / `gfx1101`).

- **Architecture:** Clean, domain-agnostic asynchronous HTTP REST API (FastAPI).
- **Concurrency Control:** Dynamic `/ocr/v1/setup` endpoint to tune concurrency at runtime.
- **Microservice Ready:** Built-in `X-Correlation-ID` tracing across logs and responses via `asgi-correlation-id`.
- **Fail-Fast:** Hardware/GPU errors are reported transparently via HTTP 503; no silent CPU fallbacks.
- **Documentation:** Complete specification and deployment guide in [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).

---

## Performance Benchmark

In-cluster benchmark on AMD Radeon RX 7800 XT (`gfx1101`) sharing GPU resources with Ollama:

| Document | Pure CPU Baseline | **ROCm GPU Microservice** | Speedup |
| :--- | :--- | :--- | :--- |
| **Standard Invoice (59 boxes, PDF)** | 4.15 s | **1.36 s** | **~3.05×** |
| **Bundestagsdrucksache (38 boxes, PDF)** | 5.15 s | **2.38 s** | **~2.16×** |

*Note: Latency includes HTTP request/response transfer, PDF rasterization (144 DPI), GPU detection, and text recognition.*

---

## Quickstart

### Local Development (Python 3.12 strictly via `uv`)

```bash
# Install dependencies
uv sync --all-groups

# Start local server
uv run uvicorn rocm_ocr.main:app --host 0.0.0.0 --port 8000 --reload

# Run test suite
uv run pytest
```

### Docker Build & Run

```bash
# Build multi-stage ROCm container
docker build -t rocm-ocr:local .

# Run container with AMD GPU access
docker run --rm -it -p 8000:8000 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  rocm-ocr:local
```

### Kubernetes / k3s Deployment

Manifests are provided in [`deploy/k8s/`](deploy/k8s/):

```bash
kubectl apply -k deploy/k8s/
kubectl rollout status deployment/rocm-ocr -n ocr
```

---

## API Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/healthz` | Liveness probe (HTTP 200). Routinely filtered from access logs. |
| `GET` | `/readyz` | Readiness probe verifying GPU provider and OCR engine health. |
| `POST` | `/ocr/v1/setup` | Dynamically update concurrency limit at runtime. |
| `POST` | `/ocr/v1/predict` | Upload PDF or image (`multipart/form-data`) for OCR text & bounding boxes. |

Full schema definitions and request/response examples are documented in [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).
