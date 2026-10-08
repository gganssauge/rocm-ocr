# rocm-ocr

Standalone, high-performance OCR inference microservice accelerated by AMD ROCm GPUs using RapidOCR and ONNX Runtime.

## Overview

`rocm-ocr` executes PP-OCRv4 models (text detection, direction classification, and recognition) on AMD ROCm devices (e.g. AMD Radeon RX 7800 XT / RDNA3 / `gfx1101`).

- **Architecture:** Clean, domain-agnostic HTTP REST API (FastAPI).
- **Concurrency Control:** Dynamic `/ocr/v1/setup` endpoint to tune concurrency at runtime.
- **Fail-Fast:** Hardware errors are reported transparently without silent CPU degradation.
- **Specification:** Detailed specification available in [docs/spec.md](docs/spec.md).

## Quickstart

```bash
uv sync --all-groups
uv run uvicorn rocm_ocr.main:app --host 0.0.0.0 --port 8000
```
