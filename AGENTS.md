# Agent Guidelines & Repository Map

Standalone, high-performance OCR inference microservice accelerated by AMD ROCm GPUs using RapidOCR and ONNX Runtime.

## Universal Invariants

- **Domain Independence:** Strictly domain-agnostic OCR inference service. Never import or couple downstream business logic (e.g., Paperless-ngx).
- **GPU Transparency & Fail-Fast:** Report hardware and GPU errors transparently via HTTP 503; no silent CPU fallbacks.
- **Runtime:** Strictly **Python 3.12** managed with `uv`. Never bump Python past 3.12 (`onnxruntime-rocm` wheels exist only for 3.12).
- **Type Checking:** Run `ty check`.

## Progressive Disclosure & Reference

Read the relevant document before taking action on that branch:

- [docs/spec.md](docs/spec.md) — API contract (`/ocr/v1/setup`, `/ocr/v1/predict`, `/healthz`, `/readyz`), concurrency semantics, data models, and error responses.
- [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) — Pipeline stages (PP-OCRv4 detection on ROCm, PP-OCRv3 recognition on CPU), RDNA3 compatibility flags, local dev server, and C-level stderr silencing.
- [docs/spec-k8s-packaging.md](docs/spec-k8s-packaging.md) — Kubernetes manifests, container packaging, and GPU device mounts.
