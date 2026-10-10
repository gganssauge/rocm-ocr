# Agent Guidelines & Repository Map

Standalone, high-performance OCR inference microservice accelerated by AMD ROCm GPUs using RapidOCR and ONNX Runtime.

> [!IMPORTANT]
> **Domain Independence:** This service is strictly a domain-agnostic OCR inference service. It must remain completely decoupled from downstream consumers (e.g., Paperless-ngx). Do not import or couple downstream business logic into this repository.

## Essential Core Rules

- **Package Manager & Execution:** Always use `uv`. Execute Python commands and scripts via `uv run <command>` (never bare `python3` or `pip`).
- **Python Version Constraint:** Strictly **Python 3.12**. Prebuilt AMD ROCm binary wheels (`onnxruntime-rocm`) exist for Python 3.12, but not for Python 3.14. Never attempt to upgrade the project's base Python past wheel availability.
- **No Ad-Hoc `-c` One-Liners:** Never execute `python -c "..."` one-liners! This breaks terminal approval prefix-matching. Use fixed scripts in `.scratch/` or standard CLI commands (`uv run pytest`, etc.).
- **Git Branching:** Direct pushes to `main` are strictly forbidden. Always use feature branches (`feature/`, `fix/`, `chore/`) and conventional commits.
- **Spec Gate (No Vibe Coding):** All implementation work must strictly conform to [`docs/spec.md`](docs/spec.md). No unannounced architectural detours or speculative code changes.
- **Code Standards & SonarCloud:**
  - Cognitive complexity $\le 15$ per function/method (SonarCloud standard).
  - Maximum function length: 30 logical lines. Extract focused private helpers with type annotations.
  - Fail-fast & transparency: Hardware/GPU errors must be reported transparently via HTTP 503; no silent CPU fallbacks.
- **Quality Gates:** Pre-commit hooks (`prek`) enforce `ruff format`, `ruff check`, and `markdownlint-cli2`.

## Key Commands

- Run single test: `uv run pytest tests/test_<module>.py -k test_<func>`
- Run all tests: `uv run pytest`
- Local dev server: `uv run uvicorn rocm_ocr.main:app --host 0.0.0.0 --port 8000 --reload`
- Run pre-commit checks: `uv run prek run --all-files`

## Progressive Disclosure & Reference

- [docs/spec.md](docs/spec.md) — Binding specification: API contract (`/ocr/v1/setup`, `/ocr/v1/predict`, `/healthz`, `/readyz`), concurrency semantics, data models, and error handling.
