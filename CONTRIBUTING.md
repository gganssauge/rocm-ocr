# Contributing to ROCm-OCR

Thank you for your interest in contributing to `rocm-ocr`!

## Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to `ganssauge@gorki120.de`.

## Development Setup

The project uses [`uv`](https://docs.astral.sh/uv/) for Python package and virtual environment management. Due to AMD ROCm ONNX Runtime binary compatibility, Python 3.12 is required.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gganssauge/rocm-ocr.git
   cd rocm-ocr
   ```

2. **Set up Python 3.12 and dependencies:**

   ```bash
   uv python install 3.12
   uv sync --all-groups
   ```

3. **Install Pre-Commit Hooks:**
   We use `prek` (configured via `.pre-commit-config.yaml`):

   ```bash
   uv run prek install
   ```

## Development Workflow

- **Branching Strategy:** Direct pushes to `main` are strictly forbidden. Always create a feature branch:

  ```bash
  git checkout -b feature/<descriptive-name>
  ```

- **Commit Messages:** Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat: add sandwich pdf generation`
  - `fix: handle gpu memory cleanup on crash`
  - `docs: update deployment architecture`
  - `chore: update dependencies`

## Code Standards & Quality Gates

- **Formatting & Linting:**

  ```bash
  uv run ruff check .
  uv run ruff format --check .
  ```

- **Type Checking:**

  ```bash
  uv run ty check
  ```

- **Testing & Coverage:**

  ```bash
  LD_LIBRARY_PATH="$PWD/lib/rocm${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    uv run pytest --cov=rocm_ocr --cov-report=term-missing
  ```

  GPU-independent tests can be run with `uv run pytest --ignore=tests/test_e2e.py`.
  The full suite requires a working ROCm installation and the local compatibility
  libraries described in the [Quickstart](README.md#-quickstart).

- **Function Metrics:**
  - Keep functions focused and concise (maximum 30 logical lines).
  - Maximum cognitive complexity per function/method: $\le 15$ (SonarCloud standard).

## Pull Request Guidelines

1. Ensure all tests and lint checks pass locally before opening a PR.
2. Link the relevant issue in the PR description (e.g., `Closes #12`).
3. Keep PRs focused on a single responsibility.
