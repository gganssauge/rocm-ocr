# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.5.26 AS uv-bin

# --- Build Stage ---
FROM ubuntu:rolling AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_PYTHON=3.12
ENV UV_PYTHON_INSTALL_DIR=/opt/python

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /bin/

RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --no-install-project --python 3.12

COPY src ./src
RUN uv sync --frozen --no-dev --python 3.12

# Download Latin recognition model and prepare self-contained /models directory
RUN mkdir -p /models && \
    curl -fL -o /models/latin_PP-OCRv3_rec_infer.onnx \
      "https://huggingface.co/breezedeus/cnocr-ppocr-latin_PP-OCRv3/resolve/main/latin_PP-OCRv3_rec_infer.onnx" && \
    cp src/rocm_ocr/resources/latin_dict.txt /models/latin_dict.txt && \
    VENV_MODELS=$(find /app/.venv -type d -name "models" | head -n 1) && \
    cp "$VENV_MODELS/ch_PP-OCRv4_det_infer.onnx" /models/ch_PP-OCRv4_det_infer.onnx && \
    cp "$VENV_MODELS/ch_ppocr_mobile_v2.0_cls_infer.onnx" /models/ch_ppocr_mobile_v2.0_cls_infer.onnx

# Generate stub libroctracer64.so.4
RUN mkdir -p /app/lib/rocm && \
    gcc -shared -fPIC -o /app/lib/rocm/libroctracer64.so.4 -Wl,-soname,libroctracer64.so.4 -x c /dev/null

# --- Runtime Stage ---
FROM ubuntu:rolling AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    librocm-smi64-7 \
    libamdhip64-7 \
    libhipblas3 \
    libhipfft0 \
    libmiopen1 \
    libgomp1 \
    rocminfo \
    hip-utils \
    && rm -rf /var/lib/apt/lists/*

# Link librocm_smi64.so.1 to the system installed library
RUN mkdir -p /app/lib/rocm && \
    ln -sf /usr/lib/x86_64-linux-gnu/librocm_smi64.so.7 /app/lib/rocm/librocm_smi64.so.1

# Ensure render group exists and configure ubuntu user
RUN (getent group render || groupadd -g 109 render) && \
    usermod -a -G video,render ubuntu

# Copy standalone Python 3.12 and application venv
COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=ubuntu:ubuntu /app/.venv /app/.venv
COPY --from=builder /app/lib/rocm/libroctracer64.so.4 /app/lib/rocm/libroctracer64.so.4
COPY --chown=ubuntu:ubuntu src /app/src
COPY --from=builder --chown=ubuntu:ubuntu /models /models

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED="1" \
    LD_LIBRARY_PATH="/app/lib/rocm" \
    HSA_OVERRIDE_GFX_VERSION="11.0.0" \
    MIOPEN_DEBUG_CONV_WINOGRAD="0" \
    MIOPEN_FIND_MODE="2" \
    MIOPEN_LOG_LEVEL="0" \
    OCR_MODEL_DIR="/models" \
    PORT=8000

USER ubuntu

EXPOSE 8000

ENTRYPOINT ["uvicorn", "rocm_ocr.main:app", "--host", "0.0.0.0", "--port", "8000"]
