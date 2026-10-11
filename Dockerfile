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
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv-bin /uv /uvx /bin/

RUN uv python install 3.12

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev --no-install-project --no-build --python 3.12

COPY src ./src
RUN uv sync --frozen --no-dev --python 3.12

# Download offline Scalar assets with cryptographic digest verification
ADD --checksum=sha256:62c34a0dda38945864011ebaf6e7fa7812a332429120d18922b34d01c4f7ccc8 \
    https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.73.1/dist/browser/standalone.js \
    /app/src/rocm_ocr/static/scalar.js

ADD --checksum=sha256:d16f72cf5a2773eba499cc7f3db1923c6f5f73de990f739e3a05806aafec15b7 \
    https://fastapi.tiangolo.com/img/favicon.png \
    /app/src/rocm_ocr/static/favicon.png

# Download PP-OCRv6 recognition model with cryptographic digest verification
ADD --checksum=sha256:5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634 \
    https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/b8f84f0b80c529de40b4fbb3544b84fa7233a513/inference.onnx \
    /models/PP-OCRv6_small_rec.onnx

# Prepare self-contained /models and generate stub libroctracer64.so.4
RUN mkdir -p /models /app/lib/rocm && \
    cp src/rocm_ocr/resources/ppocrv6_dict.txt /models/ppocrv6_dict.txt && \
    cp src/rocm_ocr/resources/latin_dict.txt /models/latin_dict.txt && \
    VENV_MODELS=$(find /app/.venv -type d -name "models" | head -n 1) && \
    cp "$VENV_MODELS/ch_PP-OCRv4_det_infer.onnx" /models/ch_PP-OCRv4_det_infer.onnx && \
    cp "$VENV_MODELS/ch_ppocr_mobile_v2.0_cls_infer.onnx" /models/ch_ppocr_mobile_v2.0_cls_infer.onnx && \
    gcc -shared -fPIC -o /app/lib/rocm/libroctracer64.so.4 -Wl,-soname,libroctracer64.so.4 -x c /dev/null

# --- Runtime Stage ---
FROM ubuntu:rolling AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    hip-utils \
    libamdhip64-7 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libhipblas3 \
    libhipfft0 \
    libmiopen1 \
    librocm-smi64-7 \
    rocminfo \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/bin/pebble \
    && mkdir -p /app/lib/rocm \
    && ln -sf /usr/lib/x86_64-linux-gnu/librocm_smi64.so.7 /app/lib/rocm/librocm_smi64.so.1 \
    && (getent group render || groupadd -g 109 render) \
    && usermod -a -G video,render ubuntu

# Copy standalone Python 3.12 and application venv
COPY --from=builder /opt/python /opt/python
COPY --from=builder --chown=ubuntu:ubuntu /app/.venv /app/.venv
COPY --from=builder /app/lib/rocm/libroctracer64.so.4 /app/lib/rocm/libroctracer64.so.4
COPY --from=builder --chown=ubuntu:ubuntu /app/src /app/src
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
