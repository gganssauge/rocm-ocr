#!/usr/bin/env bash
# Download standalone Scalar static assets for local offline development.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_DIR="${SCRIPT_DIR}/../src/rocm_ocr/static"

mkdir -p "${STATIC_DIR}"

echo "Downloading Scalar static assets to ${STATIC_DIR}..."
curl -fL -o "${STATIC_DIR}/scalar.js" "https://cdn.jsdelivr.net/npm/@scalar/api-reference"
curl -fL -o "${STATIC_DIR}/favicon.png" "https://fastapi.tiangolo.com/img/favicon.png"
echo "Scalar static assets downloaded successfully."
