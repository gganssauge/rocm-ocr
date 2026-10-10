#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIB_DIR="$REPO_ROOT/lib/rocm"

mkdir -p "$LIB_DIR"

# 1. Locate and link librocm_smi64.so to librocm_smi64.so.1
SMI_LIB=$(find /usr/lib/x86_64-linux-gnu /opt/rocm/lib -name "librocm_smi64.so*" 2>/dev/null | grep -E "so(\.[0-9]+)?$" | head -n 1 || true)
if [ -n "$SMI_LIB" ]; then
    ln -sf "$SMI_LIB" "$LIB_DIR/librocm_smi64.so.1"
    echo "Linked $SMI_LIB -> $LIB_DIR/librocm_smi64.so.1"
fi

# 2. Build stub libroctracer64.so.4 if not provided by system
TRACER_LIB=$(find /usr/lib/x86_64-linux-gnu /opt/rocm/lib -name "libroctracer64.so*" 2>/dev/null | head -n 1 || true)
if [ -n "$TRACER_LIB" ]; then
    ln -sf "$TRACER_LIB" "$LIB_DIR/libroctracer64.so.4"
    echo "Linked $TRACER_LIB -> $LIB_DIR/libroctracer64.so.4"
elif [ ! -f "$LIB_DIR/libroctracer64.so.4" ]; then
    echo "Building stub libroctracer64.so.4..."
    gcc -shared -fPIC -o "$LIB_DIR/libroctracer64.so.4" -Wl,-soname,libroctracer64.so.4 -x c /dev/null
fi

echo "ROCm compatibility libraries configured in $LIB_DIR"
