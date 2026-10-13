#!/bin/bash

platform_deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if npu-smi info >/dev/null 2>&1; then
    source "${platform_deps_dir}/npu.sh"
elif nvidia-smi -L >/dev/null 2>&1; then
    source "${platform_deps_dir}/gpu.sh"
else
    echo "Error: unable to detect an available NPU or GPU." >&2
    exit 1
fi
