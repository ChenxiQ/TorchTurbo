#!/bin/bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../deps/lf_deps.sh"

pip install torchvision==0.25.0

rm -rf ~/.cache/huggingface/modules/transformers_modules/
