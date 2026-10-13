#!/bin/bash

deps_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${deps_dir}/platform.sh"

# Use Transformers 5.5.4 while 5.6.0 mishandles Qwen3.5 FA2 with s_aux=None.
pip install transformers==5.5.4

# dependency of llama_factory
pip install safetensors==0.7.0
pip install deepspeed==0.16.0
pip install tensorboard
pip install psutil==7.2.2
pip install omegaconf
pip install pyparsing
pip install cycler
pip install kiwisolver
pip install peft==0.18.1
pip install torchaudio==2.10.0
pip install datasets==4.0.0
pip install accelerate==1.11.0
pip install trl==0.24.0
pip install matplotlib

pip install llamafactory==0.9.5

pip install numpy==2.0
