#!/usr/bin/env bash
set -euo pipefail

source /jizhicfs/lhy/activate

if ! conda env list | awk '{print $1}' | grep -qx llamafactory_qwen35_repo; then
  conda create -y -n llamafactory_qwen35_repo python=3.11
fi

conda activate llamafactory_qwen35_repo
export PYTHONNOUSERSITE=1

python -m pip install --upgrade pip packaging wheel setuptools editables "hatchling>=1.18.0"
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

cd /jizhicfs/pkuhetu/bcxu/LlamaFactory
python -m pip install --no-build-isolation -e .
python -m pip install --no-build-isolation -r requirements/metrics.txt

python -m pip install --no-deps flash-linear-attention==0.4.2 fla-core==0.4.2

python - <<'PY'
import torch
import triton
import transformers
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("triton", triton.__version__)
print("transformers", transformers.__version__)
print("fla imports ok")
PY

python -m pip check
