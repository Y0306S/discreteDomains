#!/bin/bash
# Create the dedicated Phase 0 benchmark conda environment.
set -euo pipefail

ENV_NAME="${ENV_NAME:-hpo_benchmark}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
CONDA="/root/miniconda3/bin/conda"

if ! command -v "$CONDA" >/dev/null 2>&1; then
  echo "conda not found at $CONDA"
  exit 1
fi

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[env] $ENV_NAME already exists (set RECREATE=1 to rebuild)"
  if [[ "${RECREATE:-0}" == "1" ]]; then
    "$CONDA" env remove -n "$ENV_NAME" -y
    "$CONDA" create -n "$ENV_NAME" python=3.11 -y
  fi
else
  echo "[env] Creating $ENV_NAME (python=3.11)..."
  "$CONDA" create -n "$ENV_NAME" python=3.11 -y
fi

PYTHON="/root/miniconda3/envs/${ENV_NAME}/bin/python"
PIP="/root/miniconda3/envs/${ENV_NAME}/bin/pip"

echo "[env] Installing PyTorch (CUDA 12.4)..."
"$PIP" install --upgrade pip
"$PIP" install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "[env] Installing benchmark requirements..."
"$PIP" install -r "$REPO/requirements-benchmark-env.txt"

echo "[env] Verifying imports..."
"$PYTHON" - <<'PY'
import sys
sys.path.insert(0, "/root/data/cys/blockdata_archive/llmoptimizer_benchmark")
import torch
from transformers import AutoTokenizer
from mindware import CASH
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mindware OK")
PY

touch "$REPO/.env_setup_done"

echo "[env] Done. Activate with:"
echo "  source /root/miniconda3/bin/activate $ENV_NAME"
echo "Or use: /root/miniconda3/envs/$ENV_NAME/bin/python"
