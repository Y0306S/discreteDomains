#!/bin/bash
# Dedicated venv for official automl/fanova (Hutter 2014).
# Isolated from the NumPy 2 benchmark env; the main process talks to it
# via mindware/components/optimizers/fanova_worker.py (JSON over stdin/stdout).
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
VENV="${FANOVA_VENV:-/root/venvs/fanova}"
PY311="${PYTHON311:-/root/miniconda3/envs/hpo_benchmark/bin/python}"

if ! command -v swig3.0 >/dev/null 2>&1 && ! command -v swig >/dev/null 2>&1; then
  echo "swig 3.0.12 is required to build pyrfr (SWIG 4.x produces a broken wrapper)."
  echo "Install with: apt-get install -y swig3.0"
  exit 1
fi

# pyrfr must be compiled with SWIG 3; prefer swig3.0 on PATH.
if command -v swig3.0 >/dev/null 2>&1; then
  SWIG3_BIN="$(mktemp -d)"
  ln -sfn "$(command -v swig3.0)" "$SWIG3_BIN/swig"
  export PATH="$SWIG3_BIN:$PATH"
fi

if [[ ! -x "$PY311" ]]; then
  echo "Python 3.11 not found at $PY311"
  exit 1
fi

if [[ "${RECREATE:-0}" == "1" && -d "$VENV" ]]; then
  rm -rf "$VENV"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[fanova] creating venv at $VENV"
  "$PY311" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install -U pip wheel setuptools
"$VENV/bin/python" -m pip install -r "$REPO/requirements-fanova-env.txt"

echo "[fanova] verifying import..."
"$VENV/bin/python" - <<'PY'
import numpy as np
if not hasattr(np, "float"):
    np.float = float
from fanova import fANOVA
print("ok fanova", np.__version__)
PY

echo "[fanova] export FANOVA_PYTHON=$VENV/bin/python"
echo "[fanova] done"
