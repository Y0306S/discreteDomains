#!/bin/bash
set -euo pipefail

ROOT=/root/LLMOptimizer
LOG_DIR="$ROOT/env_build_logs"
LOG="$LOG_DIR/build_envs.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

fix_requirements() {
  local src="$1"
  local dst="$2"
  local env_name="$3"
  sed -E \
    -e 's/torch==2\.0\.1\+cpu/torch==2.0.1/' \
    -e 's/torch==2\.5\.1\+cu124/torch==2.5.1/' \
    -e 's/torchvision==0\.20\.1\+cu124/torchvision==0.20.1/' \
    -e 's/torchaudio==2\.5\.1\+cu124/torchaudio==2.5.1/' \
    -e '/^llamafactory==/d' \
    -e '/^cuda-toolkit==/d' \
    -e '/^cuda-bindings==/d' \
    -e '/^cuda-pathfinder==/d' \
    "$src" > "$dst"
  if [[ "$env_name" == "openbox_eval" ]]; then
  sed -i 's/scipy==1\.15\.3/scipy==1.10.1/' "$dst"
  fi
}

create_from_requirements() {
  local name="$1"
  local python_ver="$2"
  local requirements="$3"

  echo ""
  echo "========== $name ($(date)) =========="
  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    case "$name" in
      openbox_eval)
        if conda run -n openbox_eval python -c "import openbox" 2>/dev/null; then
          echo "SKIP: $name already exists and openbox imports"
          return 0
        fi
        ;;
      llamafactory_qwen35)
        if conda run -n llamafactory_qwen35 python -c "import llamafactory" 2>/dev/null; then
          echo "SKIP: $name already exists and llamafactory imports"
          return 0
        fi
        ;;
      *)
        echo "SKIP: $name already exists"
        return 0
        ;;
    esac
  fi

  local fixed="/tmp/${name}_requirements.txt"
  fix_requirements "$requirements" "$fixed" "$name"

  conda create -y -n "$name" "python=${python_ver}"
  conda run -n "$name" python -m pip install --upgrade pip wheel setuptools

  if [[ "$name" == "openbox_eval" ]]; then
    conda run -n "$name" pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
    conda run -n "$name" pip install "numpy==1.24.4" "scipy==1.10.1" "pandas==2.0.3"
    conda run -n "$name" pip install "openbox==0.8.4"
    grep -Ev '^(torch|scipy|numpy|pandas|cvxpy|clarabel|osqp|scs)== ' "$fixed" > "/tmp/${name}_requirements_core.txt"
    conda run -n "$name" pip install -r "/tmp/${name}_requirements_core.txt" || {
      echo "WARN: some optional openbox_eval deps failed; keeping core stack"
    }
  elif [[ "$name" == "llamafactory_qwen35" ]]; then
    conda run -n "$name" pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124
    grep -Ev '^(torch|torchvision|torchaudio|nvidia-|cuda-)' "$fixed" > "/tmp/${name}_requirements_no_torch.txt"
    conda run -n "$name" pip install -r "/tmp/${name}_requirements_no_torch.txt" || {
      echo "WARN: some llamafactory deps failed; continuing with editable install"
    }
    conda run -n "$name" pip install -e "$ROOT/LlamaFactory"
  else
    conda run -n "$name" pip install -r "$fixed"
  fi

  echo "OK: created $name"
}

verify_env() {
  local name="$1"
  shift
  echo "Verifying $name..."
  conda run -n "$name" python -c "$*"
}

echo "=== LLMOptimizer env build started $(date) ==="

# Remove broken partial envs from earlier attempts
for broken in openbox_eval llamafactory_qwen35; do
  if conda env list | awk '{print $1}' | grep -qx "$broken"; then
    if ! conda run -n "$broken" python -c "import sys; sys.exit(0)" 2>/dev/null; then
      conda env remove -n "$broken" -y
      continue
    fi
    case "$broken" in
      openbox_eval)
        conda run -n openbox_eval python -c "import openbox" 2>/dev/null || conda env remove -n openbox_eval -y
        ;;
      llamafactory_qwen35)
        conda run -n llamafactory_qwen35 python -c "import llamafactory" 2>/dev/null || conda env remove -n llamafactory_qwen35 -y
        ;;
    esac
  fi
done

if ! conda env list | awk '{print $1}' | grep -qx taskset_eval; then
  echo "Creating taskset_eval from conda yml"
  grep -v '^prefix:' "$ROOT/taskset_eval_environment.yml" > /tmp/taskset_eval_environment.yml
  conda env create -f /tmp/taskset_eval_environment.yml
else
  echo "SKIP: taskset_eval already exists"
fi

create_from_requirements openbox_eval 3.10 "$ROOT/openbox_eval_requirements.txt"
create_from_requirements llamafactory_qwen35 3.11 "$ROOT/llamafactory_qwen35_requirements.txt"

verify_env taskset_eval "import tensorflow as tf; print('taskset_eval tensorflow', tf.__version__)"
verify_env openbox_eval "import openbox, sklearn, torch; print('openbox_eval OK', torch.__version__)"
verify_env llamafactory_qwen35 "import torch, llamafactory; print('llamafactory_qwen35 OK', torch.__version__)"

echo "=== LLMOptimizer env build finished $(date) ==="
conda env list
