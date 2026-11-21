#!/usr/bin/env bash
# Build LLMOptimizer conda environments (openbox_eval + llamafactory_qwen35).
# Usage:
#   nohup /root/LLMOptimizer/build_all_envs.sh >> /root/LLMOptimizer/env_build_logs/master.log 2>&1 &
# Monitor:
#   tail -f /root/LLMOptimizer/env_build_logs/master.log
#   tail -f /root/LLMOptimizer/env_build_logs/<env_name>.log

set -uo pipefail

ROOT="/root/LLMOptimizer"
LOGDIR="$ROOT/env_build_logs"
CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
STRIP_DIR="/tmp/llmoptimizer_env_yml"
OFFICIAL_CONDARC="/tmp/llmoptimizer_official_condarc.yml"

mkdir -p "$LOGDIR" "$STRIP_DIR"
source "$CONDA_SH"

# Use official Anaconda channels to avoid corrupted Tsinghua mirror repodata.
cat > "$OFFICIAL_CONDARC" <<'EOF'
channels:
  - defaults
  - conda-forge
show_channel_urls: true
default_channels:
  - https://repo.anaconda.com/pkgs/main
  - https://repo.anaconda.com/pkgs/r
custom_channels:
  conda-forge: https://conda.anaconda.org
EOF

export CONDARC="$OFFICIAL_CONDARC"

strip_yaml() {
  local src="$1"
  local dst="$STRIP_DIR/$(basename "$src" .yml)_stripped.yml"
  grep -v '^prefix:' "$src" > "$dst"
  echo "$dst"
}

create_env() {
  local yml="$1"
  local req="$2"
  local name
  name="$(grep '^name:' "$yml" | awk '{print $2}')"
  local log="$LOGDIR/${name}.log"

  if conda env list | awk '{print $1}' | grep -qx "$name"; then
    echo "[$(date)] SKIP: $name already exists" | tee -a "$log"
  else
    echo "[$(date)] CREATE: $name from $yml" | tee "$log"
    local stripped
    stripped="$(strip_yaml "$yml")"
    if ! conda env create -f "$stripped" 2>&1 | tee -a "$log"; then
      echo "[$(date)] FAILED conda create: $name" | tee -a "$log"
      return 1
    fi
  fi

  echo "[$(date)] PIP: installing requirements for $name" | tee -a "$log"
  # shellcheck disable=SC1091
  conda activate "$name"

  local req_filtered="/tmp/${name}_requirements_filtered.txt"
  grep -v -E '^(llamafactory|hpobench)==' "$req" > "$req_filtered" || true

  if [[ -s "$req_filtered" ]]; then
    pip install -r "$req_filtered" 2>&1 | tee -a "$log" || echo "[$(date)] WARN: some pip packages failed for $name" | tee -a "$log"
  fi

  if grep -q '^llamafactory==' "$req"; then
    echo "[$(date)] PIP: editable llamafactory for $name" | tee -a "$log"
    pip install -e "$ROOT/LlamaFactory" 2>&1 | tee -a "$log"
  fi

  if grep -q '^hpobench==' "$req"; then
    echo "[$(date)] PIP: installing hpobench for $name" | tee -a "$log"
    if pip install "hpobench==0.0.10" 2>&1 | tee -a "$log"; then
      :
    elif [[ -d "$ROOT/HPOBench" ]]; then
      echo "[$(date)] PIP: editable hpobench from $ROOT/HPOBench" | tee -a "$log"
      pip install -e "$ROOT/HPOBench" 2>&1 | tee -a "$log" || true
      pip install -e "$ROOT/HPOBench/.[xgboost,rf,lr,nn]" 2>&1 | tee -a "$log" || true
    else
      echo "[$(date)] WARN: hpobench install failed; clone HPOBench to $ROOT/HPOBench and re-run pip install -e" | tee -a "$log"
    fi
  fi

  conda deactivate
  echo "[$(date)] DONE: $name" | tee -a "$log"
}

cd "$ROOT"

ENVS=(
  "openbox_eval_environment.yml:openbox_eval_requirements.txt"
  "llamafactory_qwen35_environment.yml:llamafactory_qwen35_requirements.txt"
)

FAILED=()
for pair in "${ENVS[@]}"; do
  yml="${pair%%:*}"
  req="${pair##*:}"
  if ! create_env "$ROOT/$yml" "$ROOT/$req"; then
    FAILED+=("$(grep '^name:' "$ROOT/$yml" | awk '{print $2}')")
  fi
done

echo "[$(date)] BUILD SUMMARY" | tee "$LOGDIR/summary.log"
conda env list | tee -a "$LOGDIR/summary.log"
if ((${#FAILED[@]})); then
  echo "[$(date)] FAILED envs: ${FAILED[*]}" | tee -a "$LOGDIR/summary.log"
  exit 1
fi
echo "[$(date)] ALL ENV BUILDS FINISHED" | tee -a "$LOGDIR/summary.log"
