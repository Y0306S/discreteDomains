#!/bin/bash
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

ENV_NAME="${ENV_NAME:-hpo_benchmark}"
HPO_PYTHON="/root/miniconda3/envs/${ENV_NAME}/bin/python"
OPENBOX_PYTHON="/root/miniconda3/envs/openbox_eval/bin/python"
PYTHON="${PYTHON:-}"

if [[ -z "$PYTHON" ]]; then
  if [[ -x "$HPO_PYTHON" ]] && "$HPO_PYTHON" -c "from mindware import CASH" 2>/dev/null; then
    PYTHON="$HPO_PYTHON"
  else
    PYTHON="$OPENBOX_PYTHON"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_cls}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
LLM_BACKENDS="${LLM_BACKENDS:-gpt4o-mini}"
QWEN_BACKEND="${QWEN_BACKEND:-auto}"

# Build dedicated env in background (for future Qwen HF runs); do not block benchmark.
if [[ "${SETUP_ENV_BG:-1}" == "1" ]] && [[ ! -f "$REPO/.env_setup_done" ]]; then
  if ! "$HPO_PYTHON" -c "from mindware import CASH" 2>/dev/null; then
    echo "[Batch] Starting hpo_benchmark env setup in background..."
    nohup bash "$REPO/scripts/setup_benchmark_env.sh" > "$REPO/benchmark_data/env_setup.log" 2>&1 &
  fi
fi

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

echo "[Batch] Phase 0 cls benchmark"
echo "  python=$PYTHON"
echo "  output_dir=$OUTPUT_DIR"
echo "  n_trials=$N_TRIALS seed=$SEED"
echo "  llm_backends=$LLM_BACKENDS"
echo "  qwen_backend=$QWEN_BACKEND (hf = HuggingFace Transformers in-process)"

# Smoke test on ada unless SKIP_SMOKE=1
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo "[Batch] Smoke test: ada (5 optimizers, gpt-4o-mini LLM)"
  "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
    --dataset ada \
    --optimizer all \
    --algorithm random_forest \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --metric bal_acc \
    --skip-test-eval \
    --llm-backends "$LLM_BACKENDS" \
    --qwen-backend "$QWEN_BACKEND" \
    --output-dir "$OUTPUT_DIR"
fi

echo "[Batch] Full sweep: all 71 classification datasets"
"$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
  --all-datasets \
  --optimizer all \
  --algorithm random_forest \
  --n-trials "$N_TRIALS" \
  --seed "$SEED" \
  --metric bal_acc \
  --skip-test-eval \
  --llm-backends "$LLM_BACKENDS" \
  --qwen-backend "$QWEN_BACKEND" \
  --output-dir "$OUTPUT_DIR"

echo "[Batch] Aggregating results + handoff package"
"$PYTHON" examples/benchmark/plot_real_benchmark.py \
  --base-dir "$OUTPUT_DIR" \
  --handoff

echo "[Batch] Done. Handoff: $OUTPUT_DIR/handoff/"
