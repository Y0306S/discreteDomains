#!/bin/bash
# Resume the 6 eval8 variants that timed out before reaching 80 trials on some datasets.
#
# Usage:
#   bash examples/benchmark/run_eval8_completion_retry.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

RUNNER=( "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py
  --algorithm random_forest
  --n-trials "$N_TRIALS"
  --seed "$SEED"
  --skip-test-eval
  --llm-backends gpt4o-mini
  --qwen-backend auto
  --resume
  --output-dir "$OUTPUT_DIR"
  --all-datasets
)

run_ptllm() {
  local prompt="$1" datasets="$2"
  echo ""
  echo "=== RETRY ptllm $prompt | datasets=$datasets ==="
  "${RUNNER[@]}" \
    --include-datasets "$datasets" \
    --optimizer ptllm \
    --prompt-variant "$prompt" \
    --hybrid-variant H0
}

run_ptllm_tpe() {
  local prompt="$1" hybrid="$2" datasets="$3"
  echo ""
  echo "=== RETRY ptllm_tpe $prompt/$hybrid | datasets=$datasets ==="
  "${RUNNER[@]}" \
    --include-datasets "$datasets" \
    --optimizer ptllm_tpe \
    --prompt-variant "$prompt" \
    --hybrid-variant "$hybrid"
}

# Only datasets still short of 80 trials (checked 2026-07-09)
run_ptllm  P4 pol
run_ptllm  P1 pol
run_ptllm_tpe P5 H7 pol
run_ptllm_tpe P7 H1 yeast
run_ptllm_tpe P2 H0 PhishingWebsites,pol
run_ptllm_tpe P1 H0 PhishingWebsites,segment,tecator,pol

echo ""
echo "[Done] eval8 completion retry"
echo "Regenerate: bash examples/benchmark/plot_all_valid_best.sh"
