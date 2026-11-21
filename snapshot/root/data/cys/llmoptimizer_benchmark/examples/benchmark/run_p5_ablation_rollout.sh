#!/bin/bash
# P5 prompt-block ablations on eval8 (80 trials).
#
# ptllm:  P17-P20, P22-P24  (baseline: ptllm P5, already in eval8)
# hybrid: P21, P25 on ptllm_tpe P5/H2
#
# Usage:
#   DRY_RUN=1 bash examples/benchmark/run_p5_ablation_rollout.sh
#   STAGE=ptllm bash examples/benchmark/run_p5_ablation_rollout.sh
#   STAGE=hybrid bash examples/benchmark/run_p5_ablation_rollout.sh
#   STAGE=all bash examples/benchmark/run_p5_ablation_rollout.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"

PTLLM_VARIANTS=(P17 P18 P19 P20 P22 P23 P24 P26)
HYBRID_VARIANTS=(P21 P25)

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="benchmark_data/p5_ablation_dryrun"
  N_TRIALS=6
  TIME_LIMIT=300
  LLM_N_INITIAL_SAMPLES=1
  EVAL_DATASETS="yeast,boston"
  echo "[DRY_RUN] output=$OUTPUT_DIR n_trials=$N_TRIALS time_limit=${TIME_LIMIT}s llm_n_initial=$LLM_N_INITIAL_SAMPLES datasets=$EVAL_DATASETS"
else
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
  EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"
  echo "[PROD] time_limit=${TIME_LIMIT}s dataset_retries=$DATASET_RETRIES"
fi

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

BASE_RUNNER=( "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py
  --algorithm random_forest
  --n-trials "$N_TRIALS"
  --seed "$SEED"
  --skip-test-eval
  --llm-backends gpt4o-mini
  --qwen-backend auto
  --resume
  --output-dir "$OUTPUT_DIR"
  --all-datasets
  --include-datasets "$EVAL_DATASETS"
  --time-limit "$TIME_LIMIT"
  --dataset-retries "$DATASET_RETRIES"
)

if [[ "$DRY_RUN" == "1" ]]; then
  BASE_RUNNER+=( --llm-mock --llm-n-initial-samples "${LLM_N_INITIAL_SAMPLES:-1}" )
fi

run_ptllm_ablations() {
  for P in "${PTLLM_VARIANTS[@]}"; do
    echo "=== ptllm ablation: $P ==="
    "${BASE_RUNNER[@]}" --optimizer ptllm --prompt-variant "$P" --hybrid-variant H0
  done
}

run_hybrid_ablations() {
  for P in "${HYBRID_VARIANTS[@]}"; do
    echo "=== ptllm_tpe ablation: $P / H2 ==="
    "${BASE_RUNNER[@]}" --optimizer ptllm_tpe --prompt-variant "$P" --hybrid-variant H2
  done
}

case "$STAGE" in
  ptllm)  run_ptllm_ablations ;;
  hybrid) run_hybrid_ablations ;;
  all)
    run_ptllm_ablations
    run_hybrid_ablations
    ;;
  *)
    echo "Unknown STAGE=$STAGE (ptllm | hybrid | all)" >&2
    exit 1
    ;;
esac

if [[ "$DRY_RUN" == "1" ]]; then
  "$PYTHON" examples/benchmark/verify_p5_ablation_blocks.py
else
  bash examples/benchmark/plot_p5_ablations.sh || true
fi

echo ""
echo "[Done] P5 ablation rollout STAGE=$STAGE on: $EVAL_DATASETS"
echo "Output: $OUTPUT_DIR"
