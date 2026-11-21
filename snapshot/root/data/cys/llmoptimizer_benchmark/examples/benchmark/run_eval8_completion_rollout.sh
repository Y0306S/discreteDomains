#!/bin/bash
# Complete eval8 coverage for leaderboard variants stuck at 2/8 (pilot-only) or 1/8.
#
# Missing rollout datasets (6): car, PhishingWebsites, segment, tecator, socmob, pol
# ptllm_tpe P7/H1 also missing boston.
#
# Usage:
#   bash examples/benchmark/run_eval8_completion_rollout.sh
#   STAGE=ptllm_tpe_p5 bash examples/benchmark/run_eval8_completion_rollout.sh
#   DRY_RUN=1 bash examples/benchmark/run_eval8_completion_rollout.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

ROLLOUT_DS="car,PhishingWebsites,segment,tecator,socmob,pol"
P7_H1_DS="car,PhishingWebsites,segment,boston,tecator,socmob,pol"

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="benchmark_data/eval8_completion_dryrun"
  N_TRIALS=2
  ROLLOUT_DS="car,boston"
  P7_H1_DS="boston"
  echo "[DRY_RUN] n_trials=$N_TRIALS output=$OUTPUT_DIR"
fi

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

if [[ "$DRY_RUN" == "1" ]]; then
  RUNNER+=( --llm-mock )
fi

run_ptllm() {
  local prompt="$1"
  local datasets="$2"
  echo ""
  echo "=== ptllm $prompt | missing: $datasets ==="
  "${RUNNER[@]}" \
    --include-datasets "$datasets" \
    --optimizer ptllm \
    --prompt-variant "$prompt" \
    --hybrid-variant H0
}

run_ptllm_tpe() {
  local prompt="$1"
  local hybrid="$2"
  local datasets="$3"
  echo ""
  echo "=== ptllm_tpe $prompt/$hybrid | missing: $datasets ==="
  "${RUNNER[@]}" \
    --include-datasets "$datasets" \
    --optimizer ptllm_tpe \
    --prompt-variant "$prompt" \
    --hybrid-variant "$hybrid"
}

run_ptllm_tpe_rollout() {
  local prompt="$1"
  local hybrid="$2"
  run_ptllm_tpe "$prompt" "$hybrid" "$ROLLOUT_DS"
}

run_pure_ptllm() {
  for P in P1 P2 P4 P8; do
    run_ptllm "$P" "$ROLLOUT_DS"
  done
}

run_ptllm_tpe_p1_p8() {
  run_ptllm_tpe_rollout P1 H0
  run_ptllm_tpe_rollout P2 H0
  run_ptllm_tpe_rollout P4 H0
  run_ptllm_tpe_rollout P8 H0
}

run_ptllm_tpe_p5() {
  for H in H1 H3 H4 H5 H7; do
    run_ptllm_tpe_rollout P5 "$H"
  done
}

run_ptllm_tpe_p6() {
  for H in H0 H1 H3 H4 H6; do
    run_ptllm_tpe_rollout P6 "$H"
  done
}

run_ptllm_tpe_p7() {
  run_ptllm_tpe_rollout P7 H0
  run_ptllm_tpe P7 H1 "$P7_H1_DS"
}

case "$STAGE" in
  pure_ptllm)       run_pure_ptllm ;;
  ptllm_tpe_p1_p8)  run_ptllm_tpe_p1_p8 ;;
  ptllm_tpe_p5)     run_ptllm_tpe_p5 ;;
  ptllm_tpe_p6)     run_ptllm_tpe_p6 ;;
  ptllm_tpe_p7)     run_ptllm_tpe_p7 ;;
  all)
    run_pure_ptllm
    run_ptllm_tpe_p1_p8
    run_ptllm_tpe_p5
    run_ptllm_tpe_p6
    run_ptllm_tpe_p7
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    echo "  pure_ptllm | ptllm_tpe_p1_p8 | ptllm_tpe_p5 | ptllm_tpe_p6 | ptllm_tpe_p7 | all" >&2
    exit 1
    ;;
esac

echo ""
echo "[Done] eval8 completion rollout STAGE=$STAGE"
echo "Regenerate: bash examples/benchmark/plot_all_valid_best.sh"
