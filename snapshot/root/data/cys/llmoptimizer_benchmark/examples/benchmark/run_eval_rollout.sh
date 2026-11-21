#!/bin/bash
# Roll out Stage-1 / Stage-2 winners onto the 8-dataset eval set.
#
# Eval set (4 cls + 4 rgs):
#   yeast, car, PhishingWebsites, segment | boston, tecator, socmob, pol
#
# Winners (from pilot sweeps, no adult/covertype/elevators):
#   Stage 1 pure ptllm — top 3 prompts: P6, P5, P7
#   Stage 2 ptllm_tpe @ P5 — top 3 hybrids: H0, H2, H6
#   Stage 2 ptllm_tpe @ P6 — top 3 hybrids: H2, H5, H7
#
# Usage:
#   STAGE=baselines bash examples/benchmark/run_eval_rollout.sh
#   STAGE=stage1_winners bash examples/benchmark/run_eval_rollout.sh
#   STAGE=stage2_p5 bash examples/benchmark/run_eval_rollout.sh
#   STAGE=stage2_p6 bash examples/benchmark/run_eval_rollout.sh
#   STAGE=all bash examples/benchmark/run_eval_rollout.sh
#   DRY_RUN=1 bash examples/benchmark/run_eval_rollout.sh   # 2 trials, mock LLM, isolated dir
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="benchmark_data/eval_rollout_dryrun"
  N_TRIALS=2
  EVAL_DATASETS="yeast,segment,boston"
  echo "[DRY_RUN] output=$OUTPUT_DIR n_trials=$N_TRIALS datasets=$EVAL_DATASETS (mock LLM)"
else
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
  EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"
fi

# Pilot winners — edit only after re-running pick_pilot_winner on eval set
STAGE1_PROMPTS=(P6 P5 P7)
STAGE2_P5_HYBRIDS=(H0 H2 H6)
STAGE2_P6_HYBRIDS=(H2 H5 H7)

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
  --include-datasets "$EVAL_DATASETS"
)
if [[ "$DRY_RUN" == "1" ]]; then
  RUNNER+=( --llm-mock )
fi

run_baselines() {
  echo "=== Baselines (smac tpe cmaes ptllm P0 ptllm_tpe P0/H0) ==="
  "${RUNNER[@]}" \
    --optimizer all \
    --prompt-variant P0 \
    --hybrid-variant H0
}

run_stage1() {
  for P in "${STAGE1_PROMPTS[@]}"; do
    echo "=== Stage 1 winner: ptllm $P ==="
    "${RUNNER[@]}" \
      --optimizer ptllm \
      --prompt-variant "$P" \
      --hybrid-variant H0
  done
}

run_stage2_p5() {
  for H in "${STAGE2_P5_HYBRIDS[@]}"; do
    echo "=== Stage 2 P5 winner: ptllm_tpe P5/$H ==="
    "${RUNNER[@]}" \
      --optimizer ptllm_tpe \
      --prompt-variant P5 \
      --hybrid-variant "$H"
  done
}

run_stage2_p6() {
  for H in "${STAGE2_P6_HYBRIDS[@]}"; do
    echo "=== Stage 2 P6 winner: ptllm_tpe P6/$H ==="
    "${RUNNER[@]}" \
      --optimizer ptllm_tpe \
      --prompt-variant P6 \
      --hybrid-variant "$H"
  done
}

case "$STAGE" in
  baselines)   run_baselines ;;
  stage1_winners) run_stage1 ;;
  stage2_p5)   run_stage2_p5 ;;
  stage2_p6)   run_stage2_p6 ;;
  all)
    run_baselines
    run_stage1
    run_stage2_p5
    run_stage2_p6
    ;;
  *)
    echo "Unknown STAGE=$STAGE (baselines | stage1_winners | stage2_p5 | stage2_p6 | all)" >&2
    exit 1
    ;;
esac

echo ""
echo "[Done] STAGE=$STAGE eval rollout on: $EVAL_DATASETS"
echo "Monitor: bash examples/benchmark/monitor_v2_progress.sh"
