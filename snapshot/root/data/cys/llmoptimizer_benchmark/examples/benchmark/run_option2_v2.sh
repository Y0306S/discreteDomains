#!/bin/bash
# Option 2 v2 benchmark driver: pilot-then-rollout for prompt/hybrid tuning.
#
# Pilot set (5): yeast, adult, covertype | boston, elevators
#
# Usage:
#   STAGE=baselines bash examples/benchmark/run_option2_v2.sh
#   STAGE=stage1_pilot bash examples/benchmark/run_option2_v2.sh
#   STAGE=stage1_rollout PROMPT=P5 bash examples/benchmark/run_option2_v2.sh
#   STAGE=stage2_pilot PROMPT=P5 bash examples/benchmark/run_option2_v2.sh
#   STAGE=stage2_pilot PROMPT=P6 INCLUDE_DATASETS=yeast,adult,boston,elevators bash examples/benchmark/run_option2_v2.sh
#   STAGE=stage2_rollout PROMPT=P5 HYBRID=H4 bash examples/benchmark/run_option2_v2.sh
#   bash examples/benchmark/monitor_v2_progress.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
STAGE="${STAGE:-baselines}"
PROMPT="${PROMPT:-P1}"
HYBRID="${HYBRID:-H0}"
DATASET_SCOPE="${DATASET_SCOPE:-all}"  # all | pilot | remaining
INCLUDE_DATASETS="${INCLUDE_DATASETS:-}"  # e.g. yeast,adult,boston,elevators
SMOKE="${SMOKE:-0}"

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

dataset_flags() {
  local flags=()
  case "$DATASET_SCOPE" in
    pilot)     flags=(--v2-datasets --pilot-only) ;;
    remaining) flags=(--v2-datasets --remaining-only) ;;
    all)       flags=(--v2-datasets --all-datasets) ;;
    *) echo "Unknown DATASET_SCOPE=$DATASET_SCOPE" >&2; exit 1 ;;
  esac
  if [[ -n "$INCLUDE_DATASETS" ]]; then
    flags+=(--include-datasets "$INCLUDE_DATASETS")
  fi
  echo "${flags[@]}"
}

RUNNER=( "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py
  --algorithm random_forest
  --n-trials "$N_TRIALS"
  --seed "$SEED"
  --skip-test-eval
  --llm-backends gpt4o-mini
  --qwen-backend auto
  --resume
  --output-dir "$OUTPUT_DIR"
)

if [[ "$SMOKE" == "1" ]]; then
  RUNNER+=( --dataset yeast --task-type cls --n-trials 3 --optimizer smac --llm-mock )
  echo "[SMOKE] yeast x smac x 3 trials"
  "${RUNNER[@]}"
  exit 0
fi

run_llm_prompt() {
  local prompt="$1"
  echo ""
  echo "=== ptllm (prompt sweep) | prompt=$prompt | scope=$DATASET_SCOPE ==="
  "${RUNNER[@]}" \
    $(dataset_flags) \
    --optimizer ptllm \
    --prompt-variant "$prompt" \
    --hybrid-variant H0
}

run_llm_pair() {
  local prompt="$1"
  local hybrid="${2:-H0}"
  echo ""
  echo "=== ptllm / ptllm_tpe | prompt=$prompt hybrid=$hybrid | scope=$DATASET_SCOPE ==="
  "${RUNNER[@]}" \
    $(dataset_flags) \
    --optimizer ptllm ptllm_tpe \
    --prompt-variant "$prompt" \
    --hybrid-variant "$hybrid"
}

case "$STAGE" in
  baselines)
    echo "[Stage] Baselines on all 14 datasets: smac tpe cmaes ptllm ptllm_tpe (P0/H0)"
    DATASET_SCOPE=all
    "${RUNNER[@]}" \
      $(dataset_flags) \
      --optimizer all \
      --prompt-variant P0 \
      --hybrid-variant H0
    ;;
  stage1_pilot)
    echo "[Stage] Prompt sweep on PILOT (5 datasets, ptllm only): P0 P1 P2 P4 P5 P6 P7 P8"
    DATASET_SCOPE=pilot
    for P in P0 P1 P2 P4 P5 P6 P7 P8; do
      run_llm_prompt "$P"
    done
    ;;
  stage1_rollout)
    echo "[Stage] Rollout best prompt=$PROMPT to REMAINING 9 datasets (ptllm only)"
    DATASET_SCOPE=remaining
    run_llm_prompt "$PROMPT"
    ;;
  stage1_extended)
    echo "[Stage] Extended prompt sweep (ptllm only, scope=$DATASET_SCOPE)"
    for P in P3 P9 P10 P11 P12 P13 P14 P15; do
      run_llm_prompt "$P"
    done
    ;;
  stage2_pilot)
    echo "[Stage] Hybrid sweep on PILOT with prompt=$PROMPT"
    DATASET_SCOPE=pilot
    for H in H0 H1 H2 H3 H4 H5 H6 H7; do
      echo ""
      echo "=== ptllm_tpe | prompt=$PROMPT hybrid=$H | scope=pilot ==="
      "${RUNNER[@]}" \
        $(dataset_flags) \
        --optimizer ptllm_tpe \
        --prompt-variant "$PROMPT" \
        --hybrid-variant "$H"
    done
    ;;
  stage2_rollout)
    echo "[Stage] Rollout best combo prompt=$PROMPT hybrid=$HYBRID to REMAINING 9 datasets"
    DATASET_SCOPE=remaining
    "${RUNNER[@]}" \
      $(dataset_flags) \
      --optimizer ptllm_tpe \
      --prompt-variant "$PROMPT" \
      --hybrid-variant "$HYBRID"
    ;;
  # Legacy aliases
  stage1_core)
    STAGE=stage1_pilot
    exec "$0"
    ;;
  stage2)
    STAGE=stage2_pilot
    exec "$0"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    echo "  baselines | stage1_pilot | stage1_rollout | stage2_pilot | stage2_rollout" >&2
    exit 1
    ;;
esac

echo ""
echo "[Done] STAGE=$STAGE scope=$DATASET_SCOPE output=$OUTPUT_DIR"
echo "Monitor: bash examples/benchmark/monitor_v2_progress.sh"
echo "Pick winner: python examples/benchmark/pick_pilot_winner.py --base-dir $OUTPUT_DIR --scope pilot"
