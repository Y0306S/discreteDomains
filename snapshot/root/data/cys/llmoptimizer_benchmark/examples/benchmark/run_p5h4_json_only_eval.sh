#!/bin/bash
# P5/H4 json_only ablation — compare against existing ptllm_tpe__P5__H4 (rationale_json).
#
# Prereq: tests/test_parse_llm_json.py + verify_p5h4_parse_replay.py pass.
#
# Smoke (mock, no API):
#   STAGE=smoke DRY_RUN=1 bash examples/benchmark/run_p5h4_json_only_eval.sh
#
# Smoke (real API, 2 datasets):
#   STAGE=smoke bash examples/benchmark/run_p5h4_json_only_eval.sh
#
# Full eval8:
#   STAGE=full bash examples/benchmark/run_p5h4_json_only_eval.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

STAGE="${STAGE:-smoke}"
DRY_RUN="${DRY_RUN:-0}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
PROMPT="${PROMPT:-P5}"
HYBRID="${HYBRID:-H4}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"

OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2_p5h4_json_only}"
VARIANT_SUFFIX="${VARIANT_SUFFIX:-json_only}"

case "$STAGE" in
  smoke)
    EVAL_DATASETS="${EVAL_DATASETS:-yeast,boston}"
    N_TRIALS="${N_TRIALS:-20}"
    TIME_LIMIT="${TIME_LIMIT:-1800}"
  ;;
  full)
    EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"
    ;;
  *)
    echo "Unknown STAGE=$STAGE (smoke | full)" >&2
    exit 1
    ;;
esac

if [[ "$DRY_RUN" == "1" ]]; then
  N_TRIALS=4
  TIME_LIMIT=300
  echo "[DRY_RUN] mock LLM | datasets=$EVAL_DATASETS"
fi

echo "[Step 0] Parser unit tests"
"$PYTHON" tests/test_parse_llm_json.py -v

echo "[Step 0b] Replay-parse existing ptllm_tpe__P5__H4 logs"
"$PYTHON" examples/benchmark/verify_p5h4_parse_replay.py

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
  --time-limit "$TIME_LIMIT"
  --dataset-retries "$DATASET_RETRIES"
  --prompt-variant "$PROMPT"
  --optimizer ptllm_tpe
  --hybrid-variant "$HYBRID"
  --llm-response-format json_only
  --variant-suffix "$VARIANT_SUFFIX"
)

if [[ "$DRY_RUN" == "1" ]]; then
  RUNNER+=( --llm-mock --llm-n-initial-samples 1 )
fi

echo ""
echo "=== ptllm_tpe | prompt=$PROMPT hybrid=$HYBRID | json_only | STAGE=$STAGE ==="
echo "    output=$OUTPUT_DIR"
echo "    variant=ptllm_tpe__${PROMPT}__${HYBRID}${VARIANT_SUFFIX}"
"${RUNNER[@]}"

echo ""
echo "[Done] STAGE=$STAGE datasets=$EVAL_DATASETS output=$OUTPUT_DIR"
echo "After full run, regenerate leaderboard:"
echo "  python examples/benchmark/plot_all_valid_best.py --base-dir $OUTPUT_DIR"
