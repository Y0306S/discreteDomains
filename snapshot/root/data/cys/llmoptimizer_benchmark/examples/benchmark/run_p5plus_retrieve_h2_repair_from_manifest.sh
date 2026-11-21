#!/usr/bin/env bash
# Run RC/RS repair jobs listed in manifest: prompt|task_type|dataset|algorithm per line.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

MANIFEST="${1:?manifest path required}"
PARALLEL="${PARALLEL:-4}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40}"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

run_job() {
  local prompt="$1" tt="$2" ds="$3" algo="$4"
  local tag="ptllm_tpe__${prompt}__H2"
  local logf="$LOGDIR/retrieve_h2_finish_${tag}_${ds}_${algo}_${TS}.log"
  echo "[$(date -Is)] START $tag $tt/$ds/$algo"
  "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials 80 --seed 1 --skip-test-eval \
    --llm-backends gpt4o-mini \
    --output-dir "$OUTPUT_ROOT" \
    --all-datasets --include-datasets "$ds" \
    --time-limit 7200 --per-run-time-limit 300 --dataset-retries 3 \
    --optimizer ptllm_tpe \
    --prompt-variant "$prompt" \
    --hybrid-variant H2 \
    --resume >>"$logf" 2>&1
  echo "[$(date -Is)] DONE $tag $tt/$ds/$algo"
}

active=0
while IFS='|' read -r prompt tt ds algo; do
  [[ -n "$prompt" ]] || continue
  while (( active >= PARALLEL )); do
    wait -n 2>/dev/null || wait
    active=$((active - 1))
  done
  run_job "$prompt" "$tt" "$ds" "$algo" &
  active=$((active + 1))
done < "$MANIFEST"
wait
