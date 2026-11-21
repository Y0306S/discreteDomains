#!/bin/bash
# Generate tables + plots for the 8-dataset eval rollout.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE="benchmark_data/real_hpo_v2"
OUT="$BASE/analysis/eval8"
DS="${EVAL8_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"

run() { echo ""; echo "=== $* ==="; "$@"; }

run "$PYTHON" examples/benchmark/plot_baseline_vs_tuned.py \
  --base-dir "$BASE" \
  --include-datasets "$DS" \
  --ptllm-prompt P5 --tpe-prompt P5 --top-tpe-hybrids 3 \
  --out-dir "$OUT/baseline_vs_tuned_P5"

run "$PYTHON" examples/benchmark/plot_baseline_vs_tuned.py \
  --base-dir "$BASE" \
  --include-datasets "$DS" \
  --ptllm-prompt P6 --tpe-prompt P6 --tpe-hybrids H2,H5,H7 \
  --out-dir "$OUT/baseline_vs_tuned_P6"

run "$PYTHON" examples/benchmark/plot_pilot_sweep.py \
  --base-dir "$BASE" \
  --include-datasets "$DS" \
  --mode prompt \
  --out-dir "$OUT/stage1_prompts"

run "$PYTHON" examples/benchmark/plot_pilot_sweep.py \
  --base-dir "$BASE" \
  --include-datasets "$DS" \
  --mode hybrid --prompt P5 \
  --out-dir "$OUT/stage2_hybrids_P5"

run "$PYTHON" examples/benchmark/plot_pilot_sweep.py \
  --base-dir "$BASE" \
  --include-datasets "$DS" \
  --mode hybrid --prompt P6 \
  --out-dir "$OUT/stage2_hybrids_P6"

echo ""
echo "[Done] Eval8 analysis under $OUT"
