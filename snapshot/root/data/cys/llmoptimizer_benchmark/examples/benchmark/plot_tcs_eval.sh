#!/bin/bash
# TCS eval analysis: baselines + ptllm P5 + ptllm_tpe P5/H2 + P16/TCS/H2 on eval8.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE="${BASE_DIR:-benchmark_data/real_hpo_v2}"
OUT="$BASE/analysis/eval8/tcs_eval"
DS="${EVAL8_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"

"$PYTHON" examples/benchmark/plot_tcs_eval.py \
  --base-dir "$BASE" \
  --include-datasets "$DS" \
  --out-dir "$OUT"

echo "[Done] TCS analysis: $OUT"
