#!/bin/bash
# Regenerate P5 ablation analysis on eval8.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE="${BASE_DIR:-benchmark_data/real_hpo_v2}"
OUT="$BASE/analysis/eval8/p5_ablations"
DS="${EVAL8_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"

"$PYTHON" examples/benchmark/plot_p5_ablations.py \
  --base-dir "$BASE" \
  --include-datasets "$DS" \
  --out-dir "$OUT"

echo "[Done] P5 ablation analysis: $OUT"
