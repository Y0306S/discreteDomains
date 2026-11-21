#!/bin/bash
# Master valid_best CSV: every variant under real_hpo_v2 eval8 (P0–P29, hybrids, baselines).
set -euo pipefail
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
OUT="${OUT_DIR:-benchmark_data/real_hpo_v2/analysis/eval8/all_valid_best}"
"$PYTHON" examples/benchmark/plot_all_valid_best.py --out-dir "$OUT"
echo "[Done] all valid_best: $OUT"
