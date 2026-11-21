#!/bin/bash
set -euo pipefail

BASE="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune"
cd "$BASE"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate taskset_eval

OUT="eval_runs/dl_test_15/dataset_verify"
mkdir -p "$OUT"

echo "[1/2] Building one prediction per READY dataset (shared nadamw_grid config)..."
python scripts/eval/build_dataset_dryrun_predictions.py

echo ""
echo "[2/2] Dry-run via run_taskset_dl_eval_optimized.py..."
python scripts/eval/run_taskset_dl_eval_optimized.py \
  --test-data data/dl_test_15_modified.jsonl \
  --predictions "$OUT/dataset_verify_predictions.jsonl" \
  --baseline-cache "$OUT/task_baselines.json" \
  --dry-run \
  --output-json "$OUT/dataset_dryrun_report.json" \
  2>&1 | tee "$OUT/dataset_dryrun.log"

echo ""
echo "Report: $OUT/dataset_dryrun_report.json"
