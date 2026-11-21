#!/bin/bash
# Build all evaluation inputs from local data (GCS NPZ cache + dl_test_15 JSONL).
set -euo pipefail

BASE="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune"
cd "$BASE"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate taskset_eval

OUT="eval_runs/dl_test_15"
mkdir -p "$OUT"

echo "=== [1/4] Baseline cache from GCS NPZ (best_val per task) ==="
python scripts/eval/build_baselines_from_gcs_cache.py \
  --dl-test-only \
  --output "$OUT/task_baselines_from_gcs.json"

# Canonical symlink-style copy for eval scripts
cp -f "$OUT/task_baselines_from_gcs.json" "$OUT/task_baselines.json"

echo ""
echo "=== [2/4] Reference predictions (one nadamw_grid config per resolvable task) ==="
python scripts/eval/build_dl_test_reference_predictions.py

echo ""
echo "=== [3/4] Full dry-run (all resolvable tasks, real baselines) ==="
python scripts/eval/run_taskset_dl_eval_optimized.py \
  --test-data data/dl_test_15_modified.jsonl \
  --predictions "$OUT/dl_test_15_reference_predictions.jsonl" \
  --baseline-cache "$OUT/task_baselines.json" \
  --dry-run \
  --output-json "$OUT/dl_test_15_dryrun_report.json" \
  2>&1 | tee "$OUT/dl_test_15_dryrun.log"

echo ""
echo "=== [4/4] Dataset verify dry-run (22 READY TFDS datasets) ==="
python scripts/eval/build_dataset_dryrun_predictions.py
python scripts/eval/run_taskset_dl_eval_optimized.py \
  --test-data data/dl_test_15_modified.jsonl \
  --predictions "$OUT/dataset_verify/dataset_verify_predictions.jsonl" \
  --baseline-cache "$OUT/task_baselines.json" \
  --dry-run \
  --output-json "$OUT/dataset_verify/dataset_dryrun_report.json"

echo ""
echo "=== Done ==="
echo "  Baselines:  $OUT/task_baselines.json"
echo "  Predictions: $OUT/dl_test_15_reference_predictions.jsonl"
echo "  Dry-run:    $OUT/dl_test_15_dryrun_report.json"
echo "  Manifest:   $OUT/eval_package_manifest.json"
