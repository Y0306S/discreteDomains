#!/bin/bash
# Resume all dl_test_15 tier-5 CPU evals sequentially.
#
# Usage:
#   TASK_WORKERS=2 bash scripts/eval/resume_all_tier5.sh
#   nohup bash scripts/eval/resume_all_tier5.sh &

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

LOG_DIR="eval_runs/dl_test_15/tier5_cpu_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/master_resume_all_$(date +%Y%m%d_%H%M%S).log"
exec >>"$MASTER_LOG" 2>&1

TASK_WORKERS="${TASK_WORKERS:-2}"
export TASK_WORKERS

wait_model() {
  local model="$1"
  echo "[$(date -Iseconds)] waiting for $model eval to finish ..."
  while pgrep -f "run_taskset_dl_eval_optimized.py.*--model-label ${model}" >/dev/null; do
    sleep 60
  done
  echo "[$(date -Iseconds)] $model eval finished"
}

checkpoint_summary() {
  local model="$1"
  /root/miniconda3/envs/taskset_eval/bin/python3 - <<PY
import json
from pathlib import Path
from collections import Counter
model = "$model"
total = sum(1 for _ in open("data/dl_test_15_modified.jsonl"))
cp = Path(f"eval_runs/dl_test_15/tier5_cpu/{model}/eval_checkpoint.json")
if not cp.exists():
    print(f"  {model}: no checkpoint")
else:
    ckpt = json.loads(cp.read_text())
    results = ckpt.get("results", [])
    statuses = Counter(r.get("status") for r in results)
    done = len({int(r["index"]) for r in results if r.get("index") is not None})
    print(f"  {model}: {done}/{total}  {dict(statuses)}")
PY
}

echo "=== master resume all tier5 ==="
echo "  task_workers=$TASK_WORKERS"
echo "  log=$MASTER_LOG"
echo

echo "=== Phase 1: Model 3 — retry 6 amazon failures ==="
checkpoint_summary qwen_base_lora_sft_3_dl
MODEL=qwen_base_lora_sft_3_dl bash scripts/eval/retry_failed_tier5_eval.sh
wait_model qwen_base_lora_sft_3_dl
checkpoint_summary qwen_base_lora_sft_3_dl
echo

echo "=== Phase 2: Model 4 — resume failed + missing ==="
checkpoint_summary qwen_base_lora_sft_4_dl
MODEL=qwen_base_lora_sft_4_dl bash scripts/eval/retry_failed_tier5_eval.sh
wait_model qwen_base_lora_sft_4_dl
checkpoint_summary qwen_base_lora_sft_4_dl
echo

echo "=== Phase 3: Models 5, 7, gemini — fresh/resume ==="
for m in qwen_base_lora_sft_5_dl qwen_base_lora_sft_7_dl gemini_3.1_pro_preview; do
  checkpoint_summary "$m"
done
SKIP_DONE=1 MODELS="qwen_base_lora_sft_5_dl,qwen_base_lora_sft_7_dl,gemini_3.1_pro_preview" \
  bash scripts/eval/run_dl_test_15_tier5_cpu.sh
echo

echo "[$(date -Iseconds)] All tier5 models finished."
for m in qwen_base_lora_sft_3_dl qwen_base_lora_sft_4_dl qwen_base_lora_sft_5_dl qwen_base_lora_sft_7_dl gemini_3.1_pro_preview; do
  checkpoint_summary "$m"
done
