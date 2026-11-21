#!/bin/bash
# Remove failed task results and re-queue tier5 eval for one model.
#
# Usage:
#   MODEL=qwen_base_lora_sft_3_dl bash scripts/eval/retry_failed_tier5_eval.sh
#   MODEL=qwen_base_lora_sft_4_dl DRY_RUN=1 bash scripts/eval/retry_failed_tier5_eval.sh

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

MODEL="${MODEL:-qwen_base_lora_sft_3_dl}"
EVAL_DIR="eval_runs/dl_test_15"
WORK_DIR="${EVAL_DIR}/tier5_cpu/${MODEL}"
CHECKPOINT="${WORK_DIR}/eval_checkpoint.json"
EVAL_JSON="${EVAL_DIR}/${MODEL}_tier5_cpu_eval.json"
PYTHON="${TASKSET_PYTHON:-/root/miniconda3/envs/taskset_eval/bin/python}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 1
fi

echo "=== Retry failed tasks for $MODEL ==="

mapfile -t FAILED_INDICES < <(
  "$PYTHON" - "$MODEL" <<'PY'
import json, sys
from pathlib import Path
base = Path("eval_runs/dl_test_15")
model = sys.argv[1]
ckpt = json.load(open(base / "tier5_cpu" / model / "eval_checkpoint.json"))
results = ckpt.get("results", ckpt if isinstance(ckpt, list) else [])

# Load total task count from test data.
total = sum(1 for _ in open(Path("data/dl_test_15_modified.jsonl")))
all_indices = set(range(total))
done_indices = {int(r["index"]) for r in results if r.get("index") is not None}
failed_status = {
    int(r["index"])
    for r in results
    if r.get("status") in ("all_failed", "task_not_found")
}
missing = all_indices - done_indices
to_retry = sorted(failed_status | missing)
print("\n".join(str(i) for i in to_retry))
PY
)

if [[ ${#FAILED_INDICES[@]} -eq 0 ]]; then
  echo "No failed tasks in checkpoint."
  exit 0
fi

echo "Failed task indices (${#FAILED_INDICES[@]}): ${FAILED_INDICES[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 — would clean and retry these tasks only."
  exit 0
fi

# Stop any in-flight eval for this model only.
pkill -f "run_taskset_dl_eval_optimized.py.*--model-label ${MODEL}" 2>/dev/null || true
sleep 2

"$PYTHON" - <<'PY' "$MODEL" "${FAILED_INDICES[@]}"
import json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts/eval").resolve()))
from run_taskset_dl_eval_optimized import find_task_seed

model = sys.argv[1]
failed = [int(x) for x in sys.argv[2:]]
base = Path("eval_runs/dl_test_15")
work = base / "tier5_cpu" / model
ckpt_path = work / "eval_checkpoint.json"
ckpt = json.load(open(ckpt_path))
results = ckpt["results"] if isinstance(ckpt, dict) and "results" in ckpt else ckpt

failed_set = set(failed)
kept = [r for r in results if int(r.get("index", -1)) not in failed_set]
removed = [r for r in results if int(r.get("index", -1)) in failed_set]

# Resolve task names for all indices being retried.
gold_lines = Path("data/dl_test_15_modified.jsonl").read_text().splitlines()
task_names = set()
for row in removed:
    if row.get("task_name"):
        task_names.add(row["task_name"])
for idx in failed:
    if idx >= len(gold_lines):
        continue
    gold = json.loads(gold_lines[idx])
    user_msg = json.loads(gold["messages"][1]["content"])
    task_name = find_task_seed(user_msg["task_family"], user_msg["task"])
    if task_name:
        task_names.add(task_name)

train_root = work / "train_outputs"
for task_name in sorted(task_names):
    task_dir = train_root / task_name
    if task_dir.exists():
        shutil.rmtree(task_dir)
        print(f"removed train_outputs/{task_name}")

if isinstance(ckpt, dict) and "results" in ckpt:
    ckpt["results"] = kept
    payload = ckpt
else:
    payload = kept
json.dump(payload, open(ckpt_path, "w"), indent=2)
print(f"checkpoint: kept {len(kept)} ok/pending, removed {len(removed)} failed")

# Reset progress.json
prog_path = work / "progress.json"
if prog_path.exists():
    prog = json.load(open(prog_path))
    prog["done"] = len(kept)
    prog["completed_tasks"] = [
        {"index": int(r["index"]), "status": r.get("status", "ok")}
        for r in kept
        if r.get("index") is not None
    ]
    prog.pop("active_run", None)
    json.dump(prog, open(prog_path, "w"), indent=2)
PY

declare -A MODEL_PREDICTIONS=(
  ["qwen_base_lora_sft_3_dl"]="qwen_base_lora_sft_3_dl_test_predictions.jsonl"
  ["qwen_base_lora_sft_4_dl"]="qwen_base_lora_sft_4_dl_test_predictions.jsonl"
  ["qwen_base_lora_sft_5_dl"]="qwen_base_lora_sft_5_dl_test_predictions.jsonl"
  ["qwen_base_lora_sft_7_dl"]="qwen_base_lora_sft_7_dl_test_predictions.jsonl"
  ["gemini_3.1_pro_preview"]="gemini_3.1_pro_preview_test_predictions_run1.jsonl"
)

PRED="${MODEL_PREDICTIONS[$MODEL]}"
LOG_DIR="${EVAL_DIR}/tier5_cpu_logs"
mkdir -p "$LOG_DIR"

TASK_ARGS=(--task-indices "${FAILED_INDICES[@]}")

TASK_WORKERS="${TASK_WORKERS:-2}"

echo "Re-running ${#FAILED_INDICES[@]} failed tasks for $MODEL (task-workers=${TASK_WORKERS}) ..."
nohup "$PYTHON" scripts/eval/run_taskset_dl_eval_optimized.py \
  --test-data data/dl_test_15_modified.jsonl \
  --predictions "${EVAL_DIR}/${PRED}" \
  --model-label "$MODEL" \
  --baseline-cache eval_runs/dl_test_15/task_baselines.json \
  --skip-baseline-build \
  --num-replicas 3 \
  --cpu-profile \
  --task-workers "$TASK_WORKERS" \
  --work-dir "$WORK_DIR" \
  --output-json "$EVAL_JSON" \
  "${TASK_ARGS[@]}" \
  >> "${LOG_DIR}/${MODEL}_retry.log" 2>&1 &

echo "PID=$!"
echo "Log: ${LOG_DIR}/${MODEL}_retry.log"
