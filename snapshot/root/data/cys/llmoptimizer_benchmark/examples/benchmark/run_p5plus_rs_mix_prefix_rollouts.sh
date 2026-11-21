#!/usr/bin/env bash
# Mix-schedule Setup B rollouts (does not run unless you invoke this script).
#
# Prefixes:
#   0, 5, 10     → rs_warm  (P5plusRS / RSv4 / RSv5 via RS_VERSION)
#   20, 30, 40   → normal   (P5)
#   50, 70, 90   → explore  (P5plusA5Explore)
#   60, 80, 100  → exploit  (P5plusA5Exploit)
#
#   RS_VERSION=v3 bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh
#   RS_VERSION=v4 bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh
#   RS_VERSION=v5 bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/light_16.json}"
RS_VERSION="${RS_VERSION:-v3}"
COLLECT_SCHEDULE="${COLLECT_SCHEDULE:-rs_mix}"
TRAIN_MEMBER_POLICY="${TRAIN_MEMBER_POLICY:-adaptive}"
OUT_ROOT="${OUT_ROOT:-benchmark_data/rl_rollouts/B_rsmix_${RS_VERSION}_n4to10_T1p0_light16}"
BO_ROOT="${BO_ROOT:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/history_data}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_SAMPLES="${N_SAMPLES:-4}"
N_SAMPLES_MAX="${N_SAMPLES_MAX:-10}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-21600}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TEMP="${TEMP:-1.0}"
PREFIXES="${PREFIXES:-0 5 10 20 30 40 50 60 70 80 90 100}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/rsmix_${RS_VERSION}_prefix_rollouts_${TS}.log"

mkdir -p "$LOG_ROOT" "$OUT_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi

mapfile -t QUEUE < <("$PY" - <<PY
import json
from pathlib import Path
tasks = json.loads(Path("$TASKS_JSON").read_text())
for t in tasks:
    print(f"{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

IFS=',' read -ra GPUS <<<"$GPU_IDS"
NUM_GPUS=${#GPUS[@]}
TOTAL=${#QUEUE[@]}
N_PREFIX=$(wc -w <<<"$PREFIXES")
log "=== RS mix rollouts: $TOTAL tasks GPUs=${GPU_IDS} T=$TEMP n_samples=$N_SAMPLES n_max=$N_SAMPLES_MAX rs=$RS_VERSION prefixes=$PREFIXES ==="

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

declare -a GPU_PIDS=()
declare -a ALL_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[i]=""; done

count_live() {
  local i="$1" live=0
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live+1)); fi
  done
  echo "$live"
}

prune_pids() {
  local i="$1" kept=""
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$i]="${kept% }"
}

task_record_count() {
  local outdir="$1"
  local n=0
  if [[ -d "$outdir" ]]; then
    n=$("$PY" - <<PY
from pathlib import Path
root = Path("$outdir")
print(sum(sum(1 for ln in p.open() if ln.strip()) for p in root.rglob("rollouts_*.jsonl")))
PY
)
  fi
  echo "${n:-0}"
}

task_prefixes_complete() {
  local outdir="$1"
  "$PY" - <<PY
import json
import os
import sys
from pathlib import Path
from collections import Counter

root = Path("$outdir")
need = int("$N_SAMPLES")
sched = os.environ.get("COLLECT_SCHEDULE", "rs_mix")
rs_version = os.environ.get("RS_VERSION", "v3")
repo = Path.cwd()
sys.path.insert(0, str(repo / "examples" / "benchmark"))
if not root.is_dir():
    raise SystemExit(1)

if sched == "rs_mix_v13":
    from p5plus_rl_collect_schedule import mix_jobs_v13
    jobs = mix_jobs_v13(rs_version=rs_version)
else:
    prefixes = [int(x) for x in "$PREFIXES".split()]
    jobs = [(k, "mix", str(k)) for k in prefixes]

counts = Counter()
for p in root.rglob("rollouts_*.jsonl"):
    for line in p.open():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (int(rec["prefix_k"]), str(rec.get("scenario") or ""))
        counts[key] += 1

ok = True
for k, scenario, _ in jobs:
    key = (int(k), str(scenario))
    if counts.get(key, 0) < need:
        ok = False
        break
raise SystemExit(0 if ok else 1)
PY
}

EXPECT_PER_TASK="${EXPECT_PER_TASK:-$(
  if [[ "$COLLECT_SCHEDULE" == "rs_mix_v13" ]]; then echo $(( 21 * N_SAMPLES )); else echo $(( $(wc -w <<<"$PREFIXES") * N_SAMPLES )); fi
)}"

run_one() {
  local gpu="$1" tt="$2" ds="$3" algo="$4"
  local outdir="$OUT_ROOT/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/rsmix_${RS_VERSION}_${tt}_${ds}_${algo}_${TS}.log"
  local metric="bal_acc"
  [[ "$tt" == "rgs" ]] && metric="mse"
  mkdir -p "$outdir"
  local have
  have=$(task_record_count "$outdir")
  if task_prefixes_complete "$outdir"; then
    log "SKIP gpu=$gpu $tt/$ds/$algo rs=$RS_VERSION already complete prefixes=$PREFIXES records=$have"
    return 0
  fi
  log "START gpu=$gpu $tt/$ds/$algo rs=$RS_VERSION have=$have/$EXPECT_PER_TASK"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_bo_prefix_b.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
    --metric "$metric" \
    --collect-schedule "$COLLECT_SCHEDULE" \
    --train-member-policy "$TRAIN_MEMBER_POLICY" \
    --rs-version "$RS_VERSION" \
    --prefixes $PREFIXES \
    --n-samples "$N_SAMPLES" \
    --n-samples-max "$N_SAMPLES_MAX" \
    --bo-history-root "$BO_ROOT" \
    --seed "$SEED" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$outdir" \
    --qwen-backend hf \
    --qwen-model-path "$MODEL_PATH" \
    --qwen-adapter-path "$ADAPTER" \
    --llm-response-format json_only \
    --llm-max-tokens 768 \
    --llm-temperature "$TEMP" \
    >>"$logfile" 2>&1 &
  local pid=$!
  ALL_PIDS+=("$pid")
  local slot=-1
  for ((s=0; s<NUM_GPUS; s++)); do
    if [[ "${GPUS[$s]}" == "$gpu" ]]; then slot=$s; break; fi
  done
  if [[ -n "${GPU_PIDS[$slot]:-}" ]]; then
    GPU_PIDS[$slot]="${GPU_PIDS[$slot]} $pid"
  else
    GPU_PIDS[$slot]="$pid"
  fi
}

idx=0
while (( idx < TOTAL )); do
  launched=0
  for ((s=0; s<NUM_GPUS; s++)); do
    g="${GPUS[$s]}"
    prune_pids "$s"
    live=$(count_live "$s")
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < JOBS_PER_GPU && idx < TOTAL )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      IFS='|' read -r tt ds algo <<<"${QUEUE[$idx]}"
      idx=$((idx+1))
      run_one "$g" "$tt" "$ds" "$algo"
      launched=1
      live=$(count_live "$s")
      free=$(gpu_free_mb "$g" || echo 0)
      sleep 2
    done
  done
  if (( launched == 0 )); then
    sleep 20
  fi
done

log "All $TOTAL jobs dispatched — waiting"
fail=0
for pid in "${ALL_PIDS[@]}"; do
  if ! wait "$pid"; then
    log "FAIL pid=$pid"
    fail=1
  fi
done

"$PY" - <<PY | tee -a "$PIPE"
from pathlib import Path
root = Path("$OUT_ROOT")
n_rec = sum(sum(1 for ln in p.open() if ln.strip()) for p in root.rglob("rollouts_*.jsonl")) if root.exists() else 0
expect = $TOTAL * $N_PREFIX * $N_SAMPLES
print(f"RSmix: records={n_rec} expect~{expect}")
PY

if (( fail )); then
  log "=== RS mix rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== RS mix rollouts complete ==="
