#!/usr/bin/env bash
# Setup A rollouts: BO-seed 100 + 4 LLM CASH steps, light-16, T=1.0, init s1a_ep7p5.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/light_16.json}"
OUT_ROOT="${OUT_ROOT:-benchmark_data/rl_rollouts/A_boseed100_new4_T1p0_light16}"
BO_ROOT="${BO_ROOT:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data/history_data}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_NEW="${N_NEW:-4}"
BO_SEED_N="${BO_SEED_N:-100}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TEMP="${TEMP:-1.0}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_setupA_boseed_rollouts_${TS}.log"

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
log "=== Setup A rollouts: $TOTAL light16 jobs GPUs=${GPU_IDS} T=$TEMP bo_seed=$BO_SEED_N n_new=$N_NEW ==="

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

run_one() {
  local gpu="$1" tt="$2" ds="$3" algo="$4"
  local outdir="$OUT_ROOT/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/setupA_${tt}_${ds}_${algo}_${TS}.log"
  local metric="bal_acc"
  [[ "$tt" == "rgs" ]] && metric="mse"
  mkdir -p "$outdir"
  log "START gpu=$gpu $tt/$ds/$algo"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_bo_seeded_a.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
    --metric "$metric" \
    --n-new-trials "$N_NEW" \
    --bo-seed-n "$BO_SEED_N" \
    --bo-history-root "$BO_ROOT" \
    --seed "$SEED" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$outdir" \
    --prompt-variant P5 \
    --qwen-backend hf \
    --qwen-model-path "$MODEL_PATH" \
    --qwen-adapter-path "$ADAPTER" \
    --llm-response-format json_only \
    --llm-max-tokens 768 \
    --llm-temperature "$TEMP" \
    --resume \
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
n_rec = sum(sum(1 for ln in p.open() if ln.strip()) for p in root.rglob("rollouts_*.jsonl"))
n4 = sum(1 for p in root.rglob("rollouts_*.jsonl") if sum(1 for ln in p.open() if ln.strip()) >= $N_NEW)
print(f"SetupA: records={n_rec} tasks_with_{$N_NEW}plus={n4}")
PY

if (( fail )); then
  log "=== Setup A rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== Setup A rollouts complete ==="
