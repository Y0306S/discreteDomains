#!/usr/bin/env bash
# Throttled gap collect: max GAP_JOBS_PER_GPU concurrent jobs per GPU.
# Never relaunches a task that is already running. Slot occupancy is tracked
# by our PIDs (nvidia-smi lags while weights load, which previously caused storms).
#
# Usage:
#   bash examples/benchmark/run_rsmix_v12gap_collect_throttled.sh
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
PY_EVAL="${PY_EVAL:-$EVAL_ENV/bin/python}"
[[ -x "$PY_TRAIN" ]] || PY_TRAIN="/root/data/cys/blockdata_archive/miniconda3/envs/llamafactory_qwen35/bin/python"
[[ -x "$PY_EVAL" ]] || PY_EVAL="/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
GAP_PATCH_ROOT="${GAP_PATCH_ROOT:-benchmark_data/rl_rollouts/B_rsmix_v3_gap_patch_T1p0_pareto32_strict4}"
GAP_COLLECT_GPUS="${GAP_COLLECT_GPUS:-0,1,2,3}"
GAP_JOBS_PER_GPU="${GAP_JOBS_PER_GPU:-1}"
TRAIN_MEMBER_POLICY="${TRAIN_MEMBER_POLICY:-strict_first4}"
COLLECT_MODE="${COLLECT_MODE:-strict4_only}"
POLL_SEC="${POLL_SEC:-30}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="${PIPE:-$LOG_ROOT/rsmix_v12gap_collect_throttled_${TS}.log}"
mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

IFS=',' read -ra GPUS <<<"$GAP_COLLECT_GPUS"
NUM_GPUS=${#GPUS[@]}
declare -a GPU_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[i]=""; done

list_missing() {
  local flag="--tasks-needing-strict4-collect"
  if [[ "$COLLECT_MODE" == "any_missing" ]]; then
    flag="--tasks-needing-any-gap-collect"
  fi
  "$PY_TRAIN" examples/benchmark/rsmix_v12gap_gap_status.py \
    --strict4-patch "$GAP_PATCH_ROOT" "$flag" 2>/dev/null || true
}

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

running_task_names() {
  pgrep -af 'run_p5_rl_bo_prefix_b.py' 2>/dev/null \
    | grep -F "$GAP_PATCH_ROOT" \
    | grep -oE "${GAP_PATCH_ROOT}/[^ ]+" \
    | awk -F/ '{print $NF}' \
    | sort -u || true
}

task_already_running() {
  local name="$1"
  if pgrep -af 'run_p5_rl_bo_prefix_b.py' 2>/dev/null | grep -F "${GAP_PATCH_ROOT}/${name}" | grep -v grep >/dev/null; then
    return 0
  fi
  return 1
}

prune_gpu_pids() {
  local gi="$1" kept="" pid
  for pid in ${GPU_PIDS[$gi]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$gi]="${kept% }"
}

count_slot_on_gpu() {
  local gi="$1"
  prune_gpu_pids "$gi"
  local n=0 pid
  for pid in ${GPU_PIDS[$gi]:-}; do
    n=$((n + 1))
  done
  echo "$n"
}

launch_task() {
  local gi="$1" spec="$2"
  IFS='|' read -r tt ds algo have _rest <<<"$spec"
  local task_name="${tt}_${ds}_${algo}"
  local gpu="${GPUS[$gi]}"
  local outdir="$GAP_PATCH_ROOT/${task_name}"
  if task_already_running "$task_name"; then
    log "skip already-running $task_name"
    return 1
  fi
  log "launch gpu=$gpu $task_name have=$have/9 -> $outdir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY_EVAL" examples/benchmark/run_p5_rl_bo_prefix_b.py \
    --task-type "$tt" --dataset "$ds" --algorithm "$algo" \
    --metric "$( [[ "$tt" == cls ]] && echo bal_acc || echo mse )" \
    --collect-schedule rs_mix_v13_gap \
    --train-member-policy "$TRAIN_MEMBER_POLICY" \
    --rs-version v3 --n-samples 4 --n-samples-max 10 \
    --bo-history-root history_data/history_data --seed 1 --time-limit 21600 \
    --output-dir "$outdir" \
    --qwen-backend hf \
    --qwen-model-path /root/eb-public/huggingface-models/Qwen/Qwen3.5-9B \
    --qwen-adapter-path "$INIT" \
    --llm-response-format json_only --llm-max-tokens 768 --llm-temperature 1.0 \
    >>"$PIPE" 2>&1 &
  local pid=$!
  if [[ -n "${GPU_PIDS[$gi]:-}" ]]; then
    GPU_PIDS[$gi]="${GPU_PIDS[$gi]} $pid"
  else
    GPU_PIDS[$gi]="$pid"
  fi
  return 0
}

log "=== throttled gap collect jobs_per_gpu=$GAP_JOBS_PER_GPU gpus=${GAP_COLLECT_GPUS} ==="
log "pipe=$PIPE"

while true; do
  mapfile -t MISSING < <(list_missing)
  if ((${#MISSING[@]} == 0)); then
    log "all gap tasks complete"
    break
  fi

  declare -A INFLIGHT=()
  while IFS= read -r tn; do
    [[ -z "$tn" ]] && continue
    INFLIGHT["$tn"]=1
  done < <(running_task_names)

  queued=()
  for spec in "${MISSING[@]}"; do
    IFS='|' read -r tt ds algo have _rest <<<"$spec"
    tn="${tt}_${ds}_${algo}"
    if [[ -n "${INFLIGHT[$tn]:-}" ]]; then
      continue
    fi
    queued+=("$spec")
  done
  log "missing_tasks=${#MISSING[@]} inflight=${#INFLIGHT[@]} launch_queue=${#queued[@]}"

  launched=0
  qi=0
  for ((gi=0; gi<NUM_GPUS; gi++)); do
    live=$(count_slot_on_gpu "$gi")
    g="${GPUS[$gi]}"
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < GAP_JOBS_PER_GPU && qi < ${#queued[@]} )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      if launch_task "$gi" "${queued[$qi]}"; then
        launched=1
      fi
      qi=$((qi + 1))
      live=$(count_slot_on_gpu "$gi")
      free=$(gpu_free_mb "$g" || echo 0)
    done
  done
  sleep "$POLL_SEC"
done

for ((gi=0; gi<NUM_GPUS; gi++)); do
  for pid in ${GPU_PIDS[$gi]:-}; do
    wait "$pid" 2>/dev/null || true
  done
done

while pgrep -af 'run_p5_rl_bo_prefix_b.py' 2>/dev/null | grep -qF "$GAP_PATCH_ROOT"; do
  log "waiting for leftover gap workers to finish"
  sleep "$POLL_SEC"
done

mapfile -t STILL < <(list_missing)
if ((${#STILL[@]} > 0)); then
  log "ERROR still missing ${#STILL[@]} tasks"
  printf '%s\n' "${STILL[@]}" | tee -a "$PIPE"
  exit 1
fi
log "throttled gap collect DONE"
