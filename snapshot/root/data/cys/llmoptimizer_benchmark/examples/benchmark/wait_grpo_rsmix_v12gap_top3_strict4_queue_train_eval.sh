#!/usr/bin/env bash
# Wait for strict4 gap collect to finish, then merge → train (1 GPU/model) → eval (3 jobs/GPU).
#
# Usage:
#   bash examples/benchmark/wait_grpo_rsmix_v12gap_top3_strict4_queue_train_eval.sh
#   COLLECT_PID=326684 bash ...   # wait for a specific collect parent PID
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
[[ -x "$PY_TRAIN" ]] || PY_TRAIN="/root/data/cys/blockdata_archive/miniconda3/envs/llamafactory_qwen35/bin/python"

COLLECT_PID="${COLLECT_PID:-}"
THROTTLED_COLLECT_PID="${THROTTLED_COLLECT_PID:-}"
GAP_PATCH_ROOT="${GAP_PATCH_ROOT:-benchmark_data/rl_rollouts/B_rsmix_v3_gap_patch_T1p0_pareto32_strict4}"
POLL_SEC="${POLL_SEC:-120}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/grpo_v12gap_top3_strict4_queue_${TS}.log"
mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_collect_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    log "collect pid=$pid already exited"
    return 0
  fi
  log "waiting for collect pid=$pid ..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SEC"
    log "collect pid=$pid still running"
  done
  log "collect pid=$pid finished"
}

count_gap_collect_workers() {
  local n=0 pid
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    if ps -p "$pid" -o args= 2>/dev/null | grep -q 'run_p5_rl_bo_prefix_b.py'; then
      if ps -p "$pid" -o args= 2>/dev/null | grep -q "$GAP_PATCH_ROOT"; then
        n=$((n + 1))
      fi
    fi
  done < <(pgrep -f 'run_p5_rl_bo_prefix_b.py' 2>/dev/null || true)
  echo "$n"
}

wait_collect_workers_drained() {
  log "waiting for gap collect workers on $GAP_PATCH_ROOT to finish ..."
  while true; do
    local n
    n=$(count_gap_collect_workers)
    if [[ "$n" == "0" ]]; then
      log "no gap collect workers running"
      return 0
    fi
    log "gap collect workers still running: $n"
    sleep "$POLL_SEC"
  done
}

wait_gap_complete() {
  log "polling gap patch until 32/32 tasks have 9 jobs ..."
  while true; do
    if ! out=$("$PY_TRAIN" examples/benchmark/rsmix_v12gap_gap_status.py --strict4-patch "$GAP_PATCH_ROOT" 2>/dev/null); then
      log "gap status check failed; retry in ${POLL_SEC}s"
      sleep "$POLL_SEC"
      continue
    fi
    need=$(printf '%s' "$out" | "$PY_TRAIN" -c "import json,sys; print(json.load(sys.stdin)['strict4_collect_needed'])")
    log "strict4_collect_needed=$need/32"
    if [[ "$need" == "0" ]]; then
      log "gap collect complete (all tasks 9/9)"
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

log "=== queue: wait collect → merge → train → eval ==="
log "TRAIN: 1 GPU per model (GPUs 0,1,2)"
log "EVAL:  tasks_30 (30 datasets), GPUs 0-3, NUM_GPUS=4 JOBS_PER_GPU=2 (8 parallel workers)"
log "queue_log=$PIPE"

wait_collect_pid "$COLLECT_PID"
wait_collect_pid "$THROTTLED_COLLECT_PID"
wait_gap_complete
wait_collect_workers_drained

log "=== starting merge + train + eval ==="
SKIP_GAP_COLLECT=1 \
TRAIN_GPUS="0 1 2" \
EVAL_TASKS_JSON=benchmark_data/real_hpo_v2_mltest20_algo40/tasks_30.json \
EVAL_NUM_GPUS=4 \
EVAL_JOBS_PER_GPU=2 \
TRAIN_MEMBER_POLICY=strict_first4 \
RETAG_POLICY=strict_first4 \
TRAIN_STRICT_FIRST_N=4 \
bash examples/benchmark/wait_grpo_rsmix_v12gap_top3_strict4_train_eval.sh >>"$PIPE" 2>&1

log "=== queue DONE ==="
