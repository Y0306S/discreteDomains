#!/usr/bin/env bash
# After the in-flight H0 rank finishes, stop the old all-sequential launcher
# and continue with HPUCB_arm (alone) then mixed parallel for the rest.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
LOGDIR="benchmark_data/real_hpo_v2_mltest20_algo40/logs"
OLD_LOG="${OLD_LOG:-$LOGDIR/qwen_p5plus_attr_t2_20260818_174224.log}"
OLD_LAUNCHER_PID="${OLD_LAUNCHER_PID:-784993}"
HANDOVER_LOG="$LOGDIR/qwen_p5plus_attr_t2_handover.log"

log() { echo "[$(date -Is)] $*" | tee -a "$HANDOVER_LOG"; }

log "handover watching $OLD_LOG launcher_pid=$OLD_LAUNCHER_PID"

while true; do
  if grep -q 'RANK 1 complete' "$OLD_LOG" 2>/dev/null; then
    log "saw RANK 1 complete"
    break
  fi
  if ! kill -0 "$OLD_LAUNCHER_PID" 2>/dev/null; then
    log "old launcher $OLD_LAUNCHER_PID already exited before RANK 1 complete — abort"
    exit 1
  fi
  sleep 10
done

# Stop the old sequential sweep before it starts rank 2 (H2).
if kill -0 "$OLD_LAUNCHER_PID" 2>/dev/null; then
  log "stopping old launcher $OLD_LAUNCHER_PID"
  kill "$OLD_LAUNCHER_PID" 2>/dev/null || true
  sleep 3
  kill -9 "$OLD_LAUNCHER_PID" 2>/dev/null || true
fi

# Any rank-2 H2 python that slipped through — resume will redo them later.
mapfile -t H2_PIDS < <(pgrep -f 'run_toy_categorical_optimizers.py.*--hybrid-variant H2' || true)
if [[ ${#H2_PIDS[@]} -gt 0 ]]; then
  log "killing slipped H2 pids: ${H2_PIDS[*]}"
  kill "${H2_PIDS[@]}" 2>/dev/null || true
  sleep 2
  kill -9 "${H2_PIDS[@]}" 2>/dev/null || true
fi

log "starting seq H0 (resume skip) + HPUCB_arm then parallel rest"
JOBS_PER_GPU=3 NUM_GPUS=4 MODE=seq_then_parallel \
  SEQ_GPT_VARIANTS='ptllm_tpe__P5plusA5T50X50__H0,ptllm_tpe__P5plusA5T50X50__HPUCB_arm' \
  nohup bash examples/benchmark/run_p5plus_qwen_attr_t2_eval.sh \
  >>"$LOGDIR/qwen_p5plus_attr_t2_phase2.log" 2>&1 &
echo $! >"$LOGDIR/qwen_p5plus_attr_t2_phase2.pid"
log "phase2 pid=$(cat "$LOGDIR/qwen_p5plus_attr_t2_phase2.pid")"
