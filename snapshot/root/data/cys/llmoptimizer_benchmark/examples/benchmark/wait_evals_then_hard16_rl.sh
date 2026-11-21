#!/usr/bin/env bash
# Wait for current P5+ Qwen eval sweeps to finish, then run hard16 RL pipeline.
#
#   nohup bash examples/benchmark/wait_evals_then_hard16_rl.sh \
#     >> benchmark_data/real_hpo_v2_mltest20_algo40/logs/hard16_rl_waiter.out 2>&1 &
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-60}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/hard16_rl_waiter_${TS}.log"

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

evals_still_running() {
  pgrep -f '[r]un_p5plus_qwen_rs10_hpucb_decay_sweep.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5T50X50RS10' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_p5plus_qwen_combo_fanova_rs_hpucb_arm.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5TRdyFanovaX50RS' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_p5plus_qwen_fanova_full_and_tpe_xor.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5TRdyFanovaFullX50' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5T50XTPE' >/dev/null 2>&1 && return 0
  return 1
}

log "=== hard16 RL waiter START ==="
while evals_still_running; do
  n_rs=$(pgrep -cf '[P]5plusA5T50X50RS10' || true)
  n_combo=$(pgrep -cf '[P]5plusA5TRdyFanovaX50RS' || true)
  n_fanova=$(pgrep -cf '[P]5plusA5TRdyFanovaFullX50' || true)
  log "waiting current evals: RS10≈${n_rs} combo≈${n_combo} fanova≈${n_fanova}"
  sleep "$WAIT_POLL_SEC"
done
log "current evals finished — launching hard16 RL pipeline"

bash examples/benchmark/run_hard16_rl_dbest_ckpt81216.sh >>"$MASTER_LOG" 2>&1
log "=== hard16 RL waiter DONE ==="
echo "$MASTER_LOG"
