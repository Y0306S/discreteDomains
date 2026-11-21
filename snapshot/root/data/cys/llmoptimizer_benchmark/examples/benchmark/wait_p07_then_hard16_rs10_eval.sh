#!/usr/bin/env bash
# After RS10 HPUCB_arm_p07 (70→30) top-up finishes, re-eval hard16 RL ckpts on:
#   P5plusA5T50X50RS10 + HPUCB_arm (90→10 PUCB), tasks_40_hard, GPUs 1-3
#
#   nohup bash examples/benchmark/wait_p07_then_hard16_rs10_eval.sh \
#     >> benchmark_data/real_hpo_v2_mltest20_algo40/logs/wait_p07_then_hard16_rs10.out 2>&1 &
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-60}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/wait_p07_then_hard16_rs10_${TS}.log"
NUM_GPUS="${NUM_GPUS:-3}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

p07_waiter_running() {
  pgrep -f '[w]ait_hard16_then_rs10_p07_topup.sh' >/dev/null 2>&1 && return 0
  return 1
}

p07_still_running() {
  pgrep -f '[r]un_p5plus_qwen_rs10_arm3_topup.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_toy_categorical_optimizers.py.*--hybrid-variant HPUCB_arm_p07' >/dev/null 2>&1 && return 0
  return 1
}

p07_coverage() {
  PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
  export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
  "$PYTHON" - <<'PY'
from pathlib import Path
from plot_mltest20_algo40_avg_rank import find_traj, load_tasks
tasks = load_tasks(Path("benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json"))
root = Path("benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus")
disk = "ptllm_tpe_qwen__P5plusA5T50X50RS10__HPUCB_arm_p07"
n = sum(1 for t in tasks if find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"]))
print(n)
PY
}

hard16_rs10_still_running() {
  pgrep -f '[r]un_hard16_rl_dbest_ckpt81216.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_toy_categorical_optimizers.py.*s1b_rg_dbest_hard16.*P5plusA5T50X50RS10' >/dev/null 2>&1 && return 0
  return 1
}

log "=== wait p07 then hard16 RS10/HPUCB_arm eval START ==="

while p07_waiter_running || p07_still_running; do
  n=$(pgrep -cf 'HPUCB_arm_p07' || true)
  w=$(pgrep -cf 'wait_hard16_then_rs10_p07_topup' || true)
  log "waiting RS10 p07 top-up: workers≈${n} p07_waiter≈${w}"
  sleep "$WAIT_POLL_SEC"
done

while [[ "$(p07_coverage)" != "40" ]]; do
  cov=$(p07_coverage)
  log "waiting p07 coverage 40/40 (currently ${cov}/40)"
  sleep "$WAIT_POLL_SEC"
done
log "RS10 p07 top-up complete: 40/40"

while hard16_rs10_still_running; do
  n=$(pgrep -cf 's1b_rg_dbest_hard16.*P5plusA5T50X50RS10' || true)
  log "waiting stray hard16 RS10 eval to stop: workers≈${n}"
  sleep "$WAIT_POLL_SEC"
done

log "launching hard16 RS10/HPUCB_arm eval (EVAL_ONLY)"
NUM_GPUS="$NUM_GPUS" EVAL_GPU_MAP="$EVAL_GPU_MAP" JOBS_PER_GPU="$JOBS_PER_GPU" \
  EVAL_ONLY=1 EVAL_PROMPT=P5plusA5T50X50RS10 EVAL_HYBRID=HPUCB_arm EVAL_RESUME=1 \
  bash examples/benchmark/run_hard16_rl_dbest_ckpt81216.sh >>"$MASTER_LOG" 2>&1

log "=== wait p07 then hard16 RS10/HPUCB_arm eval DONE ==="
echo "$MASTER_LOG"
