#!/usr/bin/env bash
# After hard16 RL eval finishes, top up the 17 missing Qwen RS10 PUCB 70→30 slots
# (P5plusA5T50X50RS10 + HPUCB_arm_p07). GPUs 1-3 only; no GPT.
#
#   nohup bash examples/benchmark/wait_hard16_then_rs10_p07_topup.sh \
#     >> benchmark_data/real_hpo_v2_mltest20_algo40/logs/wait_hard16_then_p07.out 2>&1 &
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-60}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/wait_hard16_then_p07_${TS}.log"
NUM_GPUS="${NUM_GPUS:-3}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

# Only block on the old P5/H0 hard16 eval — not RS10 re-eval (runs after p07).
hard16_still_running() {
  pgrep -f '[r]un_hard16_rl_dbest_ckpt81216.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[r]un_toy_categorical_optimizers.py.*s1b_rg_dbest_hard16.*--prompt-variant P5 --hybrid-variant H0' >/dev/null 2>&1 && return 0
  return 1
}

log "=== wait hard16 then RS10 p07 top-up START ==="
log "NUM_GPUS=$NUM_GPUS EVAL_GPU_MAP=$EVAL_GPU_MAP JOBS_PER_GPU=$JOBS_PER_GPU"

while hard16_still_running; do
  n=$(pgrep -cf '[s]1b_rg_dbest_hard16_ckpt' || true)
  log "waiting hard16 eval: workers≈${n}"
  sleep "$WAIT_POLL_SEC"
done
log "hard16 eval finished — launching RS10 HPUCB_arm_p07 top-up (17 tasks)"

NUM_GPUS="$NUM_GPUS" EVAL_GPU_MAP="$EVAL_GPU_MAP" JOBS_PER_GPU="$JOBS_PER_GPU" \
  bash examples/benchmark/run_p5plus_qwen_rs10_arm3_topup.sh >>"$MASTER_LOG" 2>&1

log "verify coverage"
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - <<'PY' | tee -a "$MASTER_LOG"
from pathlib import Path
from plot_mltest20_algo40_avg_rank import find_traj, load_tasks
tasks = load_tasks(Path("benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json"))
root = Path("benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus")
disk = "ptllm_tpe_qwen__P5plusA5T50X50RS10__HPUCB_arm_p07"
n = sum(1 for t in tasks if find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"]))
miss = [f"{t['task_type']}/{t['dataset']}/{t['algorithm']}" for t in tasks
        if not find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"])]
print(f"p07 coverage: {n}/40")
if miss:
    print("still missing:", ", ".join(miss))
PY

log "=== wait hard16 then RS10 p07 top-up DONE ==="
echo "$MASTER_LOG"
