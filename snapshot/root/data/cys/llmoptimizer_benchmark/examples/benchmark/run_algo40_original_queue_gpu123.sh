#!/usr/bin/env bash
# Wait for the in-flight hard-40 fabert job, then eval the 25 original algo40 tasks
# that do NOT overlap with tasks_40_hard.json (15 shared tasks skipped).
# GPU packing: GPU1×2 workers, GPU2×3, GPU3×3
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

WAIT_PID="${WAIT_PID:-545038}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40_original_only25.json}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PIPE="$LOGDIR/algo40_original_only25_queue_${TS}.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

log "=== QUEUED original-only algo40 eval ($TASKS_JSON) ==="
log "waiting for PID $WAIT_PID to finish (hard-40 fabert ckpt16)..."

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done
log "wait done — PID $WAIT_PID exited"

python3 -c "
import json
from pathlib import Path
json.load(open('examples/benchmark/hybrid_variants.json'))
tasks = json.loads(Path('$TASKS_JSON').read_text())
hard = {(t['task_type'], t['dataset'], t['algorithm']) for t in json.load(open('benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40_hard.json'))}
orig = {(t['task_type'], t['dataset'], t['algorithm']) for t in json.load(open('benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json'))}
cur = {(t['task_type'], t['dataset'], t['algorithm']) for t in tasks}
assert len(cur) == 25, len(cur)
assert not (cur & hard), f'overlap with hard-40: {cur & hard}'
assert cur <= orig, 'tasks must be subset of original algo40'
print('ok: 25 original-only tasks, 0 hard-40 overlap')
"

log "launch: GPU1×2, GPU2×3, GPU3×3 | 75 jobs (25 tasks × 3 ckpts)"
TASKS_JSON="$TASKS_JSON" \
OUTPUT_ROOT=benchmark_data/real_hpo_v2_mltest20_algo40 \
NUM_GPUS=3 \
JOBS_PER_GPU=3 \
JOBS_PER_GPU_MAP=2,3,3 \
PACK_MIN_FREE_MB=20000 \
EVAL_GPU_MAP=1,2,3 \
STAGE=from_manifest \
MANIFEST_JSON=benchmark_data/rl_task_selection/s1b_rg_dbest_hard16_ckpt81216_manifest.json \
HYBRID=HPUCB_arm \
PROMPT_VARIANT=P5plusA5T50X50RS10 \
LLM_OPTIMIZERS=ptllm_tpe_qwen \
SEED=1 \
EVAL_RESUME=1 \
TIME_LIMIT=10800 \
bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== original-only algo40 eval done ==="
