#!/usr/bin/env bash
# Retry the 16 hard-40 tasks that failed due to corrupted hybrid_variants.json.
# Uses GPUs 2 and 3 only (GPU 1 may still be running fabert from the main eval).
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
mkdir -p "$LOGDIR"
PIPE="$LOGDIR/hard16_retry16_gpu23_${TS}.log"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

# Verify config is loadable before launching 48 jobs.
python3 -c "import json; json.load(open('examples/benchmark/hybrid_variants.json')); print('hybrid_variants.json OK')"

log "=== RETRY 16 tasks × 3 ckpts on GPU 2,3 ==="
TASKS_JSON=benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40_hard_retry16.json \
OUTPUT_ROOT=benchmark_data/real_hpo_v2_mltest20_algo40 \
NUM_GPUS=2 \
JOBS_PER_GPU=3 \
PACK_MIN_FREE_MB=20000 \
EVAL_GPU_MAP=2,3 \
STAGE=from_manifest \
MANIFEST_JSON=benchmark_data/rl_task_selection/s1b_rg_dbest_hard16_ckpt81216_manifest.json \
HYBRID=HPUCB_arm \
PROMPT_VARIANT=P5plusA5T50X50RS10 \
LLM_OPTIMIZERS=ptllm_tpe_qwen \
SEED=1 \
EVAL_RESUME=1 \
TIME_LIMIT=10800 \
bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== RETRY done ==="
