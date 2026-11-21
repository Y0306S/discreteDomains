#!/usr/bin/env bash
# Compare Huawei HEBO vs existing TPE planner on the two top s1b_s2g_orpo LoRA arms.
# Same P5/H0 mix, 30 ranking tasks, GPUs 1-3 × 3 workers.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_30.json}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PIPE="$LOGDIR/hebo_vs_tpe_s2g_top2_tasks30_${TS}.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

log "=== HEBO vs TPE s2g top2 tasks30 ts=$TS ==="
log "GPUs 1-3 × 3 workers | tasks=$TASKS_JSON"

TASKS_JSON="$TASKS_JSON" \
OUTPUT_ROOT=benchmark_data/real_hpo_v2_mltest20_algo40 \
NUM_GPUS=3 \
JOBS_PER_GPU=3 \
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-12000}" \
EVAL_GPU_MAP=1,2,3 \
STAGE=from_manifest \
MANIFEST_JSON=benchmark_data/rl_task_selection/s1b_s2g_orpo_hebo_vs_tpe_top2_manifest.json \
HYBRID=H0 \
PROMPT_VARIANT=P5 \
LLM_OPTIMIZERS=ptllm_hebo_qwen \
SEED=1 \
EVAL_RESUME=1 \
TIME_LIMIT=10800 \
N_TRIALS=80 \
PYTHON_CLASS=/root/data/cys/blockdata_archive/miniconda3/envs/openbox_eval/bin/python \
PYTHON_LLM=/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python \
bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== HEBO vs TPE s2g top2 tasks30 done ==="
