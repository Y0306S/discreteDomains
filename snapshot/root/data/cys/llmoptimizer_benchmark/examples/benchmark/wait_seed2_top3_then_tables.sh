#!/usr/bin/env bash
# Seed=2 replications for:
#   dpo_G_r2 (s1b s2_G), s1a_s2_D1_r2, dpo_rationale_G_r2
# Then regenerate avg-rank / pooled-norm tables.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/seed2_top3_then_tables_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

cd "$REPO"

log "=== SEED=2 eval: dpo_G_r2 + s1a_s2_D1_r2 + dpo_rationale_G_r2 (JOBS_PER_GPU=$EVAL_JOBS_PER_GPU) ==="
NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
  STAGE=seed2_top3 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=2 EVAL_RESUME=1 TIME_LIMIT="$TIME_LIMIT" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== regenerating pooled-norm / avg-rank tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
  --rgs-pooled-trial-norm \
  --out-dir benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank \
  >>"$PIPE" 2>&1

log "=== seed2_top3 + tables complete ==="
