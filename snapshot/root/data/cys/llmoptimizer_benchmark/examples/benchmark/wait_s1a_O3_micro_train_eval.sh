#!/usr/bin/env bash
# Train s1a ORPO-micro O3 (0.5 epoch) then algo40 eval (seed=1) and refresh tables.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
LF="/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli"
WEIGHT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O3"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_O3_micro_train_eval_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

log "=== train s1a ORPO-micro O3 (0.5 epoch, β=0.1, lr=5e-6) ==="
CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0}" "$LF" train \
  "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O3.yaml" \
  >>"$PIPE" 2>&1

if [[ ! -f "$WEIGHT/adapter_model.safetensors" ]]; then
  log "ERROR missing adapter $WEIGHT/adapter_model.safetensors"
  exit 1
fi
log "train done — adapter ready at $WEIGHT"

log "=== eval STAGE=s1a_s2_o3 SEED=1 JOBS_PER_GPU=$EVAL_JOBS_PER_GPU ==="
cd "$REPO"
NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
  STAGE=s1a_s2_o3 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT="$TIME_LIMIT" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== regenerate full + seed1 pooled tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
  --rgs-pooled-trial-norm \
  --out-dir benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank \
  >>"$PIPE" 2>&1
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
  --rgs-pooled-trial-norm --seed1-only \
  --out-dir benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank \
  >>"$PIPE" 2>&1

log "=== s1a O3 micro train+eval+tables complete ==="
