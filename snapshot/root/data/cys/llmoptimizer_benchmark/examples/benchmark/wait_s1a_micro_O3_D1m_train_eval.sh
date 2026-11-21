#!/usr/bin/env bash
# Wait for s1a O3 + D1m half-epoch adapters, then parallel algo40 eval + tables.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
LF="${LF:-$LF_ENV/bin/llamafactory-cli}"
export PATH="$LF_ENV/bin:$PATH"
W_O3="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O3"
W_D1M="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1m"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_micro_O3_D1m_train_eval_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_adapter() {
  local name="$1" path="$2"
  log "WAIT adapter $name → $path/adapter_model.safetensors"
  while [[ ! -f "$path/adapter_model.safetensors" ]]; do
    sleep 30
  done
  log "READY $name"
}

# Start D1m train if missing (2-GPU DPO for frozen reference model)
if [[ ! -f "$W_D1M/adapter_model.safetensors" ]]; then
  if ! pgrep -f 'llamafactory.*dpo_D1m' >/dev/null 2>&1; then
    log "=== train D1m (0.5 ep DPO-anchor, 2-GPU ZeRO-3) on GPUs ${D1M_GPUS:-1,2} ==="
    CUDA_VISIBLE_DEVICES="${D1M_GPUS:-1,2}" FORCE_TORCHRUN=1 NPROC_PER_NODE=2 DISABLE_VERSION_CHECK=1 \
      nohup "$LF" train \
      "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1m_2gpu.yaml" \
      >>"$PIPE" 2>&1 &
    echo $! >"$LOG_ROOT/s1a_D1m_train.pid"
    log "D1m train pid=$(cat "$LOG_ROOT/s1a_D1m_train.pid")"
  else
    log "D1m train already running — will wait for adapter"
  fi
fi

# If O3 missing and not training, start it on GPU 0
if [[ ! -f "$W_O3/adapter_model.safetensors" ]]; then
  if ! pgrep -f 'llamafactory.*dpo_O3' >/dev/null 2>&1; then
    log "=== train O3 (0.5 ep ORPO-micro) on GPU ${O3_GPU:-0} ==="
    CUDA_VISIBLE_DEVICES="${O3_GPU:-0}" nohup "$LF" train \
      "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O3.yaml" \
      >>"$PIPE" 2>&1 &
    echo $! >"$LOG_ROOT/s1a_O3_train.pid"
    log "O3 train pid=$(cat "$LOG_ROOT/s1a_O3_train.pid")"
  else
    log "O3 train already running — will wait for adapter"
  fi
fi

wait_adapter O3 "$W_O3"
wait_adapter D1m "$W_D1M"

# Ensure no leftover train procs hogging VRAM before eval
while pgrep -f 'llamafactory.*dpo_(O3|D1m)' >/dev/null 2>&1; do
  log "WAIT llamafactory train procs to exit before eval"
  sleep 20
done

log "=== parallel eval STAGE=s1a_s2_micro (O3+D1m) SEED=1 JOBS_PER_GPU=$EVAL_JOBS_PER_GPU ==="
cd "$REPO"
NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
  STAGE=s1a_s2_micro HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT="$TIME_LIMIT" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== regenerate full + seed1 tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
  --rgs-pooled-trial-norm \
  --out-dir benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank \
  >>"$PIPE" 2>&1
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
  --rgs-pooled-trial-norm --seed1-only \
  --out-dir benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank \
  >>"$PIPE" 2>&1

log "=== s1a micro O3+D1m train+eval+tables complete ==="
