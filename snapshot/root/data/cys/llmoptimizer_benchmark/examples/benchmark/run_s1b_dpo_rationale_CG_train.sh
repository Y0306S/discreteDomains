#!/usr/bin/env bash
# Export p5_s2_dpo_rationale (if needed), then train DPO-C + ORPO-G on s1b_ep7p5.
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
PY="${LLAMAFACTORY_PY:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
BENCH="/root/data/cys/blockdata_archive/llmoptimizer_benchmark/examples/benchmark"
DATA_TRAIN="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_train.jsonl"
MIN_PAIRS="${MIN_PAIRS:-1650}"
EXPORT_GPU="${EXPORT_GPU:-0}"
C_GPUS="${C_GPUS:-0,1}"
G_GPU="${G_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1b_dpo_rationale_CG_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

count_pairs() {
  [[ -f "$DATA_TRAIN" ]] && wc -l <"$DATA_TRAIN" || echo 0
}

run_export() {
  local elog="$LOG_ROOT/p5_s2_dpo_rationale_export_${TS}.log"
  local pg="${PARALLEL_GPUS:-0,1,2,3}"
  local wpg="${WORKERS_PER_GPU:-3}"
  log "EXPORT rationale DPO parallel GPUs=$pg workers/GPU=$wpg -> $DATA_TRAIN (log $elog)"
  cd "$BENCH"
  "$PY" export_p5_s2_dpo_rationale.py \
    --reject-rationale-mode qwen \
    --parallel-gpus "$pg" \
    --workers-per-gpu "$wpg" \
    >"$elog" 2>&1
}

wait_export() {
  local n
  n="$(count_pairs)"
  if [[ "$n" -ge "$MIN_PAIRS" ]]; then
    log "EXPORT skip — $n pairs already in $DATA_TRAIN"
    return 0
  fi
  run_export
  n="$(count_pairs)"
  log "EXPORT done — $n pairs"
  [[ "$n" -ge "$MIN_PAIRS" ]] || { log "ERROR expected >=$MIN_PAIRS pairs, got $n"; return 1; }
}

train_c() {
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_C"
  local tlog="$LOG_ROOT/s1b_dpo_rationale_C_2gpu_${TS}.log"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP train rationale_C — $w exists"
    return 0
  fi
  log "TRAIN rationale_C on GPUs=$C_GPUS -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  CUDA_VISIBLE_DEVICES="$C_GPUS" FORCE_TORCHRUN=1 NPROC_PER_NODE=2 DISABLE_VERSION_CHECK=1 \
    "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_C_2gpu.yaml" \
    >"$tlog" 2>&1
  log "TRAIN rationale_C rc=$? log=$tlog"
}

train_g() {
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G"
  local tlog="$LOG_ROOT/s1b_dpo_rationale_G_gpu${G_GPU}_${TS}.log"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP train rationale_G — $w exists"
    return 0
  fi
  log "TRAIN rationale_G ORPO on GPU=$G_GPU -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  CUDA_VISIBLE_DEVICES="$G_GPU" DISABLE_VERSION_CHECK=1 \
    "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G.yaml" \
    >"$tlog" 2>&1
  log "TRAIN rationale_G rc=$? log=$tlog"
}

log "=== s1b DPO rationale C+G pipeline ==="
wait_export
train_c &
pid_c=$!
train_g &
pid_g=$!
wait "$pid_c" "$pid_g"
log "=== pipeline complete ==="
