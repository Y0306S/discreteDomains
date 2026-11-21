#!/usr/bin/env bash
# Train s2_C and s2_G (DPO) on s1a_ep7p5 (2 GPUs each), then algo40 eval (2 GPUs × 2 jobs/model).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
TRAIN_GPUS="${TRAIN_GPUS:-0,1}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-2}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_CG_pipeline_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

run_2gpu_train() {
  local name="$1" cfg="$2"
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${name}"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP train s2_${name} — adapter exists at $w (set FORCE_RETRAIN=1 to redo)"
    return 0
  fi
  local tlog="$LOG_ROOT/s1a_s2_${name}_2gpu_${TS}.log"
  log "TRAIN s2_${name} from s1a_ep7p5 on GPUs=$TRAIN_GPUS -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" FORCE_TORCHRUN=1 NPROC_PER_NODE=2 DISABLE_VERSION_CHECK=1 \
    "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${name}_2gpu.yaml" \
    >"$tlog" 2>&1
  local rc=$?
  log "TRAIN s2_${name} finished rc=$rc (log $tlog)"
  [[ $rc -eq 0 ]] || return "$rc"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w/adapter_model.safetensors"; return 1; }
}

run_eval() {
  log "EVAL algo40 s1a_s2_C + s1a_s2_G — P5/H0 ptllm_tpe_qwen only ($EVAL_NUM_GPUS GPUs × $EVAL_JOBS_PER_GPU jobs)"
  cd "$REPO"
  NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_cg HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
  log "EVAL done"
}

log "=== s1a_ep7p5 -> s2_C, s2_G pipeline ==="
log "TRAIN_GPUS=$TRAIN_GPUS EVAL_NUM_GPUS=$EVAL_NUM_GPUS EVAL_JOBS_PER_GPU=$EVAL_JOBS_PER_GPU"

if [[ "${PARALLEL_CG:-0}" == "1" ]]; then
  log "PARALLEL_CG=1: C on $TRAIN_GPUS, G on GPU ${G_TRAIN_GPU:-2}; use wait_s1a_s2_parallel_then_eval.sh for eval"
  run_2gpu_train C llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_C_2gpu.yaml &
  pidC=$!
  G_GPU="${G_TRAIN_GPU:-2}"
  (
    CUDA_VISIBLE_DEVICES="$G_GPU" DISABLE_VERSION_CHECK=1 \
      "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G.yaml" \
      >"$LOG_ROOT/s1a_s2_G_gpu${G_GPU}_${TS}.log" 2>&1
  ) &
  pidG=$!
  wait "$pidC" || true
  wait "$pidG" || true
  run_eval
else
  run_2gpu_train C llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_C_2gpu.yaml
  run_2gpu_train G llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G_2gpu.yaml
  run_eval
fi

log "=== pipeline complete ==="
log "Compare vs s1a baseline: benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_ep7p5"
log "New runs: ..._p5_s1a_s2_C and ..._p5_s1a_s2_G"
