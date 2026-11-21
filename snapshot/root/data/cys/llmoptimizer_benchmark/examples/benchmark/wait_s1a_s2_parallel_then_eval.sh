#!/usr/bin/env bash
# Wait for in-flight s1a s2_C + s2_G training, then algo40 eval (no duplicate G on 0,1).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-2}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
G_GPU="${G_TRAIN_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_CG_parallel_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_train() {
  local label="$1" pattern="$2"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    log "waiting $label ($pattern)"
    sleep 45
  done
  log "$label train process gone"
}

start_g_on_gpu() {
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN_G:-0}" != "1" ]]; then
    log "s2_G adapter already at $w — skip ORPO train"
    return 0
  fi
  if pgrep -f 'llamafactory.*s1a_ep7p5_dpo_G' >/dev/null 2>&1; then
    log "s2_G train already running"
    return 0
  fi
  local tlog="$LOG_ROOT/s1a_s2_G_gpu${G_GPU}_${TS}.log"
  log "START s2_G ORPO on GPU $G_GPU -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  (
    CUDA_VISIBLE_DEVICES="$G_GPU" DISABLE_VERSION_CHECK=1 \
      "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G.yaml" \
      >>"$tlog" 2>&1
  ) &
  log "s2_G train pid=$! log=$tlog"
}

log "=== parallel wait + eval orchestrator ==="
start_g_on_gpu
wait_train "s2_C" 'llamafactory.*s1a_ep7p5_dpo_C'
wait_train "s2_G" 'llamafactory.*s1a_ep7p5_dpo_G'

for name in C G; do
  w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${name}"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w"; exit 1; }
done

log "EVAL P5/H0 tpe: s1a_s2_C on GPUs ${C_EVAL_GPUS:-0,1}, s1a_s2_G on GPUs ${G_EVAL_GPUS:-2,3} (parallel)"
cd "$REPO"
(
  CUDA_VISIBLE_DEVICES="${C_EVAL_GPUS:-0,1}" EVAL_GPU_MAP="${C_EVAL_GPUS:-0,1}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_c HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidC=$!
(
  CUDA_VISIBLE_DEVICES="${G_EVAL_GPUS:-2,3}" EVAL_GPU_MAP="${G_EVAL_GPUS:-2,3}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_g HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidG=$!
wait "$pidC" || true
wait "$pidG" || true
log "=== done ==="
#!/usr/bin/env bash
# Wait for in-flight s1a s2_C + s2_G training, then algo40 eval (no duplicate G on 0,1).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-2}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
G_GPU="${G_TRAIN_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_CG_parallel_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_train() {
  local label="$1" pattern="$2"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    log "waiting $label ($pattern)"
    sleep 45
  done
  log "$label train process gone"
}

start_g_on_gpu() {
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN_G:-0}" != "1" ]]; then
    log "s2_G adapter already at $w — skip ORPO train"
    return 0
  fi
  if pgrep -f 'llamafactory.*s1a_ep7p5_dpo_G' >/dev/null 2>&1; then
    log "s2_G train already running"
    return 0
  fi
  local tlog="$LOG_ROOT/s1a_s2_G_gpu${G_GPU}_${TS}.log"
  log "START s2_G ORPO on GPU $G_GPU -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  (
    CUDA_VISIBLE_DEVICES="$G_GPU" DISABLE_VERSION_CHECK=1 \
      "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G.yaml" \
      >>"$tlog" 2>&1
  ) &
  log "s2_G train pid=$! log=$tlog"
}

log "=== parallel wait + eval orchestrator ==="
start_g_on_gpu
wait_train "s2_C" 'llamafactory.*s1a_ep7p5_dpo_C'
wait_train "s2_G" 'llamafactory.*s1a_ep7p5_dpo_G'

for name in C G; do
  w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${name}"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w"; exit 1; }
done

log "EVAL P5/H0 tpe: s1a_s2_C on GPUs ${C_EVAL_GPUS:-0,1}, s1a_s2_G on GPUs ${G_EVAL_GPUS:-2,3} (parallel)"
cd "$REPO"
(
  CUDA_VISIBLE_DEVICES="${C_EVAL_GPUS:-0,1}" EVAL_GPU_MAP="${C_EVAL_GPUS:-0,1}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_c HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidC=$!
(
  CUDA_VISIBLE_DEVICES="${G_EVAL_GPUS:-2,3}" EVAL_GPU_MAP="${G_EVAL_GPUS:-2,3}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_g HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidG=$!
wait "$pidC" || true
wait "$pidG" || true
log "=== done ==="
#!/usr/bin/env bash
# Wait for in-flight s1a s2_C + s2_G training, then algo40 eval (no duplicate G on 0,1).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-2}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
G_GPU="${G_TRAIN_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_CG_parallel_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_train() {
  local label="$1" pattern="$2"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    log "waiting $label ($pattern)"
    sleep 45
  done
  log "$label train process gone"
}

start_g_on_gpu() {
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN_G:-0}" != "1" ]]; then
    log "s2_G adapter already at $w — skip ORPO train"
    return 0
  fi
  if pgrep -f 'llamafactory.*s1a_ep7p5_dpo_G' >/dev/null 2>&1; then
    log "s2_G train already running"
    return 0
  fi
  local tlog="$LOG_ROOT/s1a_s2_G_gpu${G_GPU}_${TS}.log"
  log "START s2_G ORPO on GPU $G_GPU -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  (
    CUDA_VISIBLE_DEVICES="$G_GPU" DISABLE_VERSION_CHECK=1 \
      "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G.yaml" \
      >>"$tlog" 2>&1
  ) &
  log "s2_G train pid=$! log=$tlog"
}

log "=== parallel wait + eval orchestrator ==="
start_g_on_gpu
wait_train "s2_C" 'llamafactory.*s1a_ep7p5_dpo_C'
wait_train "s2_G" 'llamafactory.*s1a_ep7p5_dpo_G'

for name in C G; do
  w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${name}"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w"; exit 1; }
done

log "EVAL P5/H0 tpe: s1a_s2_C on GPUs ${C_EVAL_GPUS:-0,1}, s1a_s2_G on GPUs ${G_EVAL_GPUS:-2,3} (parallel)"
cd "$REPO"
(
  CUDA_VISIBLE_DEVICES="${C_EVAL_GPUS:-0,1}" EVAL_GPU_MAP="${C_EVAL_GPUS:-0,1}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_c HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidC=$!
(
  CUDA_VISIBLE_DEVICES="${G_EVAL_GPUS:-2,3}" EVAL_GPU_MAP="${G_EVAL_GPUS:-2,3}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_g HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidG=$!
wait "$pidC" || true
wait "$pidG" || true
log "=== done ==="
#!/usr/bin/env bash
# Wait for in-flight s1a s2_C + s2_G training, then algo40 eval (no duplicate G on 0,1).
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-2}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
G_GPU="${G_TRAIN_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/s1a_s2_CG_parallel_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_train() {
  local label="$1" pattern="$2"
  while pgrep -f "$pattern" >/dev/null 2>&1; do
    log "waiting $label ($pattern)"
    sleep 45
  done
  log "$label train process gone"
}

start_g_on_gpu() {
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN_G:-0}" != "1" ]]; then
    log "s2_G adapter already at $w — skip ORPO train"
    return 0
  fi
  if pgrep -f 'llamafactory.*s1a_ep7p5_dpo_G' >/dev/null 2>&1; then
    log "s2_G train already running"
    return 0
  fi
  local tlog="$LOG_ROOT/s1a_s2_G_gpu${G_GPU}_${TS}.log"
  log "START s2_G ORPO on GPU $G_GPU -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  (
    CUDA_VISIBLE_DEVICES="$G_GPU" DISABLE_VERSION_CHECK=1 \
      "$LF" train "$CFG_DIR/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G.yaml" \
      >>"$tlog" 2>&1
  ) &
  log "s2_G train pid=$! log=$tlog"
}

log "=== parallel wait + eval orchestrator ==="
start_g_on_gpu
wait_train "s2_C" 'llamafactory.*s1a_ep7p5_dpo_C'
wait_train "s2_G" 'llamafactory.*s1a_ep7p5_dpo_G'

for name in C G; do
  w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${name}"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w"; exit 1; }
done

log "EVAL P5/H0 tpe: s1a_s2_C on GPUs ${C_EVAL_GPUS:-0,1}, s1a_s2_G on GPUs ${G_EVAL_GPUS:-2,3} (parallel)"
cd "$REPO"
(
  CUDA_VISIBLE_DEVICES="${C_EVAL_GPUS:-0,1}" EVAL_GPU_MAP="${C_EVAL_GPUS:-0,1}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_c HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidC=$!
(
  CUDA_VISIBLE_DEVICES="${G_EVAL_GPUS:-2,3}" EVAL_GPU_MAP="${G_EVAL_GPUS:-2,3}" NUM_GPUS=2 JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_g HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
) &
pidG=$!
wait "$pidC" || true
wait "$pidG" || true
log "=== done ==="
