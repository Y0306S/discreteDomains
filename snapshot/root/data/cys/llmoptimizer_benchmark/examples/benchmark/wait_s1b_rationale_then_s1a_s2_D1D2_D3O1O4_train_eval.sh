#!/usr/bin/env bash
# Wait for s1b dpo_rationale C/G, then:
#   phase 1: D1 on 0,1 || D2 on 2,3 (parallel)
#   phase 2: D3 on 0,1 || O1 on 2 || O4 on 3 (parallel)
#   then algo40 eval for D1/D2/D3/O1/O4
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="${PIPE_LOG:-$LOG_ROOT/s1a_s2_D1D2_D3O1O4_pipeline_${TS}.log}"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_pattern() {
  local pat="$1" label="$2"
  log "WAIT $label — until no processes match: $pat"
  while pgrep -f "$pat" >/dev/null 2>&1; do
    sleep 60
  done
  log "WAIT $label — clear"
}

train_2gpu() {
  local id="$1" cfg="$2" gpus="$3"
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${id}"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP train $id — adapter exists at $w"
    return 0
  fi
  local gpus_tag="${gpus//,/_}"
  local tlog="$LOG_ROOT/s1a_s2_${id}_2gpu_${gpus_tag}_${TS}.log"
  log "TRAIN $id (2-GPU DPO) CUDA_VISIBLE_DEVICES=$gpus -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  CUDA_VISIBLE_DEVICES="$gpus" FORCE_TORCHRUN=1 NPROC_PER_NODE=2 DISABLE_VERSION_CHECK=1 \
    "$LF" train "$CFG_DIR/$cfg" >"$tlog" 2>&1
  local rc=$?
  log "TRAIN $id finished rc=$rc log=$tlog"
  [[ $rc -eq 0 ]] || return "$rc"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w/adapter_model.safetensors"; return 1; }
}

train_1gpu() {
  local id="$1" cfg="$2" gpu="$3"
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${id}"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP train $id — adapter exists at $w"
    return 0
  fi
  local tlog="$LOG_ROOT/s1a_s2_${id}_gpu${gpu}_${TS}.log"
  log "TRAIN $id (1-GPU) CUDA_VISIBLE_DEVICES=$gpu -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_VERSION_CHECK=1 \
    "$LF" train "$CFG_DIR/$cfg" >"$tlog" 2>&1
  local rc=$?
  log "TRAIN $id finished rc=$rc log=$tlog"
  [[ $rc -eq 0 ]] || return "$rc"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w/adapter_model.safetensors"; return 1; }
}

run_eval() {
  log "EVAL algo40 s1a_s2 D1/D2/D3/O1/O4 — P5/H0 ptllm_tpe_qwen (NUM_GPUS=$EVAL_NUM_GPUS)"
  cd "$REPO"
  NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_conservative HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
  log "EVAL done"
}

log "=== s1a conservative DPO pipeline (4-GPU layout) ==="
log "phase1: D1@0,1 || D2@2,3 | phase2: D3@0,1 || O1@2 || O4@3 | PIPE=$PIPE"

wait_pattern 'llamafactory.*dpo_rationale' 's1b dpo_rationale C/G'

log "--- phase 1: D1 (0,1) + D2 (2,3) parallel ---"
train_2gpu D1 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1_2gpu.yaml "0,1" &
pid_d1=$!
train_2gpu D2 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D2_2gpu.yaml "2,3" &
pid_d2=$!
wait "$pid_d1"
wait "$pid_d2"

log "--- phase 2: D3 (0,1) + O1 (2) + O4 (3) parallel ---"
train_2gpu D3 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D3_2gpu.yaml "0,1" &
pid_d3=$!
train_1gpu O1 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O1.yaml "2" &
pid_o1=$!
train_1gpu O4 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O4.yaml "3" &
pid_o4=$!
wait "$pid_d3"
wait "$pid_o1"
wait "$pid_o4"

run_eval

log "=== pipeline complete ==="
log "Weights: ..._dpo_{D1,D2,D3,O1,O4}"
log "Eval roots: ${REPO}/benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_s2_*"
#!/usr/bin/env bash
# Wait for s1b dpo_rationale C/G, then:
#   phase 1: D1 on 0,1 || D2 on 2,3 (parallel)
#   phase 2: D3 on 0,1 || O1 on 2 || O4 on 3 (parallel)
#   then algo40 eval for D1/D2/D3/O1/O4
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
CFG_DIR="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/configs/ml"
LF="${LLAMAFACTORY_CLI:-/root/miniconda3/envs/llamafactory_qwen35/bin/llamafactory-cli}"
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="${PIPE_LOG:-$LOG_ROOT/s1a_s2_D1D2_D3O1O4_pipeline_${TS}.log}"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_pattern() {
  local pat="$1" label="$2"
  log "WAIT $label — until no processes match: $pat"
  while pgrep -f "$pat" >/dev/null 2>&1; do
    sleep 60
  done
  log "WAIT $label — clear"
}

train_2gpu() {
  local id="$1" cfg="$2" gpus="$3"
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${id}"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP train $id — adapter exists at $w"
    return 0
  fi
  local gpus_tag="${gpus//,/_}"
  local tlog="$LOG_ROOT/s1a_s2_${id}_2gpu_${gpus_tag}_${TS}.log"
  log "TRAIN $id (2-GPU DPO) CUDA_VISIBLE_DEVICES=$gpus -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  CUDA_VISIBLE_DEVICES="$gpus" FORCE_TORCHRUN=1 NPROC_PER_NODE=2 DISABLE_VERSION_CHECK=1 \
    "$LF" train "$CFG_DIR/$cfg" >"$tlog" 2>&1
  local rc=$?
  log "TRAIN $id finished rc=$rc log=$tlog"
  [[ $rc -eq 0 ]] || return "$rc"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w/adapter_model.safetensors"; return 1; }
}

train_1gpu() {
  local id="$1" cfg="$2" gpu="$3"
  local w="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_${id}"
  if [[ -f "$w/adapter_model.safetensors" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    log "SKIP train $id — adapter exists at $w"
    return 0
  fi
  local tlog="$LOG_ROOT/s1a_s2_${id}_gpu${gpu}_${TS}.log"
  log "TRAIN $id (1-GPU) CUDA_VISIBLE_DEVICES=$gpu -> $w"
  cd /root/LLMOptimizer/LlamaFactory
  export PATH="/root/miniconda3/envs/llamafactory_qwen35/bin:$PATH"
  CUDA_VISIBLE_DEVICES="$gpu" DISABLE_VERSION_CHECK=1 \
    "$LF" train "$CFG_DIR/$cfg" >"$tlog" 2>&1
  local rc=$?
  log "TRAIN $id finished rc=$rc log=$tlog"
  [[ $rc -eq 0 ]] || return "$rc"
  [[ -f "$w/adapter_model.safetensors" ]] || { log "ERROR missing $w/adapter_model.safetensors"; return 1; }
}

run_eval() {
  log "EVAL algo40 s1a_s2 D1/D2/D3/O1/O4 — P5/H0 ptllm_tpe_qwen (NUM_GPUS=$EVAL_NUM_GPUS)"
  cd "$REPO"
  NUM_GPUS="$EVAL_NUM_GPUS" JOBS_PER_GPU="$EVAL_JOBS_PER_GPU" \
    STAGE=s1a_s2_conservative HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen EVAL_RESUME=1 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
  log "EVAL done"
}

log "=== s1a conservative DPO pipeline (4-GPU layout) ==="
log "phase1: D1@0,1 || D2@2,3 | phase2: D3@0,1 || O1@2 || O4@3 | PIPE=$PIPE"

wait_pattern 'llamafactory.*dpo_rationale' 's1b dpo_rationale C/G'

log "--- phase 1: D1 (0,1) + D2 (2,3) parallel ---"
train_2gpu D1 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1_2gpu.yaml "0,1" &
pid_d1=$!
train_2gpu D2 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D2_2gpu.yaml "2,3" &
pid_d2=$!
wait "$pid_d1"
wait "$pid_d2"

log "--- phase 2: D3 (0,1) + O1 (2) + O4 (3) parallel ---"
train_2gpu D3 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D3_2gpu.yaml "0,1" &
pid_d3=$!
train_1gpu O1 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O1.yaml "2" &
pid_o1=$!
train_1gpu O4 llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O4.yaml "3" &
pid_o4=$!
wait "$pid_d3"
wait "$pid_o1"
wait "$pid_o4"

run_eval

log "=== pipeline complete ==="
log "Weights: ..._dpo_{D1,D2,D3,O1,O4}"
log "Eval roots: ${REPO}/benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1a_s2_*"
