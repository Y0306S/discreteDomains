#!/usr/bin/env bash
# LoRA P5/H4 for 4 S1 adapters — up to 2 jobs per GPU (8 concurrent max).
#
# Queue is interleaved by dataset so the first 4 dispatches spread across GPUs:
#   GPU0/1/2/3 each get a different model on yeast, then car, etc.
# Second slot per GPU opens when >=28GB free on that GPU.
#
# Usage:
#   EVAL_RESUME=1 JOBS_PER_GPU=3 bash scripts/run_s1_p5h4_eval_parallel.sh
set -euo pipefail

REPO_EVAL="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune/p5_s1_pipeline}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOG_ROOT/s1_p5h4_ptllm_smac_parallel_${TS}.log"

PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algoB}"
EVAL_RESUME="${EVAL_RESUME:-1}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-28000}"
PACK_MIN_FREE_MB_THIRD="${PACK_MIN_FREE_MB_THIRD:-20000}"
POLL_SEC="${POLL_SEC:-15}"
HYBRID="${HYBRID:-H4}"
JSON_ONLY_MAXTOK="${JSON_ONLY_MAXTOK:-768}"
RATIONALE_MAXTOK="${RATIONALE_MAXTOK:-768}"

LORA_MODELS=(s1a_ep2p5 s1b_ep2p5 s1a_ep7p5 s1b_ep7p5)

declare -A ADAPTER_FOR=(
  [s1a_ep2p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep2p5"
  [s1b_ep2p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5"
  [s1a_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1b_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5"
)
declare -A MAXTOK_FOR=(
  [s1a_ep2p5]="$JSON_ONLY_MAXTOK"
  [s1a_ep7p5]="$JSON_ONLY_MAXTOK"
  [s1b_ep2p5]="$RATIONALE_MAXTOK"
  [s1b_ep7p5]="$RATIONALE_MAXTOK"
)
declare -A RESPFMT_FOR=(
  [s1a_ep2p5]=json_only
  [s1a_ep7p5]=json_only
  [s1b_ep2p5]=rationale_json
  [s1b_ep7p5]=rationale_json
)

TASKS=(
  "yeast:lightgbm"
  "car:extra_trees"
  "PhishingWebsites:xgboost"
  "segment:catboost"
  "boston:extra_trees"
  "tecator:extra_trees"
  "socmob:extra_trees"
  "pol:xgboost"
)

RESUME_FLAG=()
if [[ "$EVAL_RESUME" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

run_lora_job() {
  local model="$1" ds="$2" algo="$3" gpu="$4"
  local adapter="${ADAPTER_FOR[$model]}"
  local out="${OUTPUT_ROOT}_p5_${model}"
  local max_tok="${MAXTOK_FOR[$model]}"
  local resp_fmt="${RESPFMT_FOR[$model]}"
  local logf="$LOG_ROOT/eval_smac_${model}_${HYBRID}_${ds}_${TS}.log"
  log "LORA gpu=$gpu model=$model $HYBRID ds=$ds algo=$algo -> $logf"
  cd "$REPO_EVAL"
  export PYTHONPATH="$REPO_EVAL/examples/benchmark:${REPO_EVAL}${PYTHONPATH:+:$PYTHONPATH}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --skip-test-eval \
    --llm-backends qwen-sft6 \
    --qwen-backend hf \
    --qwen-model-path "$BASE_MODEL" \
    --qwen-adapter-path "$adapter" \
    --llm-max-tokens "$max_tok" \
    --llm-response-format "$resp_fmt" \
    --output-dir "$out" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer ptllm_smac_qwen \
    --prompt-variant P5 \
    --hybrid-variant "$HYBRID" \
    "${RESUME_FLAG[@]}" >>"$logf" 2>&1
}

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

gpu_job_count() {
  local gpu="$1" n=0 slot
  for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
    is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && n=$((n + 1))
  done
  echo "$n"
}

can_pack_on_gpu() {
  local gpu="$1" count free
  count=$(gpu_job_count "$gpu")
  [[ "$count" -lt "$JOBS_PER_GPU" ]] || return 1
  if [[ "$count" -eq 0 ]]; then
    return 0
  fi
  free=$(gpu_free_mb "$gpu")
  if [[ "$count" -eq 1 ]]; then
    [[ "$free" -ge "$PACK_MIN_FREE_MB" ]]
  else
    # Third job: ~19GB LoRA load; require 20GB free to avoid OOM on 80GB cards.
    [[ "$free" -ge "$PACK_MIN_FREE_MB_THIRD" ]]
  fi
}

mkdir -p "$LOG_ROOT"
log "=== S1 P5/H4 ptllm_smac — up to $JOBS_PER_GPU jobs/GPU × $NUM_GPUS GPUs (max $((JOBS_PER_GPU * NUM_GPUS)) concurrent) ==="

for model in "${LORA_MODELS[@]}"; do
  w="${ADAPTER_FOR[$model]}/adapter_model.safetensors"
  [[ -f "$w" ]] || { log "ERROR: missing $w"; exit 1; }
done

# Interleave: all 4 models per dataset before moving to next dataset → spreads across GPUs.
lora_jobs=()
for T in "${TASKS[@]}"; do
  IFS=':' read -r DS ALGO <<<"$T"
  for model in "${LORA_MODELS[@]}"; do
    lora_jobs+=("$model|$HYBRID|$DS|$ALGO")
  done
done

log "Queue: ${#lora_jobs[@]} jobs (interleaved: 4 models × 8 datasets)"

declare -A SLOT_PID=()
queue_idx=0
total=${#lora_jobs[@]}

is_alive() {
  [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null
}

any_slot_alive() {
  local gpu slot
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
      is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && return 0
    done
  done
  return 1
}

while [[ $queue_idx -lt $total ]] || any_slot_alive; do
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
      pid="${SLOT_PID[${gpu}_${slot}]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[${gpu}_${slot}]=
      if [[ $queue_idx -ge $total ]]; then continue; fi
      if ! can_pack_on_gpu "$gpu"; then continue; fi
      job="${lora_jobs[$queue_idx]}"
      queue_idx=$((queue_idx + 1))
      IFS='|' read -r MODEL _HYBRID DS ALGO <<<"$job"
      log "dispatch gpu=$gpu slot=$slot $MODEL $DS ($queue_idx/$total)"
      ( run_lora_job "$MODEL" "$DS" "$ALGO" "$gpu" ) &
      SLOT_PID[${gpu}_${slot}]=$!
    done
  done
  sleep "$POLL_SEC"
done
wait

log "S1 P5/H4 ptllm_smac queue complete"
cd "$REPO_EVAL"
"$PYTHON_LLM" examples/benchmark/check_mltest20_optionB_status.py --n-trials "$N_TRIALS" --output-root "$OUTPUT_ROOT" | tee -a "$MASTER_LOG"
