#!/usr/bin/env bash
# Re-run LoRA mltest20 Option B for the 3 broken boosting datasets only:
#   yeast (lightgbm), PhishingWebsites (xgboost), pol (xgboost)
# Assumes broken run dirs were deleted. Uses q-aligned prompts + grid snap
# (code already patched) and per_run_time_limit=300.
#
# Usage:
#   EVAL_RESUME=1 JOBS_PER_GPU=1 bash scripts/run_mltest20_boosting_repair.sh
set -euo pipefail

REPO_EVAL="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune/p5_s1_pipeline}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOG_ROOT/boosting_repair_${TS}.log"

PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-28800}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-300}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algoB}"
EVAL_RESUME="${EVAL_RESUME:-1}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-28000}"
POLL_SEC="${POLL_SEC:-20}"
JSON_ONLY_MAXTOK="${JSON_ONLY_MAXTOK:-768}"
RATIONALE_MAXTOK="${RATIONALE_MAXTOK:-768}"

LORA_MODELS=(s0 s1a_ep2p5 s1b_ep2p5 s1a_ep7p5 s1b_ep7p5)
HYBRIDS=(H0 H4)

declare -A ADAPTER_FOR=(
  [s0]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0"
  [s1a_ep2p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep2p5"
  [s1b_ep2p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5"
  [s1a_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1b_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5"
)
declare -A MAXTOK_FOR=(
  [s0]="$JSON_ONLY_MAXTOK"
  [s1a_ep2p5]="$JSON_ONLY_MAXTOK"
  [s1a_ep7p5]="$JSON_ONLY_MAXTOK"
  [s1b_ep2p5]="$RATIONALE_MAXTOK"
  [s1b_ep7p5]="$RATIONALE_MAXTOK"
)
declare -A RESPFMT_FOR=(
  [s0]=json_only
  [s1a_ep2p5]=json_only
  [s1a_ep7p5]=json_only
  [s1b_ep2p5]=rationale_json
  [s1b_ep7p5]=rationale_json
)

# Only the broken boosting tasks
TASKS=(
  "yeast:lightgbm"
  "PhishingWebsites:xgboost"
  "pol:xgboost"
)

RESUME_FLAG=()
if [[ "$EVAL_RESUME" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

is_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

run_lora_job() {
  local model="$1" hybrid="$2" ds="$3" algo="$4" gpu="$5"
  local adapter="${ADAPTER_FOR[$model]}"
  local out="${OUTPUT_ROOT}_p5_${model}"
  local max_tok="${MAXTOK_FOR[$model]}"
  local resp_fmt="${RESPFMT_FOR[$model]}"
  local logf="$LOG_ROOT/repair_${model}_${hybrid}_${ds}_${TS}.log"
  log "REPAIR gpu=$gpu model=$model $hybrid ds=$ds algo=$algo per_run=${PER_RUN_TIME_LIMIT}s -> $logf"
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
    --per-run-time-limit "$PER_RUN_TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer ptllm_tpe_qwen \
    --prompt-variant P5 \
    --hybrid-variant "$hybrid" \
    "${RESUME_FLAG[@]}" >>"$logf" 2>&1
}

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

declare -A SLOT_PID
for ((g = 0; g < NUM_GPUS; g++)); do
  for ((s = 0; s < JOBS_PER_GPU; s++)); do
    SLOT_PID[${g}_${s}]=""
  done
done

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
  [[ "$free" -ge "$PACK_MIN_FREE_MB" ]]
}

find_slot() {
  local g s
  for ((g = 0; g < NUM_GPUS; g++)); do
    if can_pack_on_gpu "$g"; then
      for ((s = 0; s < JOBS_PER_GPU; s++)); do
        if ! is_alive "${SLOT_PID[${g}_${s}]:-}"; then
          echo "$g $s"
          return 0
        fi
      done
    fi
  done
  return 1
}

# Build queue: interleave by dataset so GPUs spread across models
QUEUE=()
for T in "${TASKS[@]}"; do
  IFS=':' read -r DS ALGO <<<"$T"
  for model in "${LORA_MODELS[@]}"; do
    for hybrid in "${HYBRIDS[@]}"; do
      QUEUE+=("${model}:${hybrid}:${DS}:${ALGO}")
    done
  done
done

log "Boosting repair start: ${#QUEUE[@]} jobs models=${LORA_MODELS[*]} hybrids=${HYBRIDS[*]}"
log "per_run_time_limit=${PER_RUN_TIME_LIMIT}s time_limit=${TIME_LIMIT}s jobs_per_gpu=${JOBS_PER_GPU}"

qi=0
while [[ "$qi" -lt "${#QUEUE[@]}" ]] || true; do
  # reclaim finished slots
  for ((g = 0; g < NUM_GPUS; g++)); do
    for ((s = 0; s < JOBS_PER_GPU; s++)); do
      pid="${SLOT_PID[${g}_${s}]:-}"
      if [[ -n "$pid" ]] && ! is_alive "$pid"; then
        SLOT_PID[${g}_${s}]=""
      fi
    done
  done

  # dispatch
  while [[ "$qi" -lt "${#QUEUE[@]}" ]]; do
    slot_info=$(find_slot) || break
    read -r gpu slot <<<"$slot_info"
    IFS=':' read -r model hybrid ds algo <<<"${QUEUE[$qi]}"
    qi=$((qi + 1))
    run_lora_job "$model" "$hybrid" "$ds" "$algo" "$gpu" &
    SLOT_PID[${gpu}_${slot}]=$!
    log "dispatch slot=${gpu}_${slot} ${model} ${hybrid} ${ds} (${qi}/${#QUEUE[@]})"
  done

  # done when queue empty and no live jobs
  live=0
  for ((g = 0; g < NUM_GPUS; g++)); do
    live=$((live + $(gpu_job_count "$g")))
  done
  if [[ "$qi" -ge "${#QUEUE[@]}" && "$live" -eq 0 ]]; then
    break
  fi
  sleep "$POLL_SEC"
done

log "Boosting repair queue complete"
conda run -n openbox_qwen_eval python "$REPO_EVAL/examples/benchmark/check_mltest20_optionB_status.py" 2>/dev/null | tee -a "$MASTER_LOG"
