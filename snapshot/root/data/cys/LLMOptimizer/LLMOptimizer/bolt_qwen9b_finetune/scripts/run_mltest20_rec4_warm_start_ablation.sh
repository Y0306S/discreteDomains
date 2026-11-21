#!/usr/bin/env bash
# Warm-start ablation: trial-1 = recommended4 best-of-3; trials 2-80 normal hybrid.
#
# Arms:
#   A) s1b_ep7p5 + ptllm_smac P5/H0
#   B) s1b_ep2p5 + ptllm_tpe  P5/H4
#   C) s1b_ep7p5 + ptllm_tpe  P5/H4
#
# Usage:
#   PRESETS=.../presets_best_of_3.json bash scripts/run_mltest20_rec4_warm_start_ablation.sh
set -euo pipefail

REPO_EVAL="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune/p5_s1_pipeline}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOG_ROOT/warm_start_rec4_ablation_${TS}.log"

PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
EVAL_RESUME="${EVAL_RESUME:-1}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-28000}"
POLL_SEC="${POLL_SEC:-15}"
RATIONALE_MAXTOK="${RATIONALE_MAXTOK:-768}"

PRESETS="${PRESETS:-$REPO_EVAL/benchmark_data/real_hpo_v2_mltest20_algoB/warm_start_rec4/presets_best_of_3.json}"

declare -A ADAPTER_FOR=(
  [s1b_ep2p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5"
  [s1b_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5"
)

# arm = model|optimizer|hybrid
ARMS=(
  "s1b_ep7p5|ptllm_smac_qwen|H0"
  "s1b_ep2p5|ptllm_tpe_qwen|H4"
  "s1b_ep7p5|ptllm_tpe_qwen|H4"
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

is_alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

run_job() {
  local model="$1" opt="$2" hybrid="$3" ds="$4" algo="$5" gpu="$6"
  local adapter="${ADAPTER_FOR[$model]}"
  local out="benchmark_data/real_hpo_v2_mltest20_algoB_p5_${model}"
  local logf="$LOG_ROOT/warm_start_${model}_${opt}_${hybrid}_${ds}_${TS}.log"
  log "WARM gpu=$gpu model=$model $opt P5/$hybrid ds=$ds algo=$algo -> $logf"
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
    --llm-max-tokens "$RATIONALE_MAXTOK" \
    --llm-response-format rationale_json \
    --llm-n-initial-samples 1 \
    --preset-warm-start-json "$PRESETS" \
    --variant-suffix warm_start \
    --output-dir "$out" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer "$opt" \
    --prompt-variant P5 \
    --hybrid-variant "$hybrid" \
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
  [[ "$free" -ge "$PACK_MIN_FREE_MB" ]]
}

mkdir -p "$LOG_ROOT"
[[ -f "$PRESETS" ]] || { log "ERROR: missing presets $PRESETS"; exit 1; }

log "=== warm_start ablation (recommended4 best-of-3 as trial 1) ==="
log "presets=$PRESETS"

for model in s1b_ep2p5 s1b_ep7p5; do
  w="${ADAPTER_FOR[$model]}/adapter_model.safetensors"
  [[ -f "$w" ]] || { log "ERROR: missing $w"; exit 1; }
done

jobs=()
for T in "${TASKS[@]}"; do
  IFS=':' read -r DS ALGO <<<"$T"
  for arm in "${ARMS[@]}"; do
    IFS='|' read -r MODEL OPT HYBRID <<<"$arm"
    jobs+=("$MODEL|$OPT|$HYBRID|$DS|$ALGO")
  done
done

log "Queue: ${#jobs[@]} jobs (3 arms × 8 datasets)"

declare -A SLOT_PID=()
queue_idx=0
total=${#jobs[@]}

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
      job="${jobs[$queue_idx]}"
      queue_idx=$((queue_idx + 1))
      IFS='|' read -r MODEL OPT HYBRID DS ALGO <<<"$job"
      log "dispatch gpu=$gpu slot=$slot $MODEL $OPT $HYBRID $DS ($queue_idx/$total)"
      ( run_job "$MODEL" "$OPT" "$HYBRID" "$DS" "$ALGO" "$gpu" ) &
      SLOT_PID[${gpu}_${slot}]=$!
    done
  done
  sleep "$POLL_SEC"
done
wait

log "=== warm_start ablation queue complete ==="
