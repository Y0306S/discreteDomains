#!/bin/bash
# End-to-end: train 4 P5 S1 LoRA adapters (from base Qwen), then rerun all mltest20
# Option B evals from scratch (no resume / no skip):
#   - classical baselines smac/tpe/cmaes
#   - gpt-4o-mini P16/H2
#   - LoRA P5/H0 + P5/H4 for s0 + all 4 new S1 models
#
# DOES NOT touch eval8/leaderboard RF runs in real_hpo_v2.
#
# Usage:
#   STAGE=status bash scripts/run_p5_s1_train_then_mltest20_p5h4_eval.sh
#   DRY_RUN=1 bash scripts/run_p5_s1_train_then_mltest20_p5h4_eval.sh
#   bash scripts/run_p5_s1_train_then_mltest20_p5h4_eval.sh
#   STAGE=sft-only|eval-only|all
#   EVAL_RESUME=1 bash ...    # skip completed eval variants (--resume); no dir wipe
#   CLEAR_OUTPUT=0 bash ...   # keep existing output dirs
#   NUM_GPUS=4 bash ...       # 4× A800; SFT uses TRAIN_MODE=ddp2 (2-GPU DDP per job)
#   TRAIN_MODE=ddp2|single4   # ddp2 = batch ep2p5 then ep7p5 (recommended)
#
set -euo pipefail

REPO_SFT="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune"
REPO_EVAL="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune/p5_s1_pipeline}"
STAGE="${STAGE:-all}"          # all | sft-only | eval-only | status
DRY_RUN="${DRY_RUN:-0}"
EVAL_RESUME="${EVAL_RESUME:-0}"
CLEAR_OUTPUT="${CLEAR_OUTPUT:-$([[ "$EVAL_RESUME" == "1" ]] && echo 0 || echo 1)}"
VISIBLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
NUM_GPUS="${NUM_GPUS:-$VISIBLE_GPUS}"
if [[ "$NUM_GPUS" -gt 4 ]]; then NUM_GPUS=4; fi
if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_MODE="${TRAIN_MODE:-ddp2}"

mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/pipeline_${TS}.log"

# --- SFT (4 jobs) ---
SFT_CONFIGS=(
  llamafactory_qwen35_9b_lora_p5_s1a_ep2p5.yaml
  llamafactory_qwen35_9b_lora_p5_s1b_ep2p5.yaml
  llamafactory_qwen35_9b_lora_p5_s1a_ep7p5.yaml
  llamafactory_qwen35_9b_lora_p5_s1b_ep7p5.yaml
)
declare -A SFT_WEIGHTS=(
  [llamafactory_qwen35_9b_lora_p5_s1a_ep2p5.yaml]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep2p5"
  [llamafactory_qwen35_9b_lora_p5_s1b_ep2p5.yaml]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5"
  [llamafactory_qwen35_9b_lora_p5_s1a_ep7p5.yaml]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [llamafactory_qwen35_9b_lora_p5_s1b_ep7p5.yaml]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5"
)

# --- Eval: mltest20 Option B ---
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
PYTHON_CLASSICAL="${PYTHON_CLASSICAL:-/root/miniconda3/envs/openbox_eval/bin/python}"
PYTHON_GPT="${PYTHON_GPT:-/root/miniconda3/envs/openbox_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algoB}"
CLASSICAL_PARALLEL="${CLASSICAL_PARALLEL:-3}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2/logs}"

LORA_MODELS=(s0 s1a_ep2p5 s1b_ep2p5 s1a_ep7p5 s1b_ep7p5)
HYBRIDS=(H0 H4)
CLASSICAL_OPTS=(smac tpe cmaes)

declare -A ADAPTER_FOR=(
  [s0]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0"
  [s1a_ep2p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep2p5"
  [s1b_ep2p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5"
  [s1a_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1b_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5"
)
# max_new_tokens: see experiment_utils (json_only measured max=228 → 768; rationale max=352 → 768)
JSON_ONLY_MAXTOK="${JSON_ONLY_MAXTOK:-768}"
RATIONALE_MAXTOK="${RATIONALE_MAXTOK:-768}"
declare -A MAXTOK_FOR=( [s0]="$JSON_ONLY_MAXTOK" [s1a_ep2p5]="$JSON_ONLY_MAXTOK" [s1a_ep7p5]="$JSON_ONLY_MAXTOK" [s1b_ep2p5]="$RATIONALE_MAXTOK" [s1b_ep7p5]="$RATIONALE_MAXTOK" )
declare -A RESPFMT_FOR=( [s0]=json_only [s1a_ep2p5]=json_only [s1a_ep7p5]=json_only [s1b_ep2p5]=rationale_json [s1b_ep7p5]=rationale_json )

# dataset:algorithm (Option B)
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

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

status_check() {
  cd "$REPO_EVAL"
  "$PYTHON_LLM" examples/benchmark/check_mltest20_optionB_status.py \
    --n-trials "$N_TRIALS" \
    --output-root "$OUTPUT_ROOT"
}

print_plan() {
  local n_classical=$(( ${#CLASSICAL_OPTS[@]} * ${#TASKS[@]} ))
  local n_p16=${#TASKS[@]}
  local n_lora=$(( ${#LORA_MODELS[@]} * ${#HYBRIDS[@]} * ${#TASKS[@]} ))

  log "=== P5 S1 pipeline plan (DRY_RUN=$DRY_RUN STAGE=$STAGE NUM_GPUS=$NUM_GPUS EVAL_RESUME=$EVAL_RESUME CLEAR_OUTPUT=$CLEAR_OUTPUT) ==="
  log ""
  log "PHASE 1 — SFT (TRAIN_MODE=$TRAIN_MODE): 2-GPU DDP per job, batch ep2p5 then ep7p5"
  for cfg in "${SFT_CONFIGS[@]}"; do
    log "  $cfg -> ${SFT_WEIGHTS[$cfg]}"
  done
  log ""
  log "PHASE 2 — mltest20 Option B (NOT eval8 RF leaderboard)"
  log "PHASE 2 — mltest20 Option B after all adapters ready (up to $((NUM_GPUS * 2)) LoRA eval slots)"
  log "  Jobs: classical=$n_classical  P16/H2=$n_p16  LoRA P5/H0+H4=$n_lora"
  log "  classical + P16 -> $OUTPUT_ROOT"
  for model in "${LORA_MODELS[@]}"; do
    log "  $model -> ${OUTPUT_ROOT}_p5_${model} (fmt=${RESPFMT_FOR[$model]} max_tok=${MAXTOK_FOR[$model]})"
  done
  if [[ "$STAGE" == "status" ]]; then
    log ""
    status_check | tee -a "$MASTER_LOG"
  fi
}

clear_eval_outputs() {
  if [[ "$CLEAR_OUTPUT" != "1" ]]; then
    log "CLEAR_OUTPUT=0 — keeping existing benchmark output dirs"
    return 0
  fi
  log "Clearing mltest20 Option B output dirs for fresh rerun"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY_RUN] would remove $REPO_EVAL/$OUTPUT_ROOT and ${OUTPUT_ROOT}_p5_*"
    return 0
  fi
  cd "$REPO_EVAL"
  rm -rf "$OUTPUT_ROOT"
  for model in "${LORA_MODELS[@]}"; do
    rm -rf "${OUTPUT_ROOT}_p5_${model}"
  done
  log "Output dirs cleared"
}

RESUME_FLAG=()
if [[ "$EVAL_RESUME" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

run_sft() {
  log "PHASE 1: starting SFT sweep (TRAIN_MODE=$TRAIN_MODE NUM_GPUS=$NUM_GPUS)"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY_RUN] would run: TRAIN_MODE=$TRAIN_MODE NUM_GPUS=$NUM_GPUS bash $REPO_SFT/scripts/run_p5_s1_epoch_sweep.sh"
    return 0
  fi
  TRAIN_MODE="$TRAIN_MODE" NUM_GPUS="$NUM_GPUS" bash "$REPO_SFT/scripts/run_p5_s1_epoch_sweep.sh" >>"$LOG_ROOT/sft_sweep_${TS}.log" 2>&1
  for cfg in "${SFT_CONFIGS[@]}"; do
    w="${SFT_WEIGHTS[$cfg]}/adapter_model.safetensors"
    [[ -f "$w" ]] || { log "ERROR: missing $w"; exit 1; }
  done
  log "PHASE 1: all 4 adapters present"
}

run_classical_job() {
  local opt="$1" ds="$2" algo="$3"
  local log="$LOG_ROOT/eval_classical_${opt}_${ds}_${TS}.log"
  log "CLASSICAL $opt ds=$ds algo=$algo -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  cd "$REPO_EVAL"
  export PYTHONPATH="$REPO_EVAL/examples/benchmark:${REPO_EVAL}${PYTHONPATH:+:$PYTHONPATH}"
  "$PYTHON_CLASSICAL" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --skip-test-eval \
    --optimizer "$opt" \
    --output-dir "$OUTPUT_ROOT" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    "${RESUME_FLAG[@]}" >>"$log" 2>&1
}

run_p16_job() {
  local ds="$1" algo="$2"
  local hist="history_data/mltest20_algoB_p16_h2"
  local log="$LOG_ROOT/eval_p16_${ds}_${TS}.log"
  log "P16/H2 gpt ds=$ds algo=$algo -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  cd "$REPO_EVAL"
  export PYTHONPATH="$REPO_EVAL/examples/benchmark:${REPO_EVAL}${PYTHONPATH:+:$PYTHONPATH}"
  "$PYTHON_GPT" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --skip-test-eval \
    --llm-backends gpt4o-mini \
    --output-dir "$OUTPUT_ROOT" \
    --history-dir "$hist" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer ptllm_tpe \
    --prompt-variant P16 \
    --hybrid-variant H2 \
    "${RESUME_FLAG[@]}" >>"$log" 2>&1
}

run_lora_job() {
  local model="$1" hybrid="$2" ds="$3" algo="$4" gpu="$5"
  local adapter="${ADAPTER_FOR[$model]}"
  local out="${OUTPUT_ROOT}_p5_${model}"
  local max_tok="${MAXTOK_FOR[$model]}"
  local resp_fmt="${RESPFMT_FOR[$model]}"
  local log="$LOG_ROOT/eval_${model}_${hybrid}_${ds}_${TS}.log"
  log "LORA model=$model $hybrid ds=$ds algo=$algo gpu=$gpu -> $log"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
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
    --optimizer ptllm_tpe_qwen \
    --prompt-variant P5 \
    --hybrid-variant "$hybrid" \
    "${RESUME_FLAG[@]}" >>"$log" 2>&1
}

gpu_free_mb() {
  local idx="$1"
  nvidia-smi -i "$idx" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

can_pack() {
  local idx="$1"
  local free
  free=$(gpu_free_mb "$idx")
  [[ "$free" -ge 28000 ]]
}

run_evals() {
  log "PHASE 2: mltest20 Option B fresh rerun"
  mkdir -p "$LOGDIR"
  clear_eval_outputs

  if [[ "$DRY_RUN" != "1" ]]; then
    for model in "${LORA_MODELS[@]}"; do
      adapter="${ADAPTER_FOR[$model]}"
      if [[ ! -f "$adapter/adapter_model.safetensors" ]]; then
        log "ERROR: missing adapter for $model: $adapter/adapter_model.safetensors"
        exit 1
      fi
    done
  else
    log "[DRY_RUN] skipping adapter presence check (SFT may not have run yet)"
  fi

  # --- classical (24 jobs) ---
  classical_jobs=()
  for opt in "${CLASSICAL_OPTS[@]}"; do
    for T in "${TASKS[@]}"; do
      IFS=':' read -r DS ALGO <<<"$T"
      classical_jobs+=("$opt|$DS|$ALGO")
    done
  done
  log "Running ${#classical_jobs[@]} classical baseline jobs (parallel=$CLASSICAL_PARALLEL)"
  if [[ "$DRY_RUN" != "1" ]]; then
    for job in "${classical_jobs[@]}"; do
      IFS='|' read -r OPT DS ALGO <<<"$job"
      while [[ $(jobs -rp | wc -l) -ge $CLASSICAL_PARALLEL ]]; do
        sleep 5
      done
      run_classical_job "$OPT" "$DS" "$ALGO" &
    done
    wait
  fi

  # --- P16/H2 (8 jobs) ---
  p16_jobs=()
  for T in "${TASKS[@]}"; do
    IFS=':' read -r DS ALGO <<<"$T"
    p16_jobs+=("$DS|$ALGO")
  done
  # gpt-4o-mini uses LLMCONFIGS.OPENAI_KEY from mindware/modules/configs.py
  # (same as run_option2_v2.sh / eval8 runs — no OPENAI_API_KEY env required)
  log "Running ${#p16_jobs[@]} P16/H2 gpt jobs"
  for job in "${p16_jobs[@]}"; do
    IFS='|' read -r DS ALGO <<<"$job"
    run_p16_job "$DS" "$ALGO"
  done

  # --- LoRA P5/H0+H4 (80 jobs) ---
  lora_jobs=()
  for model in "${LORA_MODELS[@]}"; do
    for hybrid in "${HYBRIDS[@]}"; do
      for T in "${TASKS[@]}"; do
        IFS=':' read -r DS ALGO <<<"$T"
        lora_jobs+=("$model|$hybrid|$DS|$ALGO")
      done
    done
  done

  log "Running ${#lora_jobs[@]} LoRA jobs (${NUM_GPUS} GPU(s), pack 2/GPU when free>=28GB)"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY_RUN] would run ${#lora_jobs[@]} LoRA jobs"
    return 0
  fi

  lora_slots=()
  for ((g = 0; g < NUM_GPUS; g++)); do
    lora_slots+=("${g}a" "${g}b")
  done

  declare -A SLOT_PID=()
  declare -A SLOT_GPU=()
  for slot in "${lora_slots[@]}"; do
    SLOT_PID[$slot]=
    SLOT_GPU[$slot]="${slot%[ab]}"
  done

  queue_idx=0
  total=${#lora_jobs[@]}

  is_alive() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
  }

  any_slot_alive() {
    local slot
    for slot in "${lora_slots[@]}"; do
      is_alive "${SLOT_PID[$slot]:-}" && return 0
    done
    return 1
  }

  while [[ $queue_idx -lt $total ]] || any_slot_alive; do
    for slot in "${lora_slots[@]}"; do
      pid="${SLOT_PID[$slot]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[$slot]=
      if [[ $queue_idx -ge $total ]]; then continue; fi
      gpu="${SLOT_GPU[$slot]}"
      if [[ "$slot" == *b ]] && ! can_pack "$gpu"; then continue; fi
      job="${lora_jobs[$queue_idx]}"
      queue_idx=$((queue_idx + 1))
      IFS='|' read -r MODEL HYBRID DS ALGO <<<"$job"
      log "dispatch slot=$slot gpu=$gpu $MODEL $HYBRID $DS ($queue_idx/$total)"
      (
        run_lora_job "$MODEL" "$HYBRID" "$DS" "$ALGO" "$gpu"
      ) &
      SLOT_PID[$slot]=$!
      sleep 90
    done
    sleep 30
  done
  wait

  log "PHASE 2: eval queue complete"
  status_check | tee -a "$MASTER_LOG"
}

case "$STAGE" in
  status)
    print_plan
    ;;
  all)
    print_plan
    run_sft
    run_evals
    ;;
  sft-only)
    print_plan
    run_sft
    ;;
  eval-only)
    print_plan
    run_evals
    ;;
  *)
    echo "Unknown STAGE=$STAGE (all | sft-only | eval-only | status)" >&2
    exit 1
    ;;
esac

log "Pipeline finished."
