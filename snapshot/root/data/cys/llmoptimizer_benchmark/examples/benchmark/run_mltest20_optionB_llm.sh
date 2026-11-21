#!/bin/bash
# Option B: LoRA P5/H0+H4 for s0/s1a/s1b (resumable) + optional gpt-4o-mini P16/H2.
# Packs up to 2 LLM jobs per GPU when free memory allows.
#
# Usage:
#   STAGE=llm bash examples/benchmark/run_mltest20_optionB_llm.sh
#   STAGE=p16 bash examples/benchmark/run_mltest20_optionB_llm.sh   # gpt-4o-mini P16/H2 only
#   STAGE=all bash examples/benchmark/run_mltest20_optionB_llm.sh
#
set -euo pipefail
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
PYTHON_GPT="${PYTHON_GPT:-/root/miniconda3/envs/openbox_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
STAGE="${STAGE:-llm}"   # llm | p16 | all
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algoB}"
mkdir -p "$LOGDIR"

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

declare -A ADAPTERS=(
  [s0]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0"
  [s1a]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0_s1a"
  [s1b]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0_s1b"
)
JSON_ONLY_MAXTOK="${JSON_ONLY_MAXTOK:-768}"
RATIONALE_MAXTOK="${RATIONALE_MAXTOK:-768}"
declare -A MAXTOK=( [s0]="$JSON_ONLY_MAXTOK" [s1a]="$JSON_ONLY_MAXTOK" [s1b]="$RATIONALE_MAXTOK" )
declare -A RESPFMT=( [s0]=json_only [s1a]=json_only [s1b]=rationale_json )

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

gpu_free_mb() {
  local idx="$1"
  nvidia-smi -i "$idx" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

# Need ~28GB free to pack another 9B LoRA job safely (KV cache headroom).
can_pack() {
  local idx="$1"
  local free
  free=$(gpu_free_mb "$idx")
  [[ "$free" -ge 28000 ]]
}

run_lora_job() {
  local model="$1" hybrid="$2" gpu="$3" ds="$4" algo="$5"
  local adapter="${ADAPTERS[$model]}"
  local out="${OUTPUT_ROOT}_p5_${model}"
  local max_tok="${MAXTOK[$model]}"
  local resp_fmt="${RESPFMT[$model]}"
  echo "[$(date -Is)] START LoRA model=$model $hybrid ds=$ds algo=$algo gpu=$gpu"
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
    --resume \
    --output-dir "$out" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer ptllm_tpe_qwen \
    --prompt-variant P5 \
    --hybrid-variant "$hybrid"
  echo "[$(date -Is)] DONE LoRA model=$model $hybrid ds=$ds"
}

run_p16_gpt_job() {
  local ds="$1" algo="$2"
  local out="${OUTPUT_ROOT}"
  local hist="history_data/mltest20_algoB_p16_h2"
  echo "[$(date -Is)] START gpt P16/H2 ds=$ds algo=$algo"
  "$PYTHON_GPT" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --skip-test-eval \
    --llm-backends gpt4o-mini \
    --resume \
    --output-dir "$out" \
    --history-dir "$hist" \
    --all-datasets \
    --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer ptllm_tpe \
    --prompt-variant P16 \
    --hybrid-variant H2
  echo "[$(date -Is)] DONE gpt P16/H2 ds=$ds"
}

# Build LoRA work queue: model × hybrid × task
LORAS_QUEUE=()
for model in s0 s1a s1b; do
  for hybrid in H0 H4; do
    for T in "${TASKS[@]}"; do
      IFS=':' read -r DS ALGO <<<"$T"
      LORAS_QUEUE+=("$model|$hybrid|$DS|$ALGO")
    done
  done
done

P16_QUEUE=()
for T in "${TASKS[@]}"; do
  IFS=':' read -r DS ALGO <<<"$T"
  P16_QUEUE+=("$DS|$ALGO")
done

TS=$(date +%Y%m%d_%H%M%S)

run_p16_stage() {
  if ! "$PYTHON_GPT" -c "import os; raise SystemExit(0 if os.environ.get('OPENAI_API_KEY') else 1)"; then
    echo "[P16] OPENAI_API_KEY not set in this environment — skip gpt-4o-mini P16/H2."
    echo "[P16] Export OPENAI_API_KEY and re-run: STAGE=p16 bash $0"
    return 0
  fi
  local log="$LOGDIR/mltest20_algoB_p16_gpt_${TS}.log"
  echo "[P16] ${#P16_QUEUE[@]} gpt-4o-mini P16/H2 jobs -> $log"
  for job in "${P16_QUEUE[@]}"; do
    IFS='|' read -r DS ALGO <<<"$job"
    run_p16_gpt_job "$DS" "$ALGO" >>"$log" 2>&1
  done
}

run_llm_stage() {
  local log="$LOGDIR/mltest20_algoB_lora_${TS}.log"
  echo "[LLM] ${#LORAS_QUEUE[@]} LoRA jobs; packing up to 2/GPU when free>=22GB" | tee "$log"

  # Worker slots: gpu0_a gpu0_b gpu1_a gpu1_b — track PIDs
  declare -A SLOT_PID=([0a]= [0b]= [1a]= [1b]=)
  declare -A SLOT_GPU=([0a]=0 [0b]=0 [1a]=1 [1b]=1)

  queue_idx=0
  total=${#LORAS_QUEUE[@]}

  is_alive() {
    local pid="$1"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
  }

  while [[ $queue_idx -lt $total ]] || is_alive "${SLOT_PID[0a]:-}" || is_alive "${SLOT_PID[0b]:-}" || is_alive "${SLOT_PID[1a]:-}" || is_alive "${SLOT_PID[1b]:-}"; do
    for slot in 0a 0b 1a 1b; do
      pid="${SLOT_PID[$slot]:-}"
      if is_alive "$pid"; then
        continue
      fi
      SLOT_PID[$slot]=
      if [[ $queue_idx -ge $total ]]; then
        continue
      fi
      gpu="${SLOT_GPU[$slot]}"
      # slot *b* is the packed second job — only if enough free mem
      if [[ "$slot" == *b ]]; then
        if ! can_pack "$gpu"; then
          continue
        fi
      fi
      job="${LORAS_QUEUE[$queue_idx]}"
      queue_idx=$((queue_idx + 1))
      IFS='|' read -r MODEL HYBRID DS ALGO <<<"$job"
      echo "[$(date -Is)] dispatch slot=$slot gpu=$gpu $MODEL $HYBRID $DS ($queue_idx/$total)" | tee -a "$log"
      (
        run_lora_job "$MODEL" "$HYBRID" "$gpu" "$DS" "$ALGO"
      ) >>"$log" 2>&1 &
      SLOT_PID[$slot]=$!
      # give model load time before packing second job on same GPU
      sleep 90
    done
    sleep 30
  done
  echo "[LLM] all LoRA jobs done $(date -Is)" | tee -a "$log"
}

case "$STAGE" in
  llm) run_llm_stage ;;
  p16) run_p16_stage ;;
  all)
    run_p16_stage &
    run_llm_stage
    wait
    ;;
  *) echo "STAGE=llm|p16|all" >&2; exit 1 ;;
esac
