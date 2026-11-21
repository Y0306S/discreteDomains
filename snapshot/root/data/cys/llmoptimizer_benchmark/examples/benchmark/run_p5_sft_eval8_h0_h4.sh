#!/bin/bash
# Eval fine-tuned P5 LoRA adapters (S1a / S1b) on eval8 with P5/H0 and P5/H4.
#
# Does NOT overwrite original gpt-4o-mini real_hpo_v2 runs — writes to a new OUTPUT_DIR.
# Note: train data overlaps all 8 eval datasets (~6.5% of S1 rows); treat as smoke, not clean OOD.
#
# Setup / dry-run (no real LLM calls):
#   DRY_RUN=1 bash examples/benchmark/run_p5_sft_eval8_h0_h4.sh
#
# Launch one model + one hybrid:
#   MODEL=s1a HYBRID=H0 bash examples/benchmark/run_p5_sft_eval8_h0_h4.sh
#
# Full matrix (S1a+S1b × H0+H4) on eval8:
#   STAGE=all bash examples/benchmark/run_p5_sft_eval8_h0_h4.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

# Dedicated eval env (CUDA + peft + mindware). Do not use llamafactory_qwen35 for this.
PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
if ! "$PYTHON" -c "import peft, torch, mindware" 2>/dev/null; then
  echo "[error] PYTHON=$PYTHON missing peft/torch/mindware. Install deps into openbox_qwen_eval or set PYTHON=..." >&2
  exit 1
fi
echo "[env] PYTHON=$PYTHON cuda=$("$PYTHON" -c 'import torch; print(torch.cuda.is_available())')"

N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
STAGE="${STAGE:-one}"          # one | all
MODEL="${MODEL:-s1a}"          # s1a | s1b
HYBRID="${HYBRID:-H0}"         # H0 | H4
PROMPT="${PROMPT:-P5}"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
ALGORITHM="${ALGORITHM:-random_forest}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_S1A="${ADAPTER_S1A:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0_s1a}"
ADAPTER_S1B="${ADAPTER_S1B:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0_s1b}"

case "$MODEL" in
  # S1a = config JSON only; S1b = Rationale + JSON. Cap generation to cut eval wall time.
  s1a) ADAPTER_PATH="$ADAPTER_S1A"; MODEL_TAG="s1a"; LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-768}"; LLM_RESPONSE_FORMAT="${LLM_RESPONSE_FORMAT:-json_only}" ;;
  s1b) ADAPTER_PATH="$ADAPTER_S1B"; MODEL_TAG="s1b"; LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-768}"; LLM_RESPONSE_FORMAT="${LLM_RESPONSE_FORMAT:-rationale_json}" ;;
  *) echo "Unknown MODEL=$MODEL (s1a|s1b)" >&2; exit 1 ;;
esac

if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER_PATH/adapter_model.safetensors" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/p5_sft_eval8_dryrun}"
  N_TRIALS=4
  TIME_LIMIT=300
  EVAL_DATASETS="yeast,boston"
  echo "[DRY_RUN] mock LLM | datasets=$EVAL_DATASETS n_trials=$N_TRIALS"
else
  OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_v2_p5_sft_${MODEL_TAG}}"
  EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"
  echo "[PROD] model=$MODEL_TAG adapter=$ADAPTER_PATH"
  echo "[PROD] prompt=$PROMPT hybrids will run as configured | datasets=$EVAL_DATASETS"
  echo "[PROD] output=$OUTPUT_DIR (separate from gpt-4o-mini real_hpo_v2)"
fi

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES

BASE_RUNNER=( "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py
  --algorithm "$ALGORITHM"
  --n-trials "$N_TRIALS"
  --seed "$SEED"
  --skip-test-eval
  --llm-backends qwen-sft6
  --qwen-backend hf
  --qwen-model-path "$BASE_MODEL"
  --qwen-adapter-path "$ADAPTER_PATH"
  --llm-max-tokens "$LLM_MAX_TOKENS"
  --llm-response-format "$LLM_RESPONSE_FORMAT"
  --resume
  --output-dir "$OUTPUT_DIR"
  --all-datasets
  --include-datasets "$EVAL_DATASETS"
  --time-limit "$TIME_LIMIT"
  --dataset-retries "$DATASET_RETRIES"
  --prompt-variant "$PROMPT"
)

if [[ "$DRY_RUN" == "1" ]]; then
  BASE_RUNNER+=( --llm-mock --llm-n-initial-samples 1 )
fi

run_one() {
  local hybrid="$1"
  local model_tag="$2"
  local adapter="$3"
  echo ""
  echo "=== ptllm_tpe_qwen | prompt=$PROMPT hybrid=$hybrid | model=$model_tag ==="
  echo "    adapter=$adapter"
  echo "    output=$OUTPUT_DIR"
  "${BASE_RUNNER[@]}" \
    --qwen-adapter-path "$adapter" \
    --optimizer ptllm_tpe_qwen \
    --hybrid-variant "$hybrid"
}

run_model_matrix() {
  local model_tag="$1"
  local adapter="$2"
  # refresh OUTPUT_DIR / ADAPTER for this model when STAGE=all
  OUTPUT_DIR="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_p5_sft_${model_tag}}"
  if [[ "$DRY_RUN" == "1" ]]; then
    OUTPUT_DIR="benchmark_data/p5_sft_eval8_dryrun_${model_tag}"
  fi
  case "$model_tag" in
    s1a) LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-768}"; LLM_RESPONSE_FORMAT="${LLM_RESPONSE_FORMAT:-json_only}" ;;
    s1b) LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-768}"; LLM_RESPONSE_FORMAT="${LLM_RESPONSE_FORMAT:-rationale_json}" ;;
  esac
  BASE_RUNNER=( "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$ALGORITHM"
    --n-trials "$N_TRIALS"
    --seed "$SEED"
    --skip-test-eval
    --llm-backends qwen-sft6
    --qwen-backend hf
    --qwen-model-path "$BASE_MODEL"
    --qwen-adapter-path "$adapter"
    --llm-max-tokens "$LLM_MAX_TOKENS"
    --llm-response-format "$LLM_RESPONSE_FORMAT"
    --resume
    --output-dir "$OUTPUT_DIR"
    --all-datasets
    --include-datasets "$EVAL_DATASETS"
    --time-limit "$TIME_LIMIT"
    --dataset-retries "$DATASET_RETRIES"
    --prompt-variant "$PROMPT"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    BASE_RUNNER+=( --llm-mock --llm-n-initial-samples 1 )
  fi
  for H in H0 H4; do
    run_one "$H" "$model_tag" "$adapter"
  done
}

case "$STAGE" in
  one)
    run_one "$HYBRID" "$MODEL_TAG" "$ADAPTER_PATH"
    ;;
  all)
    if [[ "$DRY_RUN" != "1" ]]; then
      EVAL_DATASETS="${EVAL_DATASETS:-yeast,car,PhishingWebsites,segment,boston,tecator,socmob,pol}"
    fi
    run_model_matrix s1a "$ADAPTER_S1A"
    run_model_matrix s1b "$ADAPTER_S1B"
    ;;
  *)
    echo "Unknown STAGE=$STAGE (one | all)" >&2
    exit 1
    ;;
esac

echo ""
echo "[Setup/Run done] STAGE=$STAGE"
echo "  Prompt=$PROMPT  (H0=mix50, H4=phase20_mix50)"
echo "  Datasets=$EVAL_DATASETS"
echo "  Note: eval8 overlaps SFT train (all 8 datasets)."
