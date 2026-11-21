#!/usr/bin/env bash
# On-policy P5 rollouts with direct holdout-eval rewards (Δbest primary).
set -euo pipefail

BENCHMARK_ROOT="${BENCHMARK_ROOT:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
SCRIPT="${BENCHMARK_ROOT}/examples/benchmark/run_p5_rl_rollout_buffer.py"
CONDA_ENV="${CONDA_ENV:-openbox_qwen_eval}"

# Defaults aligned with mltest20 Option B + s0 json_only
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_ROOT}/benchmark_data/rl_rollout_buffers}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-json_only}"
MAX_TOKENS_JSON_ONLY="${MAX_TOKENS_JSON_ONLY:-768}"
MAX_TOKENS_RATIONALE="${MAX_TOKENS_RATIONALE:-768}"
if [[ "${RESPONSE_FORMAT}" == "rationale_json" ]]; then
  MAX_TOKENS="${MAX_TOKENS:-$MAX_TOKENS_RATIONALE}"
else
  MAX_TOKENS="${MAX_TOKENS:-$MAX_TOKENS_JSON_ONLY}"
fi

TASK_TYPE="${TASK_TYPE:-cls}"
DATASET="${DATASET:-yeast}"
ALGORITHM="${ALGORITHM:-lightgbm}"
MLTEST20_ALL="${MLTEST20_ALL:-0}"

EXTRA_ARGS=()
if [[ "${MLTEST20_ALL}" == "1" ]]; then
  EXTRA_ARGS+=(--mltest20-all)
else
  EXTRA_ARGS+=(--task-type "${TASK_TYPE}" --dataset "${DATASET}" --algorithm "${ALGORITHM}")
fi

echo "[run_p5_rl_rollout_buffer] env=${CONDA_ENV} adapter=${ADAPTER}"
echo "[run_p5_rl_rollout_buffer] output=${OUTPUT_ROOT} trials=${N_TRIALS}"

conda run -n "${CONDA_ENV}" python "${SCRIPT}" \
  "${EXTRA_ARGS[@]}" \
  --prompt-variant P5 \
  --hybrid-variant H0 \
  --qwen-model-path "${BASE_MODEL}" \
  --qwen-adapter-path "${ADAPTER}" \
  --llm-response-format "${RESPONSE_FORMAT}" \
  --llm-max-tokens "${MAX_TOKENS}" \
  --n-trials "${N_TRIALS}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_ROOT}" \
  "$@"
