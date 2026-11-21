#!/usr/bin/env bash
# Run the Qwen3.5-9B SFT pipeline:
#   1. Full SFT configs sequentially on all 8 GPUs.
#   2. LoRA configs in parallel on 4-GPU partitions; next job takes over GPUs
#      when a running job finishes.
#
# Usage:
#   bash run_sft_pipeline.sh
#   nohup bash run_sft_pipeline.sh > /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/pipeline.log 2>&1 &
#   bash run_sft_pipeline.sh --skip-full
#   bash run_sft_pipeline.sh --skip-lora

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
PIPELINE_LOG="${PROJECT_DIR}/pipeline.log"

mkdir -p "${LOG_DIR}"

# Activate the LLaMA-Factory environment that has DeepSpeed installed.
source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
export PYTHONNOUSERSITE=1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating environment: llamafactory_qwen35" | tee -a "${PIPELINE_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python: $(which python)" | tee -a "${PIPELINE_LOG}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] LLaMA-Factory: $(which llamafactory-cli)" | tee -a "${PIPELINE_LOG}"

# Forward any CLI flags (e.g. --skip-full, --skip-lora) to the Python script.
python "${SCRIPT_DIR}/run_sft_pipeline.py" \
  --log-dir "${LOG_DIR}" \
  "$@" \
  2>&1 | tee -a "${PIPELINE_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline exited with code ${EXIT_CODE}" | tee -a "${PIPELINE_LOG}"
exit ${EXIT_CODE}
