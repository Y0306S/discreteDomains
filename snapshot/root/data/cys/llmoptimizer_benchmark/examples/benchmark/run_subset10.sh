#!/bin/bash
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PYTHON="${PYTHON:-/root/miniconda3/envs/openbox_eval/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_data/real_hpo_cls}"
N_TRIALS="${N_TRIALS:-10}"
SEED="${SEED:-1}"
LLM_BACKENDS="${LLM_BACKENDS:-gpt4o-mini}"

DATASETS=(
  yeast car christine
  adult nomao PhishingWebsites Fashion-MNIST
  MiniBooNE covertype porto-seguro
)

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

echo "[Subset10] Phase 0 cls benchmark (10 datasets, ${N_TRIALS} trials)"
echo "  python=$PYTHON"
echo "  output_dir=$OUTPUT_DIR"
echo "  datasets=${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
  echo ""
  echo "[Subset10] === $ds ==="
  "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
    --dataset "$ds" \
    --optimizer all \
    --algorithm random_forest \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --metric bal_acc \
    --skip-test-eval \
    --llm-backends "$LLM_BACKENDS" \
    --qwen-backend auto \
    --resume \
    --output-dir "$OUTPUT_DIR"
done

echo ""
echo "[Subset10] Aggregating results + handoff package"
"$PYTHON" examples/benchmark/plot_real_benchmark.py \
  --base-dir "$OUTPUT_DIR" \
  --handoff

echo "[Subset10] Done. Handoff: $OUTPUT_DIR/handoff/"
