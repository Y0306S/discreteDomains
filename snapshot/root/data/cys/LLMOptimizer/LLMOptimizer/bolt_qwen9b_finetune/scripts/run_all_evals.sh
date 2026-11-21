#!/bin/bash
set -e

# Evaluation pipeline for base + LoRA 1/2/3/4 + Full SFT on test set
# Uses OpenBox benchmark evaluator

source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35

EVAL_CLI="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_config_cli.py"
TEST_DATA="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/sft_test_holdout.jsonl"
EVAL_DIR="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs"
OUTPUT_BASE="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/outputs"
BASE_MODEL="/jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$EVAL_DIR"

# Clean up old evaluation files for models we're re-evaluating
echo "=== Cleaning old eval files ==="
rm -f "$EVAL_DIR"/qwen_base_test_predictions.jsonl
rm -f "$EVAL_DIR"/qwen_base_test_openbox_eval.json
rm -f "$EVAL_DIR"/qwen_sft{1,2,3,4}_test_predictions.jsonl
rm -f "$EVAL_DIR"/qwen_sft{1,2,3,4}_test_openbox_eval.json
rm -f "$EVAL_DIR"/qwen_full4_test_predictions.jsonl
rm -f "$EVAL_DIR"/qwen_full4_test_openbox_eval.json
rm -f "$EVAL_DIR"/qwen_all_test_openbox_eval.json
echo "Cleaned."

# Helper function to generate predictions for a model
generate() {
    local label="$1"
    local adapter_dir="$2"
    local out_jsonl="$3"

    echo ""
    echo "========================================"
    echo "Generating predictions: $label"
    echo "========================================"

    if [ -n "$adapter_dir" ]; then
        python "$EVAL_CLI" \
            --generate qwen \
            --dataset-jsonl "$TEST_DATA" \
            --predictions-out "$out_jsonl" \
            --base-model "$BASE_MODEL" \
            --adapter-dir "$adapter_dir" \
            --model-label "$label" \
            --max-new-tokens 768
    else
        python "$EVAL_CLI" \
            --generate qwen \
            --dataset-jsonl "$TEST_DATA" \
            --predictions-out "$out_jsonl" \
            --base-model "$BASE_MODEL" \
            --model-label "$label" \
            --max-new-tokens 768
    fi
}

# Helper function to run OpenBox evaluation
evaluate() {
    local label="$1"
    local pred_jsonl="$2"
    local out_json="$3"

    echo ""
    echo "========================================"
    echo "OpenBox evaluating: $label"
    echo "========================================"

    python "$EVAL_CLI" \
        --dataset-jsonl "$TEST_DATA" \
        --predictions-jsonl "$pred_jsonl" \
        --output-json "$out_json"
}

# 1. Base model (no adapter)
generate "qwen_base" "" "$EVAL_DIR/qwen_base_test_predictions.jsonl"

# 2. LoRA 1
generate "qwen_lora1" "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft" "$EVAL_DIR/qwen_sft1_test_predictions.jsonl"

# 3. LoRA 2
generate "qwen_lora2" "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_2" "$EVAL_DIR/qwen_sft2_test_predictions.jsonl"

# 4. LoRA 3
generate "qwen_lora3" "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_3" "$EVAL_DIR/qwen_sft3_test_predictions.jsonl"

# 5. LoRA 4
generate "qwen_lora4" "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_4" "$EVAL_DIR/qwen_sft4_test_predictions.jsonl"

# 6. Full SFT 4 (checkpoint-45 is a merged full model, load as base)
export FULL_SFT_MODEL="$OUTPUT_BASE/llamafactory_qwen35_9b_base_full_sft_4/checkpoint-45"
python "$EVAL_CLI" \
    --generate qwen \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/qwen_full4_test_predictions.jsonl" \
    --base-model "$FULL_SFT_MODEL" \
    --model-label "qwen_full4" \
    --max-new-tokens 768

# Now run OpenBox evaluation for each
evaluate "qwen_base" "$EVAL_DIR/qwen_base_test_predictions.jsonl" "$EVAL_DIR/qwen_base_test_openbox_eval.json"
evaluate "qwen_lora1" "$EVAL_DIR/qwen_sft1_test_predictions.jsonl" "$EVAL_DIR/qwen_sft1_test_openbox_eval.json"
evaluate "qwen_lora2" "$EVAL_DIR/qwen_sft2_test_predictions.jsonl" "$EVAL_DIR/qwen_sft2_test_openbox_eval.json"
evaluate "qwen_lora3" "$EVAL_DIR/qwen_sft3_test_predictions.jsonl" "$EVAL_DIR/qwen_sft3_test_openbox_eval.json"
evaluate "qwen_lora4" "$EVAL_DIR/qwen_sft4_test_predictions.jsonl" "$EVAL_DIR/qwen_sft4_test_openbox_eval.json"
evaluate "qwen_full4" "$EVAL_DIR/qwen_full4_test_predictions.jsonl" "$EVAL_DIR/qwen_full4_test_openbox_eval.json"

# Combine all into one evaluation file
echo ""
echo "========================================"
echo "Combining all evaluations"
echo "========================================"

python3 -c "
import json, sys
from pathlib import Path

files = [
    'qwen_base_test_openbox_eval.json',
    'qwen_sft1_test_openbox_eval.json',
    'qwen_sft2_test_openbox_eval.json',
    'qwen_sft3_test_openbox_eval.json',
    'qwen_sft4_test_openbox_eval.json',
    'qwen_full4_test_openbox_eval.json',
]

base = Path('$EVAL_DIR')
combined = {'summary': {}, 'results': []}

for f in files:
    p = base / f
    if not p.exists():
        print(f'Missing: {f}', file=sys.stderr)
        continue
    data = json.loads(p.read_text())
    combined['summary'].update(data.get('summary', {}))
    combined['results'].extend(data.get('results', []))

out = base / 'qwen_all_test_openbox_eval.json'
out.write_text(json.dumps(combined, indent=2, sort_keys=True))
print(f'Combined: {out}')
print(json.dumps(combined['summary'], indent=2))
"

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETE"
echo "========================================"
