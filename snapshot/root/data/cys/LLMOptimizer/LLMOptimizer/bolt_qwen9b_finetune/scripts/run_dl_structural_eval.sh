#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35

EVAL_SCRIPT="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_dl_structural.py"
TEST_DATA="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data/dl_test.jsonl"
EVAL_DIR="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs"
OUTPUT_BASE="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/outputs"
BASE_MODEL="/jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base"

mkdir -p "$EVAL_DIR"

# Use a single GPU for inference (generation is memory-light with bf16)
export CUDA_VISIBLE_DEVICES=0

eval_model() {
    local label="$1"
    local adapter_dir="$2"
    local out_jsonl="$EVAL_DIR/dl_${label}_predictions.jsonl"
    local out_json="$EVAL_DIR/dl_${label}_structural_eval.json"

    echo ""
    echo "========================================"
    echo "Evaluating: $label"
    echo "========================================"

    if [ -n "$adapter_dir" ]; then
        python "$EVAL_SCRIPT" \
            --test-data "$TEST_DATA" \
            --base-model "$BASE_MODEL" \
            --adapter-dir "$adapter_dir" \
            --model-label "$label" \
            --output-jsonl "$out_jsonl" \
            --output-json "$out_json"
    else
        python "$EVAL_SCRIPT" \
            --test-data "$TEST_DATA" \
            --base-model "$BASE_MODEL" \
            --model-label "$label" \
            --output-jsonl "$out_jsonl" \
            --output-json "$out_json"
    fi
}

# 1. Base model (no adapter)
eval_model "qwen_base" ""

# 2. LoRA SFT dl
eval_model "qwen_lora_sft_dl" "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_dl"

# 3. LoRA SFT 2 dl
eval_model "qwen_lora_sft_2_dl" "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_2_dl"

# 4. Full SFT 4 dl (load as base model since it's a full fine-tune)
eval_model "qwen_full_sft_4_dl" "$OUTPUT_BASE/llamafactory_qwen35_9b_base_full_sft_4_dl"

# Combine all evaluations into one summary
echo ""
echo "========================================"
echo "Combining all evaluations"
echo "========================================"

python3 -c "
import json
from pathlib import Path

base = Path('$EVAL_DIR')
models = ['qwen_base', 'qwen_lora_sft_dl', 'qwen_lora_sft_2_dl', 'qwen_full_sft_4_dl']

combined = {}
for model in models:
    p = base / f'dl_{model}_structural_eval.json'
    if p.exists():
        data = json.loads(p.read_text())
        combined[model] = {
            'parse_rate': data.get('parse_rate'),
            'valid_family_rate': data.get('valid_family_rate'),
            'family_match_rate': data.get('family_match_rate'),
            'exact_match_rate': data.get('exact_match_rate'),
            'mean_key_overlap': data.get('mean_key_overlap'),
            'mean_numeric_mae': data.get('mean_numeric_mae'),
        }
        print(f'{model}:')
        for k, v in combined[model].items():
            print(f'  {k}: {v}')
        print()

out = base / 'dl_all_structural_eval.json'
out.write_text(json.dumps(combined, indent=2))
print(f'Combined summary written to: {out}')
"

echo ""
echo "========================================"
echo "ALL DL STRUCTURAL EVALUATIONS COMPLETE"
echo "========================================"
