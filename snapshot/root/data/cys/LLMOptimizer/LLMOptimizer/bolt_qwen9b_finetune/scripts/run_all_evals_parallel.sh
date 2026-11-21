#!/bin/bash
set -e

# Parallel evaluation: one model per GPU
# Qwen models on GPUs 0-5, Gemini API on CPU

source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35

EVAL_CLI="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_config_cli.py"
TEST_DATA="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/sft_test_holdout.jsonl"
EVAL_DIR="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs"
OUTPUT_BASE="/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/outputs"
BASE_MODEL="/jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base"
FULL_SFT="$OUTPUT_BASE/llamafactory_qwen35_9b_base_full_sft_4/checkpoint-45"

export PYTHONUNBUFFERED=1
export OPENAI_HK_API_KEY="${OPENAI_HK_API_KEY:?Set OPENAI_HK_API_KEY}"

mkdir -p "$EVAL_DIR"

# Clean up old eval files for models we're re-evaluating
echo "=== Cleaning old eval files ==="
rm -f "$EVAL_DIR"/qwen_base_test_predictions.jsonl
rm -f "$EVAL_DIR"/qwen_base_test_openbox_eval.json
rm -f "$EVAL_DIR"/qwen_sft{1,2,3,4}_test_predictions.jsonl
rm -f "$EVAL_DIR"/qwen_sft{1,2,3,4}_test_openbox_eval.json
rm -f "$EVAL_DIR"/qwen_full4_test_predictions.jsonl
rm -f "$EVAL_DIR"/qwen_full4_test_openbox_eval.json
rm -f "$EVAL_DIR"/gemini_31_flash_test_predictions.jsonl
rm -f "$EVAL_DIR"/gemini_31_flash_test_openbox_eval.json
rm -f "$EVAL_DIR"/qwen_all_test_openbox_eval.json
echo "Cleaned."

# Launch all Qwen generations in parallel on different GPUs

echo "=== Launching parallel Qwen generations ==="

# GPU 0: base
CUDA_VISIBLE_DEVICES=0 python "$EVAL_CLI" \
    --generate qwen \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/qwen_base_test_predictions.jsonl" \
    --base-model "$BASE_MODEL" \
    --model-label "qwen_base" \
    --max-new-tokens 768 > "$EVAL_DIR/gen_base.log" 2>&1 &
PID_BASE=$!

# GPU 1: LoRA 1
CUDA_VISIBLE_DEVICES=1 python "$EVAL_CLI" \
    --generate qwen \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/qwen_sft1_test_predictions.jsonl" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft" \
    --model-label "qwen_lora1" \
    --max-new-tokens 768 > "$EVAL_DIR/gen_lora1.log" 2>&1 &
PID_LORA1=$!

# GPU 2: LoRA 2
CUDA_VISIBLE_DEVICES=2 python "$EVAL_CLI" \
    --generate qwen \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/qwen_sft2_test_predictions.jsonl" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_2" \
    --model-label "qwen_lora2" \
    --max-new-tokens 768 > "$EVAL_DIR/gen_lora2.log" 2>&1 &
PID_LORA2=$!

# GPU 3: LoRA 3
CUDA_VISIBLE_DEVICES=3 python "$EVAL_CLI" \
    --generate qwen \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/qwen_sft3_test_predictions.jsonl" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_3" \
    --model-label "qwen_lora3" \
    --max-new-tokens 768 > "$EVAL_DIR/gen_lora3.log" 2>&1 &
PID_LORA3=$!

# GPU 4: LoRA 4
CUDA_VISIBLE_DEVICES=4 python "$EVAL_CLI" \
    --generate qwen \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/qwen_sft4_test_predictions.jsonl" \
    --base-model "$BASE_MODEL" \
    --adapter-dir "$OUTPUT_BASE/llamafactory_qwen35_9b_base_lora_sft_4" \
    --model-label "qwen_lora4" \
    --max-new-tokens 768 > "$EVAL_DIR/gen_lora4.log" 2>&1 &
PID_LORA4=$!

# GPU 5: Full SFT 4
CUDA_VISIBLE_DEVICES=5 python "$EVAL_CLI" \
    --generate qwen \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/qwen_full4_test_predictions.jsonl" \
    --base-model "$FULL_SFT" \
    --model-label "qwen_full4" \
    --max-new-tokens 768 > "$EVAL_DIR/gen_full4.log" 2>&1 &
PID_FULL4=$!

# CPU: Gemini 3.1 Flash (API-based, no GPU needed)
python "$EVAL_CLI" \
    --generate openai \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-out "$EVAL_DIR/gemini_31_flash_test_predictions.jsonl" \
    --api-model "gemini-3.1-flash" \
    --api-base-url "https://api.openai-hk.com/v1" \
    --api-key "$OPENAI_HK_API_KEY" \
    --model-label "gemini_3.1_flash" \
    --max-new-tokens 768 > "$EVAL_DIR/gen_gemini.log" 2>&1 &
PID_GEMINI=$!

echo "Launched all 7 generation jobs in parallel:"
echo "  GPU 0: base     (PID $PID_BASE)"
echo "  GPU 1: lora1    (PID $PID_LORA1)"
echo "  GPU 2: lora2    (PID $PID_LORA2)"
echo "  GPU 3: lora3    (PID $PID_LORA3)"
echo "  GPU 4: lora4    (PID $PID_LORA4)"
echo "  GPU 5: full4    (PID $PID_FULL4)"
echo "  CPU:   gemini   (PID $PID_GEMINI)"
echo ""
echo "Waiting for all generations to complete..."
wait $PID_BASE
wait $PID_LORA1
wait $PID_LORA2
wait $PID_LORA3
wait $PID_LORA4
wait $PID_FULL4
wait $PID_GEMINI

echo ""
echo "=== All generations complete ==="
echo "Prediction files:"
for f in "$EVAL_DIR"/qwen_*_test_predictions.jsonl "$EVAL_DIR"/gemini_*_test_predictions.jsonl; do
    [ -f "$f" ] && echo "  $(basename $f): $(wc -l < $f) lines"
done

# Now run OpenBox evaluations in parallel
echo ""
echo "=== Launching parallel OpenBox evaluations ==="

python "$EVAL_CLI" \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-jsonl "$EVAL_DIR/qwen_base_test_predictions.jsonl" \
    --output-json "$EVAL_DIR/qwen_base_test_openbox_eval.json" > "$EVAL_DIR/eval_base.log" 2>&1 &
PID_EBASE=$!

python "$EVAL_CLI" \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-jsonl "$EVAL_DIR/qwen_sft1_test_predictions.jsonl" \
    --output-json "$EVAL_DIR/qwen_sft1_test_openbox_eval.json" > "$EVAL_DIR/eval_lora1.log" 2>&1 &
PID_EL1=$!

python "$EVAL_CLI" \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-jsonl "$EVAL_DIR/qwen_sft2_test_predictions.jsonl" \
    --output-json "$EVAL_DIR/qwen_sft2_test_openbox_eval.json" > "$EVAL_DIR/eval_lora2.log" 2>&1 &
PID_EL2=$!

python "$EVAL_CLI" \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-jsonl "$EVAL_DIR/qwen_sft3_test_predictions.jsonl" \
    --output-json "$EVAL_DIR/qwen_sft3_test_openbox_eval.json" > "$EVAL_DIR/eval_lora3.log" 2>&1 &
PID_EL3=$!

python "$EVAL_CLI" \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-jsonl "$EVAL_DIR/qwen_sft4_test_predictions.jsonl" \
    --output-json "$EVAL_DIR/qwen_sft4_test_openbox_eval.json" > "$EVAL_DIR/eval_lora4.log" 2>&1 &
PID_EL4=$!

python "$EVAL_CLI" \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-jsonl "$EVAL_DIR/qwen_full4_test_predictions.jsonl" \
    --output-json "$EVAL_DIR/qwen_full4_test_openbox_eval.json" > "$EVAL_DIR/eval_full4.log" 2>&1 &
PID_EF4=$!

python "$EVAL_CLI" \
    --dataset-jsonl "$TEST_DATA" \
    --predictions-jsonl "$EVAL_DIR/gemini_31_flash_test_predictions.jsonl" \
    --output-json "$EVAL_DIR/gemini_31_flash_test_openbox_eval.json" > "$EVAL_DIR/eval_gemini.log" 2>&1 &
PID_EGEM=$!

echo "Waiting for all OpenBox evaluations..."
wait $PID_EBASE
wait $PID_EL1
wait $PID_EL2
wait $PID_EL3
wait $PID_EL4
wait $PID_EF4
wait $PID_EGEM

echo ""
echo "=== Combining all evaluations ==="
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
    'gemini_31_flash_test_openbox_eval.json',
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
print('')
print(json.dumps(combined['summary'], indent=2))
"

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETE"
echo "========================================"
