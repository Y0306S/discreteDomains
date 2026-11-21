#!/bin/bash
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
source /jizhicfs/johnnyslin/anaconda3/bin/activate llamafactory_qwen35
python /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_dl_structural.py \
    --test-data /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data/dl_test.jsonl \
    --base-model /jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base \
    --model-label qwen_base \
    --output-jsonl /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/dl_qwen_base_predictions.jsonl \
    --output-json /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs/dl_qwen_base_structural_eval.json \
    --max-new-tokens 768
