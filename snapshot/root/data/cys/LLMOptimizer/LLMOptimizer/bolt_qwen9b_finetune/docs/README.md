# BOLT-Style Qwen3.5 9B Warm-Start Fine-Tuning

This folder contains the successful LLaMA-Factory LoRA fine-tuning path for
Qwen3.5-9B on BOLT-style AutoML warm-start data mined from
`/jizhicfs/pkuhetu/bcxu/llmoptimizer_benchmark`.

The target behavior is warm-start recommendation only: one best configuration
per dataset and algorithm. HPO trajectories are not used as training sequences.

## Data

```bash
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune
python scripts/data/build_bolt_ml_sft_data.py
python scripts/data/validate_sft_data.py
```

Current generated split:

- `data/full.jsonl`: 658 examples
- `data/train.jsonl`: 526 examples
- `data/validation.jsonl`: 132 examples

Nearest-neighbor models/signals are excluded everywhere.

## Successful Training Run

Environment:

```bash
source /jizhicfs/lhy/activate
conda activate llamafactory_qwen35_repo
```

Run:

```bash
cd /jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune
CUDA_VISIBLE_DEVICES=4 ./scripts/train/run_llamafactory_qwen35_base_full.sh \
  per_device_train_batch_size=2 \
  gradient_accumulation_steps=8
```

Successful output:

```text
outputs/llamafactory_qwen35_9b_base_lora_sft
```

Final metrics from the completed run:

- train steps: 99
- epochs: 3
- train loss: 0.6265
- eval loss: 0.5696
- runtime: 12:00

The exact config used by the run is:

```text
configs/ml/llamafactory_qwen35_9b_base_lora_sft.yaml
```

## Benchmark History Regeneration

Classification:

```bash
cd /jizhicfs/pkuhetu/bcxu/llmoptimizer_benchmark
python examples/benchmark/run_hpo_history_example.py \
  --task_type cls \
  --dataset ada \
  --algorithm lightgbm \
  --metric bal_acc \
  --epoch 30 \
  --history_dir ./history_data/history_data
```

Regression:

```bash
cd /jizhicfs/pkuhetu/bcxu/llmoptimizer_benchmark
python examples/benchmark/run_hpo_history_example.py \
  --task_type rgs \
  --dataset abalone \
  --algorithm lightgbm \
  --metric mse \
  --epoch 30 \
  --history_dir ./history_data/history_data
```

## Evaluate One Config

This wrapper requires the full `llmoptimizer_benchmark` runtime dependencies,
including `openbox`:

```bash
python scripts/eval/evaluate_config_cli.py \
  --task-type cls \
  --dataset ada \
  --algorithm lightgbm \
  --hyperparameters '{"lightgbm:learning_rate": 0.1, "lightgbm:n_estimators": 500}'
```

## Compare Gemini vs Fine-Tuned Qwen

The canonical algorithm and hyperparameter schema is in:

```text
configs/ml/ml_model_schema.json
```

Run a local dry-run:

```bash
python scripts/eval/compare_gemini_qwen_eval.py --dry-run --limit 2
```

Run Qwen only:

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/eval/compare_gemini_qwen_eval.py \
  --models qwen \
  --limit 10
```

Run Gemini and Qwen:

```bash
export GEMINI_API_KEY=...
CUDA_VISIBLE_DEVICES=4 python scripts/eval/compare_gemini_qwen_eval.py \
  --models qwen gemini \
  --gemini-model gemini-3.1-flash \
  --limit 20
```

Outputs:

```text
eval_comparison/gemini_vs_qwen_validation.jsonl
eval_comparison/gemini_vs_qwen_validation.summary.json
```

The comparison normalizes returned configs to canonical algorithm-prefixed
keys and reports parse/schema/key/value similarity to the held-out historical
best config. To measure true downstream HPO quality, evaluate selected
predicted configs with `evaluate_config_cli.py` or the benchmark runtime.
