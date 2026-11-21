#!/usr/bin/env bash
set -euo pipefail

# Run on the approved LLMOptimizer2 allocation; the supervisor rechecks resources.
export PYTHONPATH=/root/data/cys/baseline/envs/bl_bolt_ml/lib/python3.11/site-packages
exec /root/data/cys/envs/openbox_qwen_eval/bin/python \
  /root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains/train_bolt_taskset_sft.py \
  --detach --host LLMOptimizer2 --gpus 1 --cpus 21,22,23
