#!/usr/bin/env bash
# Launch Phase A (top-3 strict4) and Phase B (v13 full sweep) in parallel.
set -euo pipefail
REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
LOG_ROOT="/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"

nohup bash examples/benchmark/wait_b_prefix_grpo_rsmix_v13_pareto32_ckpt_sweep.sh \
  >>"$LOG_ROOT/launch_v13_${TS}.log" 2>&1 &
PID_B=$!
echo "Phase B v13 pid=$PID_B log=$LOG_ROOT/launch_v13_${TS}.log"

nohup bash examples/benchmark/wait_grpo_rsmix_v12_top3_strict4_train_eval.sh \
  >>"$LOG_ROOT/launch_phaseA_top3_${TS}.log" 2>&1 &
PID_A=$!
echo "Phase A top3 pid=$PID_A log=$LOG_ROOT/launch_phaseA_top3_${TS}.log"
