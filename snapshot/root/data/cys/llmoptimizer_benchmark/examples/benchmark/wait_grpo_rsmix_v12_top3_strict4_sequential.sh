#!/usr/bin/env bash
# Phase A trains: strict first-4 only, ONE job at a time on GPU 0 (avoids OOM vs v13 collect).
# After all 3 trains, runs RS3+RS5 eval (same as wait_grpo_rsmix_v12_top3_strict4_train_eval.sh).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
TRAIN_GPU="${TRAIN_GPU:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-48000}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/grpo_top3_strict4_sequential_${TS}.log"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

wait_gpu() {
  log "waiting GPU $TRAIN_GPU free >= ${MIN_FREE_MB}MB ..."
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$TRAIN_GPU" 2>/dev/null | tr -d ' ')
    if [[ -n "$free" && "$free" -ge "$MIN_FREE_MB" ]]; then
      log "GPU $TRAIN_GPU free=${free}MB — ready"
      return 0
    fi
    sleep 120
  done
}

declare -a SPECS=(
  "v4|32|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32|benchmark_data/rl_rollouts/B_rsmix_v4_n4to10_T1p0_pareto32_v12|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt32_strict4_r1"
  "v4|16|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt16|benchmark_data/rl_rollouts/B_rsmix_v4_n4to10_T1p0_pareto32_v12|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt16_strict4_r1"
  "v3|8|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12/ckpt8|benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T1p0_pareto32_v12|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12_ckpt8_strict4_r1"
)

ORDER="benchmark_data/rl_task_selection/picked_32_pareto_train_order.json"
export PATH="$LF_ENV/bin:$PATH"

log "=== Phase A sequential strict-first-4 train (no last-4) GPU=$TRAIN_GPU ==="

for spec in "${SPECS[@]}"; do
  IFS='|' read -r ver ckpt init buf out <<<"$spec"
  wait_gpu
  log "=== train $ver ckpt$ckpt strict4 -> $out ==="
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$buf" \
    --strict-first-n 4 \
    --init-adapter "$init" \
    --output-dir "$out" \
    --dataset-order-file "$ORDER" \
    --strict-dataset-order \
    --advantage-mode grpo_group \
    --dataset-sequential \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 >>"$PIPE" 2>&1
  log "train $ver ckpt$ckpt DONE"
done

log "=== trains done — running eval via main Phase A script (SKIP_TRAIN=1) ==="
SKIP_TRAIN=1 bash examples/benchmark/wait_grpo_rsmix_v12_top3_strict4_train_eval.sh >>"$PIPE" 2>&1

log "=== Phase A sequential DONE pipe=$PIPE ==="
