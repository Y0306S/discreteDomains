#!/usr/bin/env bash
# One Setup-B dsseq train on medium-32:
#   - mid-checkpoint after top-8 high-growth datasets -> b_prefix_dsseq_hg8
#   - final after all 32 -> b_prefix_dsseq_m32
# Then algo40 eval both + tables.
#
# Prerequisites: B rollouts for all medium-32 under BUF (light16 already done;
# launch missing 16 via run_setupB_prefix_rollouts.sh with medium_32_minus_light16.json).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
BUF="${BUF:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_medium32}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/medium_32_train_order_hg8_first.json}"
OUT_MID="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_hg8"
OUT_FINAL="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_m32"
TRAIN_GPU="${TRAIN_GPU:-0}"
MIN_RECORDS="${MIN_RECORDS:-1280}"  # 32 × 10 × 4
MID_AFTER="${MID_AFTER:-8}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/b_prefix_dsseq_m32_hg8ckpt_train_eval_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

count_records() {
  "$PY_TRAIN" - <<PY
from pathlib import Path
root = Path("$BUF")
n = 0
if root.exists():
    for p in root.rglob("rollouts_*.jsonl"):
        n += sum(1 for ln in p.open() if ln.strip())
print(n)
PY
}

log "=== wait B medium32 buffer ($BUF) need >= $MIN_RECORDS ==="
while true; do
  n=$(count_records)
  log "buffer records=$n"
  if (( n >= MIN_RECORDS )); then
    break
  fi
  sleep 60
done

export PATH="$LF_ENV/bin:$PATH"
if [[ -f "$OUT_FINAL/adapter_model.safetensors" && -f "$OUT_MID/adapter_model.safetensors" ]]; then
  log "both adapters present — skip train"
else
  log "=== train once: mid@$MID_AFTER -> $OUT_MID ; final -> $OUT_FINAL GPU=$TRAIN_GPU ==="
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$OUT_FINAL" \
    --mid-checkpoint-dir "$OUT_MID" \
    --mid-checkpoint-after "$MID_AFTER" \
    --dataset-order-file "$ORDER" \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    >>"$PIPE" 2>&1
  log "train done"
fi

log "=== algo40 eval both hg8 + m32 ==="
NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}" PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
  STAGE=s1a_b_prefix_dsseq_hg8_m32 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true
log "=== complete ==="
echo "$PIPE"
