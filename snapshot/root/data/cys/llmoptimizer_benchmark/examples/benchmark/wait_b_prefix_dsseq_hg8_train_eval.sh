#!/usr/bin/env bash
# Train Setup-B dsseq on top-8 light16 by BO growth_pct (4 cls + 4 rgs), then algo40 eval + tables.
# Reuses existing B prefix rollouts filtered to light_8_high_growth.json.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
BUF="${BUF:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_light8_highgrowth}"
OUT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_hg8"
TRAIN_GPU="${TRAIN_GPU:-0}"
MIN_RECORDS="${MIN_RECORDS:-320}"  # 8 × 10 prefixes × 4 samples
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/b_prefix_dsseq_hg8_train_eval_${TS}.log"

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

n=$(count_records)
log "=== B dsseq hg8: buffer=$BUF records=$n (need >= $MIN_RECORDS) ==="
if (( n < MIN_RECORDS )); then
  log "ERROR buffer too small"
  exit 1
fi

export PATH="$LF_ENV/bin:$PATH"
if [[ -f "$OUT/adapter_model.safetensors" ]]; then
  log "adapter already present — skip train: $OUT"
else
  log "=== train B_dsseq_hg8 GPU=$TRAIN_GPU per_dataset --dataset-sequential ==="
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$OUT" \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    >>"$PIPE" 2>&1
  log "train done → $OUT"
fi

log "=== algo40 eval STAGE=s1a_b_prefix_dsseq_hg8 ==="
NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}" PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
  STAGE=s1a_b_prefix_dsseq_hg8 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== regenerate avg-rank tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== B dsseq hg8 train+eval+tables complete ==="
echo "$PIPE"
#!/usr/bin/env bash
# Train Setup-B dsseq on top-8 light16 by BO growth_pct (4 cls + 4 rgs), then algo40 eval + tables.
# Reuses existing B prefix rollouts filtered to light_8_high_growth.json.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
BUF="${BUF:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_light8_highgrowth}"
OUT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_hg8"
TRAIN_GPU="${TRAIN_GPU:-0}"
MIN_RECORDS="${MIN_RECORDS:-320}"  # 8 × 10 prefixes × 4 samples
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/b_prefix_dsseq_hg8_train_eval_${TS}.log"

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

n=$(count_records)
log "=== B dsseq hg8: buffer=$BUF records=$n (need >= $MIN_RECORDS) ==="
if (( n < MIN_RECORDS )); then
  log "ERROR buffer too small"
  exit 1
fi

export PATH="$LF_ENV/bin:$PATH"
if [[ -f "$OUT/adapter_model.safetensors" ]]; then
  log "adapter already present — skip train: $OUT"
else
  log "=== train B_dsseq_hg8 GPU=$TRAIN_GPU per_dataset --dataset-sequential ==="
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$BUF" \
    --init-adapter "$INIT" \
    --output-dir "$OUT" \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    >>"$PIPE" 2>&1
  log "train done → $OUT"
fi

log "=== algo40 eval STAGE=s1a_b_prefix_dsseq_hg8 ==="
NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}" PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
  STAGE=s1a_b_prefix_dsseq_hg8 HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== regenerate avg-rank tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== B dsseq hg8 train+eval+tables complete ==="
echo "$PIPE"
