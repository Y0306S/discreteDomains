#!/usr/bin/env bash
# After Setup A/B rollouts: train 2 loss variants each from s1a_ep7p5, then algo40 eval.
# Setup A buffers: global vs per_dataset+sequential
# Setup B buffers: grpo_group vs per_dataset+sequential
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
BUF_A="${BUF_A:-benchmark_data/rl_rollouts/A_boseed100_new4_T1p0_light16}"
BUF_B="${BUF_B:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_light16}"
W_A_GLOBAL="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_a_boseed_global"
W_A_DSSEQ="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_a_boseed_dsseq"
W_B_GRPO="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_grpo"
W_B_DSSEQ="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq"
TRAIN_GPU_A_GLOBAL="${TRAIN_GPU_A_GLOBAL:-0}"
TRAIN_GPU_A_DSSEQ="${TRAIN_GPU_A_DSSEQ:-1}"
TRAIN_GPU_B_GRPO="${TRAIN_GPU_B_GRPO:-2}"
TRAIN_GPU_B_DSSEQ="${TRAIN_GPU_B_DSSEQ:-3}"
# Back-compat aliases
TRAIN_GPU_A="${TRAIN_GPU_A:-$TRAIN_GPU_A_GLOBAL}"
TRAIN_GPU_B="${TRAIN_GPU_B:-$TRAIN_GPU_A_DSSEQ}"
MIN_A="${MIN_A:-64}"   # 16 tasks × 4
MIN_B="${MIN_B:-640}"  # 16 × 10 × 4
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/setupAB_train_eval_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

count_records() {
  local root="$1"
  "$PY_TRAIN" - <<PY
from pathlib import Path
root = Path("$root")
n = 0
if root.exists():
    for p in root.rglob("rollouts_*.jsonl"):
        n += sum(1 for ln in p.open() if ln.strip())
print(n)
PY
}

wait_buf() {
  local name="$1" root="$2" need="$3"
  while true; do
    n=$(count_records "$root")
    log "WAIT $name buffer records=$n (need >= $need)"
    if (( n >= need )); then
      break
    fi
    sleep 60
  done
  log "READY $name buffer records=$n"
}

train_one() {
  local gpu="$1" name="$2" buf="$3" out="$4" mode="$5" extra="${6:-}"
  if [[ -f "$out/adapter_model.safetensors" ]]; then
    log "$name adapter already present — skip"
    return 0
  fi
  log "=== train $name mode=$mode GPU=$gpu ==="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$buf" \
    --init-adapter "$INIT" \
    --output-dir "$out" \
    --advantage-mode "$mode" \
    $extra \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    >>"$PIPE" 2>&1
}

log "=== Setup A/B train+eval pipeline (4-way GPU parallel) ==="
wait_buf "A" "$BUF_A" "$MIN_A"
wait_buf "B" "$BUF_B" "$MIN_B"

export PATH="$LF_ENV/bin:$PATH"
# All four adapters in parallel on distinct GPUs
train_one "$TRAIN_GPU_A_GLOBAL" "A_global" "$BUF_A" "$W_A_GLOBAL" "global" &
pid_ag=$!
train_one "$TRAIN_GPU_A_DSSEQ" "A_dsseq" "$BUF_A" "$W_A_DSSEQ" "per_dataset" "--dataset-sequential" &
pid_ad=$!
train_one "$TRAIN_GPU_B_GRPO" "B_grpo" "$BUF_B" "$W_B_GRPO" "grpo_group" &
pid_bg=$!
train_one "$TRAIN_GPU_B_DSSEQ" "B_dsseq" "$BUF_B" "$W_B_DSSEQ" "per_dataset" "--dataset-sequential" &
pid_bd=$!
fail=0
for pid in "$pid_ag" "$pid_ad" "$pid_bg" "$pid_bd"; do
  if ! wait "$pid"; then
    log "ERROR train pid=$pid failed"
    fail=1
  fi
done
if (( fail )); then
  log "ERROR: one or more trains failed"
  exit 1
fi

log "READY all 4 adapters"

log "=== algo40 eval STAGE=s1a_setup_ab (3 workers/GPU) ==="
NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-3}" PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
  STAGE=s1a_setup_ab HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== Setup A/B train+eval+tables complete ==="
