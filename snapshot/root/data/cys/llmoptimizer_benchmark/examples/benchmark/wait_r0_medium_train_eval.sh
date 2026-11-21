#!/usr/bin/env bash
# After R0 medium rollouts finish: REINFORCE train R0c+R0d, then algo40 eval + tables.
# Waits for medium buffers; trains on GPUs 2,3 by default (leaves 0,1 for light train/eval).
# Medium eval waits for the light waiter to finish so evals don't fight for GPUs.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
W_R0C="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0c"
W_R0D="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0d"
BUF_C="benchmark_data/rl_rollouts/R0c_T0p7_medium32"
BUF_D="benchmark_data/rl_rollouts/R0d_T1p0_medium32"
# 32 tasks × 40 trials
MIN_RECORDS_PER_ARM="${MIN_RECORDS_PER_ARM:-1200}"
TRAIN_GPU_C="${TRAIN_GPU_C:-2}"
TRAIN_GPU_D="${TRAIN_GPU_D:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_medium_train_eval_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

count_records() {
  local root="$1"
  "$EVAL_ENV/bin/python" - <<PY
from pathlib import Path
root = Path("$root")
n = 0
if root.exists():
    for p in root.rglob("rollouts_*.jsonl"):
        n += sum(1 for ln in p.open() if ln.strip())
print(n)
PY
}

medium_rollouts_alive() {
  pgrep -f 'run_r0_medium_rollouts.sh' >/dev/null 2>&1
}

# Only count medium-buffer rollout workers (avoid false "still running" from light/other).
medium_workers_alive() {
  pgrep -af 'run_p5_rl_rollout_buffer.py' 2>/dev/null | grep -E 'R0c_T0p7_medium32|R0d_T1p0_medium32' | grep -v grep >/dev/null
}

log "=== wait for R0 medium rollout launcher / buffers ==="
while medium_rollouts_alive; do
  nc=$(count_records "$BUF_C"); nd=$(count_records "$BUF_D")
  log "WAIT medium rollouts… R0c_records=$nc R0d_records=$nd"
  sleep 60
done

while true; do
  nc=$(count_records "$BUF_C"); nd=$(count_records "$BUF_D")
  log "buffers R0c=$nc R0d=$nd (need >= $MIN_RECORDS_PER_ARM each)"
  if (( nc >= MIN_RECORDS_PER_ARM && nd >= MIN_RECORDS_PER_ARM )); then
    break
  fi
  if ! medium_workers_alive; then
    if (( nc < MIN_RECORDS_PER_ARM || nd < MIN_RECORDS_PER_ARM )); then
      log "ERROR: medium rollout workers gone but buffers incomplete (R0c=$nc R0d=$nd)"
      exit 1
    fi
  fi
  sleep 60
done
log "READY buffers R0c=$nc R0d=$nd"

export PATH="$LF_ENV/bin:$PATH"
declare -a TRAIN_PIDS=()

start_train() {
  local gpu="$1" name="$2" buf="$3" out="$4"
  if [[ -f "$out/adapter_model.safetensors" ]]; then
    log "$name adapter already present — skip train"
    return 0
  fi
  if pgrep -f "train_p5_reinforce.py .*--output-dir ${out}" >/dev/null 2>&1; then
    log "$name train already running — will wait for adapter"
    return 0
  fi
  log "=== train $name REINFORCE on GPU $gpu ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$buf" \
    --init-adapter "$INIT" \
    --output-dir "$out" \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    >>"$PIPE" 2>&1 &
  TRAIN_PIDS+=("$!")
}

start_train "$TRAIN_GPU_C" "R0c" "$BUF_C" "$W_R0C"
start_train "$TRAIN_GPU_D" "R0d" "$BUF_D" "$W_R0D"

for pid in "${TRAIN_PIDS[@]:-}"; do
  if ! wait "$pid"; then
    log "ERROR train pid=$pid failed"
    exit 1
  fi
done

while [[ ! -f "$W_R0C/adapter_model.safetensors" ]] || [[ ! -f "$W_R0D/adapter_model.safetensors" ]]; do
  if ! pgrep -f 'train_p5_reinforce.py' >/dev/null 2>&1; then
    log "ERROR: no train_p5_reinforce running but medium adapters incomplete"
    [[ -f "$W_R0C/adapter_model.safetensors" ]] || log " missing $W_R0C"
    [[ -f "$W_R0D/adapter_model.safetensors" ]] || log " missing $W_R0D"
    exit 1
  fi
  log "WAIT adapters… R0c=$([[ -f $W_R0C/adapter_model.safetensors ]] && echo ok || echo pending) R0d=$([[ -f $W_R0D/adapter_model.safetensors ]] && echo ok || echo pending)"
  sleep 30
done
log "READY adapters R0c+R0d"

# Let light train+eval finish first (it uses GPUs 0,1; avoid overlapping algo40 evals).
while pgrep -f 'wait_r0_light_train_eval.sh' >/dev/null 2>&1; do
  log "WAIT light train/eval waiter to finish before medium eval…"
  sleep 60
done
# Also wait if a light eval is still running under a detached shell
while pgrep -af 'run_mltest20_algo40_eval.sh' 2>/dev/null | grep -E 's1a_r0_light|STAGE=s1a_r0_light' | grep -v grep >/dev/null; do
  log "WAIT light algo40 eval still running…"
  sleep 60
done

log "=== algo40 eval STAGE=s1a_r0_medium (R0c+R0d) SEED=1 ==="
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-}"
log "eval GPUs: NUM_GPUS=$EVAL_NUM_GPUS EVAL_GPU_MAP=${EVAL_GPU_MAP:-identity}"
NUM_GPUS="$EVAL_NUM_GPUS" EVAL_GPU_MAP="$EVAL_GPU_MAP" JOBS_PER_GPU=3 PACK_MIN_FREE_MB=20000 \
  STAGE=s1a_r0_medium HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== regenerate full + seed1 tables ==="
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
  --rgs-pooled-trial-norm \
  --out-dir benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank \
  >>"$PIPE" 2>&1
python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py \
  --rgs-pooled-trial-norm --seed1-only \
  --out-dir benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank \
  >>"$PIPE" 2>&1

log "=== R0 medium train+eval+tables complete ==="
