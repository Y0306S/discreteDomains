#!/usr/bin/env bash
# After R0 light rollouts finish: REINFORCE train R0a+R0b, then algo40 eval + tables.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
W_R0A="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0a"
W_R0B="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0b"
BUF_A="benchmark_data/rl_rollouts/R0a_T0p7_light16"
BUF_B="benchmark_data/rl_rollouts/R0b_T1p0_light16"
MIN_RECORDS_PER_ARM="${MIN_RECORDS_PER_ARM:-500}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_train_eval_${TS}.log"

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

log "=== wait for R0 light rollout launcher / buffers ==="
# Wait until rollout launcher exits (if still running) and buffers are large enough
while pgrep -f 'run_r0_light_rollouts.sh' >/dev/null 2>&1; do
  na=$(count_records "$BUF_A"); nb=$(count_records "$BUF_B")
  log "WAIT rollouts running… R0a_records=$na R0b_records=$nb"
  sleep 60
done

while true; do
  na=$(count_records "$BUF_A"); nb=$(count_records "$BUF_B")
  log "buffers R0a=$na R0b=$nb (need >= $MIN_RECORDS_PER_ARM each)"
  if (( na >= MIN_RECORDS_PER_ARM && nb >= MIN_RECORDS_PER_ARM )); then
    break
  fi
  # If launcher died early with too few records, keep waiting a bit then fail
  if ! pgrep -f 'run_p5_rl_rollout_buffer.py' >/dev/null 2>&1; then
    if (( na < MIN_RECORDS_PER_ARM || nb < MIN_RECORDS_PER_ARM )); then
      log "ERROR: rollout workers gone but buffers incomplete (R0a=$na R0b=$nb)"
      exit 1
    fi
  fi
  sleep 60
done
log "READY buffers R0a=$na R0b=$nb"

# Train R0a/R0b in parallel on GPU 0 and GPU 1 (policy+ref ~2x model each).
export PATH="$LF_ENV/bin:$PATH"
declare -a TRAIN_PIDS=()

start_train() {
  local gpu="$1" name="$2" buf="$3" out="$4"
  if [[ -f "$out/adapter_model.safetensors" ]]; then
    log "$name adapter already present — skip train"
    return 0
  fi
  # Another train already writing this output? wait for it instead of launching a duplicate.
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

start_train 0 "R0a" "$BUF_A" "$W_R0A"
start_train 1 "R0b" "$BUF_B" "$W_R0B"

for pid in "${TRAIN_PIDS[@]:-}"; do
  if ! wait "$pid"; then
    log "ERROR train pid=$pid failed"
    exit 1
  fi
done

# If we skipped launch because a pre-existing train was running, wait for adapters.
while [[ ! -f "$W_R0A/adapter_model.safetensors" ]] || [[ ! -f "$W_R0B/adapter_model.safetensors" ]]; do
  if ! pgrep -f 'train_p5_reinforce.py' >/dev/null 2>&1; then
    log "ERROR: no train_p5_reinforce running but adapters incomplete"
    [[ -f "$W_R0A/adapter_model.safetensors" ]] || log " missing $W_R0A"
    [[ -f "$W_R0B/adapter_model.safetensors" ]] || log " missing $W_R0B"
    exit 1
  fi
  log "WAIT adapters… R0a=$([[ -f $W_R0A/adapter_model.safetensors ]] && echo ok || echo pending) R0b=$([[ -f $W_R0B/adapter_model.safetensors ]] && echo ok || echo pending)"
  sleep 30
done
log "READY adapters R0a+R0b"

log "=== algo40 eval STAGE=s1a_r0_light (R0a+R0b) SEED=1 ==="
# Default leave GPUs 2,3 free for R0c/R0d medium rollouts (override via EVAL_NUM_GPUS / EVAL_GPU_MAP).
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-2}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-0,1}"
log "eval GPUs: NUM_GPUS=$EVAL_NUM_GPUS EVAL_GPU_MAP=$EVAL_GPU_MAP"
NUM_GPUS="$EVAL_NUM_GPUS" EVAL_GPU_MAP="$EVAL_GPU_MAP" JOBS_PER_GPU=3 PACK_MIN_FREE_MB=20000 \
  STAGE=s1a_r0_light HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
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

log "=== R0 light train+eval+tables complete ==="
#!/usr/bin/env bash
# After R0 light rollouts finish: REINFORCE train R0a+R0b, then algo40 eval + tables.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
INIT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
W_R0A="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0a"
W_R0B="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0b"
BUF_A="benchmark_data/rl_rollouts/R0a_T0p7_light16"
BUF_B="benchmark_data/rl_rollouts/R0b_T1p0_light16"
MIN_RECORDS_PER_ARM="${MIN_RECORDS_PER_ARM:-500}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_train_eval_${TS}.log"

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

log "=== wait for R0 light rollout launcher / buffers ==="
# Wait until rollout launcher exits (if still running) and buffers are large enough
while pgrep -f 'run_r0_light_rollouts.sh' >/dev/null 2>&1; do
  na=$(count_records "$BUF_A"); nb=$(count_records "$BUF_B")
  log "WAIT rollouts running… R0a_records=$na R0b_records=$nb"
  sleep 60
done

while true; do
  na=$(count_records "$BUF_A"); nb=$(count_records "$BUF_B")
  log "buffers R0a=$na R0b=$nb (need >= $MIN_RECORDS_PER_ARM each)"
  if (( na >= MIN_RECORDS_PER_ARM && nb >= MIN_RECORDS_PER_ARM )); then
    break
  fi
  # If launcher died early with too few records, keep waiting a bit then fail
  if ! pgrep -f 'run_p5_rl_rollout_buffer.py' >/dev/null 2>&1; then
    if (( na < MIN_RECORDS_PER_ARM || nb < MIN_RECORDS_PER_ARM )); then
      log "ERROR: rollout workers gone but buffers incomplete (R0a=$na R0b=$nb)"
      exit 1
    fi
  fi
  sleep 60
done
log "READY buffers R0a=$na R0b=$nb"

# Train R0a/R0b in parallel on GPU 0 and GPU 1 (policy+ref ~2x model each).
export PATH="$LF_ENV/bin:$PATH"
declare -a TRAIN_PIDS=()

start_train() {
  local gpu="$1" name="$2" buf="$3" out="$4"
  if [[ -f "$out/adapter_model.safetensors" ]]; then
    log "$name adapter already present — skip train"
    return 0
  fi
  # Another train already writing this output? wait for it instead of launching a duplicate.
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

start_train 0 "R0a" "$BUF_A" "$W_R0A"
start_train 1 "R0b" "$BUF_B" "$W_R0B"

for pid in "${TRAIN_PIDS[@]:-}"; do
  if ! wait "$pid"; then
    log "ERROR train pid=$pid failed"
    exit 1
  fi
done

# If we skipped launch because a pre-existing train was running, wait for adapters.
while [[ ! -f "$W_R0A/adapter_model.safetensors" ]] || [[ ! -f "$W_R0B/adapter_model.safetensors" ]]; do
  if ! pgrep -f 'train_p5_reinforce.py' >/dev/null 2>&1; then
    log "ERROR: no train_p5_reinforce running but adapters incomplete"
    [[ -f "$W_R0A/adapter_model.safetensors" ]] || log " missing $W_R0A"
    [[ -f "$W_R0B/adapter_model.safetensors" ]] || log " missing $W_R0B"
    exit 1
  fi
  log "WAIT adapters… R0a=$([[ -f $W_R0A/adapter_model.safetensors ]] && echo ok || echo pending) R0b=$([[ -f $W_R0B/adapter_model.safetensors ]] && echo ok || echo pending)"
  sleep 30
done
log "READY adapters R0a+R0b"

log "=== algo40 eval STAGE=s1a_r0_light (R0a+R0b) SEED=1 ==="
# Default leave GPUs 2,3 free for R0c/R0d medium rollouts (override via EVAL_NUM_GPUS / EVAL_GPU_MAP).
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-2}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-0,1}"
log "eval GPUs: NUM_GPUS=$EVAL_NUM_GPUS EVAL_GPU_MAP=$EVAL_GPU_MAP"
NUM_GPUS="$EVAL_NUM_GPUS" EVAL_GPU_MAP="$EVAL_GPU_MAP" JOBS_PER_GPU=3 PACK_MIN_FREE_MB=20000 \
  STAGE=s1a_r0_light HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
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

log "=== R0 light train+eval+tables complete ==="
