#!/usr/bin/env bash
# Parallel R0c (T=0.7) + R0d (T=1.0) REINFORCE rollouts on medium-32 tasks.
# Default: physical GPUs 2,3 (leave 0,1 for light R0a/R0b train+eval).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/medium_32.json}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
# Comma-separated physical GPU ids
GPU_IDS="${GPU_IDS:-2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_medium_rollouts_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

IFS=',' read -ra GPUS <<<"$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$TASKS_JSON" ]]; then
  echo "Missing tasks: $TASKS_JSON" >&2
  exit 1
fi

declare -a QUEUE=()
while IFS= read -r line; do
  QUEUE+=("$line")
done < <("$PY" - <<'PY'
import json
from pathlib import Path
tasks = json.loads(Path("benchmark_data/rl_task_selection/medium_32.json").read_text())
arms = [
    ("R0c", 0.7, "benchmark_data/rl_rollouts/R0c_T0p7_medium32"),
    ("R0d", 1.0, "benchmark_data/rl_rollouts/R0d_T1p0_medium32"),
]
for t in tasks:
    for arm, temp, out in arms:
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 medium rollouts: $TOTAL jobs (32 tasks × 2 temps) GPUs=${GPU_IDS} jobs/gpu=$JOBS_PER_GPU ==="
log "adapter=$ADAPTER n_trials=$N_TRIALS seed=$SEED metric defaults: bal_acc/mse"

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

declare -a GPU_PIDS=()
declare -a ALL_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[i]=""; done

count_live() {
  local i="$1" live=0
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live+1)); fi
  done
  echo "$live"
}

prune_pids() {
  local i="$1" kept=""
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$i]="${kept% }"
}

run_one() {
  local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
  local outdir="$outroot/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_${TS}.log"
  local metric="bal_acc"
  [[ "$tt" == "rgs" ]] && metric="mse"
  mkdir -p "$outdir"
  log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo metric=$metric"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
    --metric "$metric" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$outdir" \
    --prompt-variant P5 \
    --hybrid-variant H0 \
    --qwen-backend hf \
    --qwen-model-path "$MODEL_PATH" \
    --qwen-adapter-path "$ADAPTER" \
    --llm-response-format json_only \
    --llm-max-tokens 768 \
    --llm-temperature "$temp" \
    --reward-delta-best 1.0 \
    --reward-absolute 0.1 \
    --penalty-parse-fail 0.5 \
    --penalty-duplicate 0.1 \
    --penalty-eval-fail 1.0 \
    --resume \
    >>"$logfile" 2>&1 &
  local pid=$!
  ALL_PIDS+=("$pid")
  # map physical gpu -> slot index for PID tracking
  local slot=-1
  for ((s=0; s<NUM_GPUS; s++)); do
    if [[ "${GPUS[$s]}" == "$gpu" ]]; then slot=$s; break; fi
  done
  if [[ -n "${GPU_PIDS[$slot]:-}" ]]; then
    GPU_PIDS[$slot]="${GPU_PIDS[$slot]} $pid"
  else
    GPU_PIDS[$slot]="$pid"
  fi
}

idx=0
while (( idx < TOTAL )); do
  launched=0
  for ((s=0; s<NUM_GPUS; s++)); do
    g="${GPUS[$s]}"
    prune_pids "$s"
    live=$(count_live "$s")
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < JOBS_PER_GPU && idx < TOTAL )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
      idx=$((idx+1))
      run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
      launched=1
      live=$(count_live "$s")
      free=$(gpu_free_mb "$g" || echo 0)
      sleep 2
    done
  done
  if (( launched == 0 )); then
    sleep 20
  fi
done

log "All $TOTAL jobs dispatched — waiting for ${#ALL_PIDS[@]} pids"
fail=0
for pid in "${ALL_PIDS[@]}"; do
  if ! wait "$pid"; then
    log "FAIL pid=$pid"
    fail=1
  fi
done

"$PY" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path
for name in ["R0c_T0p7_medium32", "R0d_T1p0_medium32"]:
    root = Path("benchmark_data/rl_rollouts") / name
    n_rec = 0
    n40 = 0
    if root.exists():
        for p in root.rglob("rollouts_*.jsonl"):
            n = sum(1 for ln in p.open() if ln.strip())
            n_rec += n
            if n >= 40:
                n40 += 1
    print(f"{name}: records={n_rec} tasks_with_40plus={n40}")
PY

if (( fail )); then
  log "=== R0 medium rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== R0 medium rollouts complete ==="
#!/usr/bin/env bash
# Parallel R0c (T=0.7) + R0d (T=1.0) REINFORCE rollouts on medium-32 tasks.
# Default: physical GPUs 2,3 (leave 0,1 for light R0a/R0b train+eval).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/medium_32.json}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
# Comma-separated physical GPU ids
GPU_IDS="${GPU_IDS:-2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_medium_rollouts_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

IFS=',' read -ra GPUS <<<"$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$TASKS_JSON" ]]; then
  echo "Missing tasks: $TASKS_JSON" >&2
  exit 1
fi

declare -a QUEUE=()
while IFS= read -r line; do
  QUEUE+=("$line")
done < <("$PY" - <<'PY'
import json
from pathlib import Path
tasks = json.loads(Path("benchmark_data/rl_task_selection/medium_32.json").read_text())
arms = [
    ("R0c", 0.7, "benchmark_data/rl_rollouts/R0c_T0p7_medium32"),
    ("R0d", 1.0, "benchmark_data/rl_rollouts/R0d_T1p0_medium32"),
]
for t in tasks:
    for arm, temp, out in arms:
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 medium rollouts: $TOTAL jobs (32 tasks × 2 temps) GPUs=${GPU_IDS} jobs/gpu=$JOBS_PER_GPU ==="
log "adapter=$ADAPTER n_trials=$N_TRIALS seed=$SEED metric defaults: bal_acc/mse"

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

declare -a GPU_PIDS=()
declare -a ALL_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[i]=""; done

count_live() {
  local i="$1" live=0
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live+1)); fi
  done
  echo "$live"
}

prune_pids() {
  local i="$1" kept=""
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$i]="${kept% }"
}

run_one() {
  local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
  local outdir="$outroot/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_${TS}.log"
  local metric="bal_acc"
  [[ "$tt" == "rgs" ]] && metric="mse"
  mkdir -p "$outdir"
  log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo metric=$metric"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
    --metric "$metric" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$outdir" \
    --prompt-variant P5 \
    --hybrid-variant H0 \
    --qwen-backend hf \
    --qwen-model-path "$MODEL_PATH" \
    --qwen-adapter-path "$ADAPTER" \
    --llm-response-format json_only \
    --llm-max-tokens 768 \
    --llm-temperature "$temp" \
    --reward-delta-best 1.0 \
    --reward-absolute 0.1 \
    --penalty-parse-fail 0.5 \
    --penalty-duplicate 0.1 \
    --penalty-eval-fail 1.0 \
    --resume \
    >>"$logfile" 2>&1 &
  local pid=$!
  ALL_PIDS+=("$pid")
  # map physical gpu -> slot index for PID tracking
  local slot=-1
  for ((s=0; s<NUM_GPUS; s++)); do
    if [[ "${GPUS[$s]}" == "$gpu" ]]; then slot=$s; break; fi
  done
  if [[ -n "${GPU_PIDS[$slot]:-}" ]]; then
    GPU_PIDS[$slot]="${GPU_PIDS[$slot]} $pid"
  else
    GPU_PIDS[$slot]="$pid"
  fi
}

idx=0
while (( idx < TOTAL )); do
  launched=0
  for ((s=0; s<NUM_GPUS; s++)); do
    g="${GPUS[$s]}"
    prune_pids "$s"
    live=$(count_live "$s")
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < JOBS_PER_GPU && idx < TOTAL )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
      idx=$((idx+1))
      run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
      launched=1
      live=$(count_live "$s")
      free=$(gpu_free_mb "$g" || echo 0)
      sleep 2
    done
  done
  if (( launched == 0 )); then
    sleep 20
  fi
done

log "All $TOTAL jobs dispatched — waiting for ${#ALL_PIDS[@]} pids"
fail=0
for pid in "${ALL_PIDS[@]}"; do
  if ! wait "$pid"; then
    log "FAIL pid=$pid"
    fail=1
  fi
done

"$PY" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path
for name in ["R0c_T0p7_medium32", "R0d_T1p0_medium32"]:
    root = Path("benchmark_data/rl_rollouts") / name
    n_rec = 0
    n40 = 0
    if root.exists():
        for p in root.rglob("rollouts_*.jsonl"):
            n = sum(1 for ln in p.open() if ln.strip())
            n_rec += n
            if n >= 40:
                n40 += 1
    print(f"{name}: records={n_rec} tasks_with_40plus={n40}")
PY

if (( fail )); then
  log "=== R0 medium rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== R0 medium rollouts complete ==="
#!/usr/bin/env bash
# Parallel R0c (T=0.7) + R0d (T=1.0) REINFORCE rollouts on medium-32 tasks.
# Default: physical GPUs 2,3 (leave 0,1 for light R0a/R0b train+eval).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/medium_32.json}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
# Comma-separated physical GPU ids
GPU_IDS="${GPU_IDS:-2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_medium_rollouts_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

IFS=',' read -ra GPUS <<<"$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$TASKS_JSON" ]]; then
  echo "Missing tasks: $TASKS_JSON" >&2
  exit 1
fi

declare -a QUEUE=()
while IFS= read -r line; do
  QUEUE+=("$line")
done < <("$PY" - <<'PY'
import json
from pathlib import Path
tasks = json.loads(Path("benchmark_data/rl_task_selection/medium_32.json").read_text())
arms = [
    ("R0c", 0.7, "benchmark_data/rl_rollouts/R0c_T0p7_medium32"),
    ("R0d", 1.0, "benchmark_data/rl_rollouts/R0d_T1p0_medium32"),
]
for t in tasks:
    for arm, temp, out in arms:
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 medium rollouts: $TOTAL jobs (32 tasks × 2 temps) GPUs=${GPU_IDS} jobs/gpu=$JOBS_PER_GPU ==="
log "adapter=$ADAPTER n_trials=$N_TRIALS seed=$SEED metric defaults: bal_acc/mse"

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

declare -a GPU_PIDS=()
declare -a ALL_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[i]=""; done

count_live() {
  local i="$1" live=0
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live+1)); fi
  done
  echo "$live"
}

prune_pids() {
  local i="$1" kept=""
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$i]="${kept% }"
}

run_one() {
  local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
  local outdir="$outroot/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_${TS}.log"
  local metric="bal_acc"
  [[ "$tt" == "rgs" ]] && metric="mse"
  mkdir -p "$outdir"
  log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo metric=$metric"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
    --metric "$metric" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$outdir" \
    --prompt-variant P5 \
    --hybrid-variant H0 \
    --qwen-backend hf \
    --qwen-model-path "$MODEL_PATH" \
    --qwen-adapter-path "$ADAPTER" \
    --llm-response-format json_only \
    --llm-max-tokens 768 \
    --llm-temperature "$temp" \
    --reward-delta-best 1.0 \
    --reward-absolute 0.1 \
    --penalty-parse-fail 0.5 \
    --penalty-duplicate 0.1 \
    --penalty-eval-fail 1.0 \
    --resume \
    >>"$logfile" 2>&1 &
  local pid=$!
  ALL_PIDS+=("$pid")
  # map physical gpu -> slot index for PID tracking
  local slot=-1
  for ((s=0; s<NUM_GPUS; s++)); do
    if [[ "${GPUS[$s]}" == "$gpu" ]]; then slot=$s; break; fi
  done
  if [[ -n "${GPU_PIDS[$slot]:-}" ]]; then
    GPU_PIDS[$slot]="${GPU_PIDS[$slot]} $pid"
  else
    GPU_PIDS[$slot]="$pid"
  fi
}

idx=0
while (( idx < TOTAL )); do
  launched=0
  for ((s=0; s<NUM_GPUS; s++)); do
    g="${GPUS[$s]}"
    prune_pids "$s"
    live=$(count_live "$s")
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < JOBS_PER_GPU && idx < TOTAL )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
      idx=$((idx+1))
      run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
      launched=1
      live=$(count_live "$s")
      free=$(gpu_free_mb "$g" || echo 0)
      sleep 2
    done
  done
  if (( launched == 0 )); then
    sleep 20
  fi
done

log "All $TOTAL jobs dispatched — waiting for ${#ALL_PIDS[@]} pids"
fail=0
for pid in "${ALL_PIDS[@]}"; do
  if ! wait "$pid"; then
    log "FAIL pid=$pid"
    fail=1
  fi
done

"$PY" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path
for name in ["R0c_T0p7_medium32", "R0d_T1p0_medium32"]:
    root = Path("benchmark_data/rl_rollouts") / name
    n_rec = 0
    n40 = 0
    if root.exists():
        for p in root.rglob("rollouts_*.jsonl"):
            n = sum(1 for ln in p.open() if ln.strip())
            n_rec += n
            if n >= 40:
                n40 += 1
    print(f"{name}: records={n_rec} tasks_with_40plus={n40}")
PY

if (( fail )); then
  log "=== R0 medium rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== R0 medium rollouts complete ==="
#!/usr/bin/env bash
# Parallel R0c (T=0.7) + R0d (T=1.0) REINFORCE rollouts on medium-32 tasks.
# Default: physical GPUs 2,3 (leave 0,1 for light R0a/R0b train+eval).
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/medium_32.json}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
# Comma-separated physical GPU ids
GPU_IDS="${GPU_IDS:-2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_medium_rollouts_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

IFS=',' read -ra GPUS <<<"$GPU_IDS"
NUM_GPUS=${#GPUS[@]}

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$TASKS_JSON" ]]; then
  echo "Missing tasks: $TASKS_JSON" >&2
  exit 1
fi

declare -a QUEUE=()
while IFS= read -r line; do
  QUEUE+=("$line")
done < <("$PY" - <<'PY'
import json
from pathlib import Path
tasks = json.loads(Path("benchmark_data/rl_task_selection/medium_32.json").read_text())
arms = [
    ("R0c", 0.7, "benchmark_data/rl_rollouts/R0c_T0p7_medium32"),
    ("R0d", 1.0, "benchmark_data/rl_rollouts/R0d_T1p0_medium32"),
]
for t in tasks:
    for arm, temp, out in arms:
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 medium rollouts: $TOTAL jobs (32 tasks × 2 temps) GPUs=${GPU_IDS} jobs/gpu=$JOBS_PER_GPU ==="
log "adapter=$ADAPTER n_trials=$N_TRIALS seed=$SEED metric defaults: bal_acc/mse"

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

declare -a GPU_PIDS=()
declare -a ALL_PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do GPU_PIDS[i]=""; done

count_live() {
  local i="$1" live=0
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live+1)); fi
  done
  echo "$live"
}

prune_pids() {
  local i="$1" kept=""
  for pid in ${GPU_PIDS[$i]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$i]="${kept% }"
}

run_one() {
  local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
  local outdir="$outroot/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_${TS}.log"
  local metric="bal_acc"
  [[ "$tt" == "rgs" ]] && metric="mse"
  mkdir -p "$outdir"
  log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo metric=$metric"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
    --metric "$metric" \
    --n-trials "$N_TRIALS" \
    --seed "$SEED" \
    --time-limit "$TIME_LIMIT" \
    --output-dir "$outdir" \
    --prompt-variant P5 \
    --hybrid-variant H0 \
    --qwen-backend hf \
    --qwen-model-path "$MODEL_PATH" \
    --qwen-adapter-path "$ADAPTER" \
    --llm-response-format json_only \
    --llm-max-tokens 768 \
    --llm-temperature "$temp" \
    --reward-delta-best 1.0 \
    --reward-absolute 0.1 \
    --penalty-parse-fail 0.5 \
    --penalty-duplicate 0.1 \
    --penalty-eval-fail 1.0 \
    --resume \
    >>"$logfile" 2>&1 &
  local pid=$!
  ALL_PIDS+=("$pid")
  # map physical gpu -> slot index for PID tracking
  local slot=-1
  for ((s=0; s<NUM_GPUS; s++)); do
    if [[ "${GPUS[$s]}" == "$gpu" ]]; then slot=$s; break; fi
  done
  if [[ -n "${GPU_PIDS[$slot]:-}" ]]; then
    GPU_PIDS[$slot]="${GPU_PIDS[$slot]} $pid"
  else
    GPU_PIDS[$slot]="$pid"
  fi
}

idx=0
while (( idx < TOTAL )); do
  launched=0
  for ((s=0; s<NUM_GPUS; s++)); do
    g="${GPUS[$s]}"
    prune_pids "$s"
    live=$(count_live "$s")
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < JOBS_PER_GPU && idx < TOTAL )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
      idx=$((idx+1))
      run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
      launched=1
      live=$(count_live "$s")
      free=$(gpu_free_mb "$g" || echo 0)
      sleep 2
    done
  done
  if (( launched == 0 )); then
    sleep 20
  fi
done

log "All $TOTAL jobs dispatched — waiting for ${#ALL_PIDS[@]} pids"
fail=0
for pid in "${ALL_PIDS[@]}"; do
  if ! wait "$pid"; then
    log "FAIL pid=$pid"
    fail=1
  fi
done

"$PY" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path
for name in ["R0c_T0p7_medium32", "R0d_T1p0_medium32"]:
    root = Path("benchmark_data/rl_rollouts") / name
    n_rec = 0
    n40 = 0
    if root.exists():
        for p in root.rglob("rollouts_*.jsonl"):
            n = sum(1 for ln in p.open() if ln.strip())
            n_rec += n
            if n >= 40:
                n40 += 1
    print(f"{name}: records={n_rec} tasks_with_40plus={n40}")
PY

if (( fail )); then
  log "=== R0 medium rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== R0 medium rollouts complete ==="
