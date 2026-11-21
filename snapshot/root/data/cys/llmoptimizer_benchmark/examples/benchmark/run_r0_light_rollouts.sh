#!/usr/bin/env bash
# Parallel R0a (T=0.7) + R0b (T=1.0) REINFORCE rollout collection on light-16 tasks.
# Pure LLM proposer (ptllm_qwen) from frozen s1a_ep7p5; 40 trials; reward abs=0.1.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/light_16.json}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_rollouts_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$TASKS_JSON" ]]; then
  echo "Missing tasks: $TASKS_JSON" >&2
  exit 1
fi

declare -a QUEUE=()
# arm|temp|outdir|tt|ds|algo
while IFS= read -r line; do
  QUEUE+=("$line")
done < <("$PY" - <<'PY'
import json
from pathlib import Path
tasks = json.loads(Path("benchmark_data/rl_task_selection/light_16.json").read_text())
arms = [
    ("R0a", 0.7, "benchmark_data/rl_rollouts/R0a_T0p7_light16"),
    ("R0b", 1.0, "benchmark_data/rl_rollouts/R0b_T1p0_light16"),
]
# Interleave arms so both temps collect in parallel
for t in tasks:
    for arm, temp, out in arms:
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 light rollouts: $TOTAL jobs (16 tasks × 2 temps) GPUs=$NUM_GPUS jobs/gpu=$JOBS_PER_GPU ==="
log "adapter=$ADAPTER n_trials=$N_TRIALS seed=$SEED"

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

# Track pids per GPU + global list for final wait
declare -a GPU_PIDS=()
declare -a ALL_PIDS=()
for ((g=0; g<NUM_GPUS; g++)); do GPU_PIDS[g]=""; done

count_live() {
  local g="$1" live=0
  for pid in ${GPU_PIDS[$g]:-}; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live+1)); fi
  done
  echo "$live"
}

prune_pids() {
  local g="$1" kept=""
  for pid in ${GPU_PIDS[$g]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$g]="${kept% }"
}

run_one() {
  local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
  local outdir="$outroot/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_${TS}.log"
  mkdir -p "$outdir"
  log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo -> $outdir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
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
  if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then
    GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]} $pid"
  else
    GPU_PIDS[$gpu]="$pid"
  fi
}

idx=0
while (( idx < TOTAL )); do
  launched=0
  for ((g=0; g<NUM_GPUS; g++)); do
    prune_pids "$g"
    live=$(count_live "$g")
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < JOBS_PER_GPU && idx < TOTAL )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
      idx=$((idx+1))
      run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
      launched=1
      live=$(count_live "$g")
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

# Summary counts
"$PY" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path
from collections import Counter
roots = [
    Path("benchmark_data/rl_rollouts/R0a_T0p7_light16"),
    Path("benchmark_data/rl_rollouts/R0b_T1p0_light16"),
]
for root in roots:
    n_buf = 0
    n_rec = 0
    tasks_done = 0
    if root.exists():
        for p in root.rglob("rollouts_*.jsonl"):
            n_buf += 1
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            n_rec += len(lines)
            if len(lines) >= 40:
                tasks_done += 1
            elif len(lines) > 0:
                tasks_done += 0
    print(f"{root.name}: buffers={n_buf} records={n_rec} tasks_with_40plus={tasks_done}")
PY

if (( fail )); then
  log "=== R0 light rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== R0 light rollouts complete ==="
#!/usr/bin/env bash
# Parallel R0a (T=0.7) + R0b (T=1.0) REINFORCE rollout collection on light-16 tasks.
# Pure LLM proposer (ptllm_qwen) from frozen s1a_ep7p5; 40 trials; reward abs=0.1.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/light_16.json}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_rollouts_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

if [[ ! -f "$ADAPTER/adapter_model.safetensors" ]]; then
  echo "Missing adapter: $ADAPTER/adapter_model.safetensors" >&2
  exit 1
fi
if [[ ! -f "$TASKS_JSON" ]]; then
  echo "Missing tasks: $TASKS_JSON" >&2
  exit 1
fi

declare -a QUEUE=()
# arm|temp|outdir|tt|ds|algo
while IFS= read -r line; do
  QUEUE+=("$line")
done < <("$PY" - <<'PY'
import json
from pathlib import Path
tasks = json.loads(Path("benchmark_data/rl_task_selection/light_16.json").read_text())
arms = [
    ("R0a", 0.7, "benchmark_data/rl_rollouts/R0a_T0p7_light16"),
    ("R0b", 1.0, "benchmark_data/rl_rollouts/R0b_T1p0_light16"),
]
# Interleave arms so both temps collect in parallel
for t in tasks:
    for arm, temp, out in arms:
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 light rollouts: $TOTAL jobs (16 tasks × 2 temps) GPUs=$NUM_GPUS jobs/gpu=$JOBS_PER_GPU ==="
log "adapter=$ADAPTER n_trials=$N_TRIALS seed=$SEED"

gpu_free_mb() {
  local g="$1"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' '
}

# Track pids per GPU + global list for final wait
declare -a GPU_PIDS=()
declare -a ALL_PIDS=()
for ((g=0; g<NUM_GPUS; g++)); do GPU_PIDS[g]=""; done

count_live() {
  local g="$1" live=0
  for pid in ${GPU_PIDS[$g]:-}; do
    if kill -0 "$pid" 2>/dev/null; then live=$((live+1)); fi
  done
  echo "$live"
}

prune_pids() {
  local g="$1" kept=""
  for pid in ${GPU_PIDS[$g]:-}; do
    if kill -0 "$pid" 2>/dev/null; then kept+="$pid "; fi
  done
  GPU_PIDS[$g]="${kept% }"
}

run_one() {
  local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
  local outdir="$outroot/${tt}_${ds}_${algo}"
  local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_${TS}.log"
  mkdir -p "$outdir"
  log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo -> $outdir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
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
  if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then
    GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]} $pid"
  else
    GPU_PIDS[$gpu]="$pid"
  fi
}

idx=0
while (( idx < TOTAL )); do
  launched=0
  for ((g=0; g<NUM_GPUS; g++)); do
    prune_pids "$g"
    live=$(count_live "$g")
    free=$(gpu_free_mb "$g" || echo 0)
    while (( live < JOBS_PER_GPU && idx < TOTAL )); do
      if (( free < PACK_MIN_FREE_MB && live > 0 )); then
        break
      fi
      IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
      idx=$((idx+1))
      run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
      launched=1
      live=$(count_live "$g")
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

# Summary counts
"$PY" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path
from collections import Counter
roots = [
    Path("benchmark_data/rl_rollouts/R0a_T0p7_light16"),
    Path("benchmark_data/rl_rollouts/R0b_T1p0_light16"),
]
for root in roots:
    n_buf = 0
    n_rec = 0
    tasks_done = 0
    if root.exists():
        for p in root.rglob("rollouts_*.jsonl"):
            n_buf += 1
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            n_rec += len(lines)
            if len(lines) >= 40:
                tasks_done += 1
            elif len(lines) > 0:
                tasks_done += 0
    print(f"{root.name}: buffers={n_buf} records={n_rec} tasks_with_40plus={tasks_done}")
PY

if (( fail )); then
  log "=== R0 light rollouts finished WITH FAILURES ==="
  exit 1
fi
log "=== R0 light rollouts complete ==="
