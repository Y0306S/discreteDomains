#!/usr/bin/env bash
# Resume failed R0 light RGS rollouts (metric mse fix), then wait→train→eval.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_rgs_resume_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

declare -a QUEUE=()
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
for t in tasks:
    if t["task_type"] != "rgs":
        continue
    for arm, temp, out in arms:
        d = Path(out) / f"{t['task_type']}_{t['dataset']}_{t['algorithm']}"
        n = 0
        if d.exists():
            for p in d.rglob("rollouts_*.jsonl"):
                n += sum(1 for ln in p.open() if ln.strip())
        if n >= 40:
            continue
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 RGS resume: $TOTAL missing jobs (mse metric) ==="
if (( TOTAL == 0 )); then
  log "Nothing to resume"
else
  declare -a GPU_PIDS=() ALL_PIDS=()
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
  gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
  }

  run_one() {
    local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
    local outdir="$outroot/${tt}_${ds}_${algo}"
    local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_resume_${TS}.log"
    mkdir -p "$outdir"
    log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
      --task-type "$tt" --dataset "$ds" --algorithm "$algo" --metric mse \
      --n-trials "$N_TRIALS" --seed "$SEED" --time-limit "$TIME_LIMIT" \
      --output-dir "$outdir" --prompt-variant P5 --hybrid-variant H0 \
      --qwen-backend hf --qwen-model-path "$MODEL_PATH" --qwen-adapter-path "$ADAPTER" \
      --llm-response-format json_only --llm-max-tokens 768 --llm-temperature "$temp" \
      --reward-delta-best 1.0 --reward-absolute 0.1 \
      --penalty-parse-fail 0.5 --penalty-duplicate 0.1 --penalty-eval-fail 1.0 \
      --resume >>"$logfile" 2>&1 &
    local pid=$!
    ALL_PIDS+=("$pid")
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]} $pid"; else GPU_PIDS[$gpu]="$pid"; fi
  }

  idx=0
  while (( idx < TOTAL )); do
    launched=0
    for ((g=0; g<NUM_GPUS; g++)); do
      prune_pids "$g"
      live=$(count_live "$g")
      free=$(gpu_free_mb "$g" || echo 0)
      while (( live < JOBS_PER_GPU && idx < TOTAL )); do
        if (( free < PACK_MIN_FREE_MB && live > 0 )); then break; fi
        IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
        idx=$((idx+1))
        run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
        launched=1
        live=$(count_live "$g")
        free=$(gpu_free_mb "$g" || echo 0)
        sleep 2
      done
    done
    if (( launched == 0 )); then sleep 20; fi
  done

  log "Waiting for ${#ALL_PIDS[@]} RGS resume jobs"
  fail=0
  for pid in "${ALL_PIDS[@]}"; do
    if ! wait "$pid"; then log "FAIL pid=$pid"; fail=1; fi
  done
  if (( fail )); then
    log "=== RGS resume finished WITH FAILURES ==="
    exit 1
  fi
  log "=== RGS resume complete ==="
fi

# Hand off to train+eval waiter (expect ~640 records/arm)
export MIN_RECORDS_PER_ARM="${MIN_RECORDS_PER_ARM:-600}"
log "=== starting wait_r0_light_train_eval (MIN_RECORDS=$MIN_RECORDS_PER_ARM) ==="
exec bash examples/benchmark/wait_r0_light_train_eval.sh
#!/usr/bin/env bash
# Resume failed R0 light RGS rollouts (metric mse fix), then wait→train→eval.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_rgs_resume_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

declare -a QUEUE=()
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
for t in tasks:
    if t["task_type"] != "rgs":
        continue
    for arm, temp, out in arms:
        d = Path(out) / f"{t['task_type']}_{t['dataset']}_{t['algorithm']}"
        n = 0
        if d.exists():
            for p in d.rglob("rollouts_*.jsonl"):
                n += sum(1 for ln in p.open() if ln.strip())
        if n >= 40:
            continue
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 RGS resume: $TOTAL missing jobs (mse metric) ==="
if (( TOTAL == 0 )); then
  log "Nothing to resume"
else
  declare -a GPU_PIDS=() ALL_PIDS=()
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
  gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
  }

  run_one() {
    local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
    local outdir="$outroot/${tt}_${ds}_${algo}"
    local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_resume_${TS}.log"
    mkdir -p "$outdir"
    log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
      --task-type "$tt" --dataset "$ds" --algorithm "$algo" --metric mse \
      --n-trials "$N_TRIALS" --seed "$SEED" --time-limit "$TIME_LIMIT" \
      --output-dir "$outdir" --prompt-variant P5 --hybrid-variant H0 \
      --qwen-backend hf --qwen-model-path "$MODEL_PATH" --qwen-adapter-path "$ADAPTER" \
      --llm-response-format json_only --llm-max-tokens 768 --llm-temperature "$temp" \
      --reward-delta-best 1.0 --reward-absolute 0.1 \
      --penalty-parse-fail 0.5 --penalty-duplicate 0.1 --penalty-eval-fail 1.0 \
      --resume >>"$logfile" 2>&1 &
    local pid=$!
    ALL_PIDS+=("$pid")
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]} $pid"; else GPU_PIDS[$gpu]="$pid"; fi
  }

  idx=0
  while (( idx < TOTAL )); do
    launched=0
    for ((g=0; g<NUM_GPUS; g++)); do
      prune_pids "$g"
      live=$(count_live "$g")
      free=$(gpu_free_mb "$g" || echo 0)
      while (( live < JOBS_PER_GPU && idx < TOTAL )); do
        if (( free < PACK_MIN_FREE_MB && live > 0 )); then break; fi
        IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
        idx=$((idx+1))
        run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
        launched=1
        live=$(count_live "$g")
        free=$(gpu_free_mb "$g" || echo 0)
        sleep 2
      done
    done
    if (( launched == 0 )); then sleep 20; fi
  done

  log "Waiting for ${#ALL_PIDS[@]} RGS resume jobs"
  fail=0
  for pid in "${ALL_PIDS[@]}"; do
    if ! wait "$pid"; then log "FAIL pid=$pid"; fail=1; fi
  done
  if (( fail )); then
    log "=== RGS resume finished WITH FAILURES ==="
    exit 1
  fi
  log "=== RGS resume complete ==="
fi

# Hand off to train+eval waiter (expect ~640 records/arm)
export MIN_RECORDS_PER_ARM="${MIN_RECORDS_PER_ARM:-600}"
log "=== starting wait_r0_light_train_eval (MIN_RECORDS=$MIN_RECORDS_PER_ARM) ==="
exec bash examples/benchmark/wait_r0_light_train_eval.sh
#!/usr/bin/env bash
# Resume failed R0 light RGS rollouts (metric mse fix), then wait→train→eval.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_rgs_resume_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

declare -a QUEUE=()
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
for t in tasks:
    if t["task_type"] != "rgs":
        continue
    for arm, temp, out in arms:
        d = Path(out) / f"{t['task_type']}_{t['dataset']}_{t['algorithm']}"
        n = 0
        if d.exists():
            for p in d.rglob("rollouts_*.jsonl"):
                n += sum(1 for ln in p.open() if ln.strip())
        if n >= 40:
            continue
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 RGS resume: $TOTAL missing jobs (mse metric) ==="
if (( TOTAL == 0 )); then
  log "Nothing to resume"
else
  declare -a GPU_PIDS=() ALL_PIDS=()
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
  gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
  }

  run_one() {
    local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
    local outdir="$outroot/${tt}_${ds}_${algo}"
    local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_resume_${TS}.log"
    mkdir -p "$outdir"
    log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
      --task-type "$tt" --dataset "$ds" --algorithm "$algo" --metric mse \
      --n-trials "$N_TRIALS" --seed "$SEED" --time-limit "$TIME_LIMIT" \
      --output-dir "$outdir" --prompt-variant P5 --hybrid-variant H0 \
      --qwen-backend hf --qwen-model-path "$MODEL_PATH" --qwen-adapter-path "$ADAPTER" \
      --llm-response-format json_only --llm-max-tokens 768 --llm-temperature "$temp" \
      --reward-delta-best 1.0 --reward-absolute 0.1 \
      --penalty-parse-fail 0.5 --penalty-duplicate 0.1 --penalty-eval-fail 1.0 \
      --resume >>"$logfile" 2>&1 &
    local pid=$!
    ALL_PIDS+=("$pid")
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]} $pid"; else GPU_PIDS[$gpu]="$pid"; fi
  }

  idx=0
  while (( idx < TOTAL )); do
    launched=0
    for ((g=0; g<NUM_GPUS; g++)); do
      prune_pids "$g"
      live=$(count_live "$g")
      free=$(gpu_free_mb "$g" || echo 0)
      while (( live < JOBS_PER_GPU && idx < TOTAL )); do
        if (( free < PACK_MIN_FREE_MB && live > 0 )); then break; fi
        IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
        idx=$((idx+1))
        run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
        launched=1
        live=$(count_live "$g")
        free=$(gpu_free_mb "$g" || echo 0)
        sleep 2
      done
    done
    if (( launched == 0 )); then sleep 20; fi
  done

  log "Waiting for ${#ALL_PIDS[@]} RGS resume jobs"
  fail=0
  for pid in "${ALL_PIDS[@]}"; do
    if ! wait "$pid"; then log "FAIL pid=$pid"; fail=1; fi
  done
  if (( fail )); then
    log "=== RGS resume finished WITH FAILURES ==="
    exit 1
  fi
  log "=== RGS resume complete ==="
fi

# Hand off to train+eval waiter (expect ~640 records/arm)
export MIN_RECORDS_PER_ARM="${MIN_RECORDS_PER_ARM:-600}"
log "=== starting wait_r0_light_train_eval (MIN_RECORDS=$MIN_RECORDS_PER_ARM) ==="
exec bash examples/benchmark/wait_r0_light_train_eval.sh
#!/usr/bin/env bash
# Resume failed R0 light RGS rollouts (metric mse fix), then wait→train→eval.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

EVAL_ENV="${EVAL_ENV:-/root/miniconda3/envs/openbox_qwen_eval}"
PY="${PY:-$EVAL_ENV/bin/python}"
ADAPTER="${ADAPTER:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
N_TRIALS="${N_TRIALS:-40}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
MODEL_PATH="${MODEL_PATH:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/r0_light_rgs_resume_${TS}.log"

mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

declare -a QUEUE=()
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
for t in tasks:
    if t["task_type"] != "rgs":
        continue
    for arm, temp, out in arms:
        d = Path(out) / f"{t['task_type']}_{t['dataset']}_{t['algorithm']}"
        n = 0
        if d.exists():
            for p in d.rglob("rollouts_*.jsonl"):
                n += sum(1 for ln in p.open() if ln.strip())
        if n >= 40:
            continue
        print(f"{arm}|{temp}|{out}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
PY
)

TOTAL=${#QUEUE[@]}
log "=== R0 RGS resume: $TOTAL missing jobs (mse metric) ==="
if (( TOTAL == 0 )); then
  log "Nothing to resume"
else
  declare -a GPU_PIDS=() ALL_PIDS=()
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
  gpu_free_mb() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | tr -d ' '
  }

  run_one() {
    local gpu="$1" arm="$2" temp="$3" outroot="$4" tt="$5" ds="$6" algo="$7"
    local outdir="$outroot/${tt}_${ds}_${algo}"
    local logfile="$LOG_ROOT/r0_${arm}_${tt}_${ds}_${algo}_resume_${TS}.log"
    mkdir -p "$outdir"
    log "START gpu=$gpu arm=$arm T=$temp $tt/$ds/$algo"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" examples/benchmark/run_p5_rl_rollout_buffer.py \
      --task-type "$tt" --dataset "$ds" --algorithm "$algo" --metric mse \
      --n-trials "$N_TRIALS" --seed "$SEED" --time-limit "$TIME_LIMIT" \
      --output-dir "$outdir" --prompt-variant P5 --hybrid-variant H0 \
      --qwen-backend hf --qwen-model-path "$MODEL_PATH" --qwen-adapter-path "$ADAPTER" \
      --llm-response-format json_only --llm-max-tokens 768 --llm-temperature "$temp" \
      --reward-delta-best 1.0 --reward-absolute 0.1 \
      --penalty-parse-fail 0.5 --penalty-duplicate 0.1 --penalty-eval-fail 1.0 \
      --resume >>"$logfile" 2>&1 &
    local pid=$!
    ALL_PIDS+=("$pid")
    if [[ -n "${GPU_PIDS[$gpu]:-}" ]]; then GPU_PIDS[$gpu]="${GPU_PIDS[$gpu]} $pid"; else GPU_PIDS[$gpu]="$pid"; fi
  }

  idx=0
  while (( idx < TOTAL )); do
    launched=0
    for ((g=0; g<NUM_GPUS; g++)); do
      prune_pids "$g"
      live=$(count_live "$g")
      free=$(gpu_free_mb "$g" || echo 0)
      while (( live < JOBS_PER_GPU && idx < TOTAL )); do
        if (( free < PACK_MIN_FREE_MB && live > 0 )); then break; fi
        IFS='|' read -r arm temp outroot tt ds algo <<<"${QUEUE[$idx]}"
        idx=$((idx+1))
        run_one "$g" "$arm" "$temp" "$outroot" "$tt" "$ds" "$algo"
        launched=1
        live=$(count_live "$g")
        free=$(gpu_free_mb "$g" || echo 0)
        sleep 2
      done
    done
    if (( launched == 0 )); then sleep 20; fi
  done

  log "Waiting for ${#ALL_PIDS[@]} RGS resume jobs"
  fail=0
  for pid in "${ALL_PIDS[@]}"; do
    if ! wait "$pid"; then log "FAIL pid=$pid"; fail=1; fi
  done
  if (( fail )); then
    log "=== RGS resume finished WITH FAILURES ==="
    exit 1
  fi
  log "=== RGS resume complete ==="
fi

# Hand off to train+eval waiter (expect ~640 records/arm)
export MIN_RECORDS_PER_ARM="${MIN_RECORDS_PER_ARM:-600}"
log "=== starting wait_r0_light_train_eval (MIN_RECORDS=$MIN_RECORDS_PER_ARM) ==="
exec bash examples/benchmark/wait_r0_light_train_eval.sh
