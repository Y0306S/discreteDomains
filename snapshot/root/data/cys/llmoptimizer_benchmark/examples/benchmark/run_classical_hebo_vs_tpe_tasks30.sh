#!/usr/bin/env bash
# Pure classical TPE vs HEBO on the 30 ranking tasks.
# TPE already has 80-trial trajs (resume skip). HEBO is new.
# Pack 3 CPU workers per GPU on GPUs 0-1 (6 concurrent jobs).
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_30.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40/classical}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
# HEBO needs gpytorch; openbox_qwen_eval already has HEBO 0.3.6.
PYTHON="${PYTHON:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-0,1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-2000}"
POLL_SEC="${POLL_SEC:-10}"
EVAL_RESUME="${EVAL_RESUME:-1}"
OPTIMIZERS="${OPTIMIZERS:-hebo}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/classical_hebo_vs_tpe_tasks30_${TS}.log"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

mapfile -t TASK_LINES < <("$PYTHON" - <<PY
import json
from pathlib import Path
tasks=json.loads(Path("$TASKS_JSON").read_text())
for t in tasks:
    print(f"{t['task_type']}|{t['dataset']}|{t['algorithm']}")
print(f"# n={len(tasks)}", flush=True)
PY
)
TASKS=()
for line in "${TASK_LINES[@]}"; do
  [[ "$line" == \#* ]] && continue
  TASKS+=("$line")
done

IFS=',' read -ra GPU_IDS <<<"$EVAL_GPU_MAP"
NUM_GPUS="${#GPU_IDS[@]}"

log "=== classical HEBO vs TPE tasks30 ts=$TS ==="
log "GPUs ${EVAL_GPU_MAP} × ${JOBS_PER_GPU} workers | tasks=${#TASKS[@]} | opts=${OPTIMIZERS}"
log "python=$PYTHON output=$OUTPUT_ROOT"

RESUME_FLAG=()
[[ "$EVAL_RESUME" == "1" ]] && RESUME_FLAG=(--resume)

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

jobs=()
read -r -a _opts <<<"$OPTIMIZERS"
for t in "${TASKS[@]}"; do
  IFS='|' read -r tt ds algo <<<"$t"
  for opt in "${_opts[@]}"; do
    jobs+=("$tt|$ds|$algo|$opt")
  done
done
log "queue: ${#jobs[@]} jobs"

run_one() {
  local tt="$1" ds="$2" algo="$3" opt="$4" pdev="$5"
  local logf="$LOGDIR/classical_${opt}_${ds}_${TS}.log"
  log "START gpu=$pdev $opt $tt/$ds/$algo"
  CUDA_VISIBLE_DEVICES="$pdev" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
  OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}" \
  "$PYTHON" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval \
    --optimizer "$opt" \
    "${RESUME_FLAG[@]}" \
    --output-dir "$OUTPUT_ROOT" \
    --all-datasets --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" --dataset-retries "$DATASET_RETRIES" \
    >>"$logf" 2>&1
  log "DONE gpu=$pdev $opt $tt/$ds/$algo"
}

declare -A SLOT_PID=()
queue_idx=0
total=${#jobs[@]}

while [[ $queue_idx -lt $total ]] || {
    any=0
    for ((g=0; g<NUM_GPUS; g++)); do
      for ((s=0; s<JOBS_PER_GPU; s++)); do
        is_alive "${SLOT_PID[${g}_${s}]:-}" && any=1
      done
    done
    [[ $any -eq 1 ]]
  }; do
  for ((g=0; g<NUM_GPUS; g++)); do
    pdev="${GPU_IDS[$g]}"
    for ((s=0; s<JOBS_PER_GPU; s++)); do
      pid="${SLOT_PID[${g}_${s}]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[${g}_${s}]=
      [[ $queue_idx -ge $total ]] && continue
      count=0
      for ((k=0; k<JOBS_PER_GPU; k++)); do
        is_alive "${SLOT_PID[${g}_${k}]:-}" && count=$((count+1))
      done
      [[ $count -lt $JOBS_PER_GPU ]] || continue
      if [[ $count -ge 1 ]]; then
        free=$(gpu_free_mb "$pdev")
        [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
      fi
      job="${jobs[$queue_idx]}"
      queue_idx=$((queue_idx+1))
      IFS='|' read -r tt ds algo opt <<<"$job"
      run_one "$tt" "$ds" "$algo" "$opt" "$pdev" &
      SLOT_PID[${g}_${s}]=$!
      log "packed $queue_idx/$total -> gpu $pdev slot $s pid=${SLOT_PID[${g}_${s}]} $opt $ds"
    done
  done
  sleep "$POLL_SEC"
done

log "=== classical HEBO vs TPE tasks30 done ==="
