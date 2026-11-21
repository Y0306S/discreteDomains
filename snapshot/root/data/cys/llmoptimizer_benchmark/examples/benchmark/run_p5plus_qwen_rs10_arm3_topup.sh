#!/usr/bin/env bash
# Top-up arm 3 only: P5plusA5T50X50RS10 + HPUCB_arm_p07 (17 failed tasks).
# Failed at launch due to transient hybrid_variants.json parse error (now fixed).
#
#   bash examples/benchmark/run_p5plus_qwen_rs10_arm3_topup.sh
#   NUM_GPUS=1 EVAL_GPU_MAP=0 bash examples/benchmark/run_p5plus_qwen_rs10_arm3_topup.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_rs10_arm3_topup_17.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
OPT_BASE="${OPT_BASE:-ptllm_tpe_qwen}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-1,2,3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-20}"
EVAL_RESUME="${EVAL_RESUME:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_rs10_arm3_topup_${TS}.log"
PROMPT="P5plusA5T50X50RS10"
HYBRID="HPUCB_arm_p07"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

"$PYTHON_LLM" - <<PY
from experiment_utils import load_hybrid_variants, apply_hybrid_settings
from argparse import Namespace
args = Namespace(llm_history_window=80)
apply_hybrid_settings(args, "HPUCB_arm_p07")
print("ok HPUCB_arm_p07")
PY

mapfile -t TASK_LINES < <("$PYTHON_LLM" - <<PY
import json
from pathlib import Path
tasks = json.loads(Path("$TASKS_JSON").read_text())
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

RESUME_FLAG=()
[[ "$EVAL_RESUME" == "1" ]] && RESUME_FLAG=(--resume)

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

physical_gpu_for_slot() {
  local slot="$1"
  IFS=',' read -ra _map <<<"$EVAL_GPU_MAP"
  echo "${_map[$slot]:-$slot}"
}

run_one_task() {
  local tt="$1" ds="$2" algo="$3" gpu="$4"
  local pdev
  pdev="$(physical_gpu_for_slot "$gpu")"
  local tag="${OPT_BASE}__${PROMPT}__${HYBRID}"
  local logf="$LOGDIR/qwen_${tag}_${ds}_${algo}_topup_${TS}.log"
  log "  gpu=$gpu (phys $pdev) $tag $tt/$ds/$algo"
  CUDA_VISIBLE_DEVICES="$pdev" "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval \
    --llm-backends qwen-sft6 --qwen-backend hf \
    --qwen-model-path "$BASE_MODEL" \
    --qwen-adapter-path "$ADAPTER_PATH" \
    --llm-max-tokens 768 \
    --llm-response-format rationale_json \
    --output-dir "$OUTPUT_ROOT" \
    --all-datasets --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" \
    --per-run-time-limit "$PER_RUN_TIME_LIMIT" \
    --dataset-retries "$DATASET_RETRIES" \
    --optimizer "$OPT_BASE" \
    --prompt-variant "$PROMPT" \
    --hybrid-variant "$HYBRID" \
    "${RESUME_FLAG[@]}" >>"$logf" 2>&1
}

declare -A SLOT_PID=()
queue_idx=0
total=${#TASKS[@]}

log "=== RS10 arm3 top-up ==="
log "tasks=$total gpus=$NUM_GPUS (phys $EVAL_GPU_MAP) jobs_per_gpu=$JOBS_PER_GPU hybrid=$HYBRID"

while [[ $queue_idx -lt $total ]] || {
    any=0
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
        is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && any=1
      done
    done
    [[ $any -eq 1 ]]
  }; do
  for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
      pid="${SLOT_PID[${gpu}_${slot}]:-}"
      if is_alive "$pid"; then continue; fi
      SLOT_PID[${gpu}_${slot}]=
      [[ $queue_idx -ge $total ]] && continue
      count=0
      for ((s=0; s<JOBS_PER_GPU; s++)); do
        is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
      done
      [[ $count -lt $JOBS_PER_GPU ]] || continue
      if [[ $count -ge 1 ]]; then
        pdev="$(physical_gpu_for_slot "$gpu")"
        free=$(gpu_free_mb "$pdev")
        [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
      fi
      line="${TASKS[$queue_idx]}"
      queue_idx=$((queue_idx+1))
      IFS='|' read -r tt ds algo <<<"$line"
      log "dispatch $queue_idx/$total gpu=$gpu $ds/$algo"
      ( run_one_task "$tt" "$ds" "$algo" "$gpu" ) &
      SLOT_PID[${gpu}_${slot}]=$!
    done
  done
  sleep "$POLL_SEC"
done
wait
log "=== RS10 arm3 top-up done ($total tasks) ==="
