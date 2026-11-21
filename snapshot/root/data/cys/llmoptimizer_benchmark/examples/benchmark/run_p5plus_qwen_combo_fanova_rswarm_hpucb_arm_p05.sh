#!/usr/bin/env bash
# Rectified combo on Qwen LoRA s1b_attr_t2 (does not overwrite kitchen-sink TRdyFanovaX50RS):
#   prompt  = P5plusA5TRdyFanovaX50RSWarm
#             proxy Fanova + X50 + RS similar-task retrieve ONLY until trial 50
#   hybrid  = HPUCB_arm_p05
#             PUCB arm-best Q, LLM ω 0.9→0.5, TPE ω 0.1→0.5
#
# Default packing leaves GPU 0 free: 3 GPUs × 3 workers on physical 1,2,3.
#
#   bash examples/benchmark/run_p5plus_qwen_combo_fanova_rswarm_hpucb_arm_p05.sh
#   NUM_GPUS=3 JOBS_PER_GPU=3 EVAL_GPU_MAP=1,2,3 bash ...
#   DRY_RUN=1 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
PROMPT_VARIANT="${PROMPT_VARIANT:-P5plusA5TRdyFanovaX50RSWarm}"
HYBRID_VARIANT="${HYBRID_VARIANT:-HPUCB_arm_p05}"
OPT_BASE="${OPT_BASE:-ptllm_tpe_qwen}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-20}"
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-1,2,3}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TAG="${OPT_BASE}__${PROMPT_VARIANT}__${HYBRID_VARIANT}"
MASTER_LOG="$LOGDIR/qwen_combo_fanova_rswarm_hpucb_arm_p05_${TS}.log"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  log "ERROR missing adapter $ADAPTER_PATH/adapter_model.safetensors"
  exit 1
fi

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

if [[ "$DRY_RUN" == "1" ]]; then
  N_TRIALS=6
  TIME_LIMIT=300
  TASKS=("cls|yeast|lightgbm")
  log "DRY_RUN: n_trials=$N_TRIALS single task"
fi

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

physical_gpu_for_slot() {
  local slot="$1"
  if [[ -n "$EVAL_GPU_MAP" ]]; then
    IFS=',' read -ra _map <<<"$EVAL_GPU_MAP"
    echo "${_map[$slot]:-$slot}"
  else
    echo "$slot"
  fi
}

run_one_task() {
  local tt="$1" ds="$2" algo="$3" gpu="$4"
  local pdev
  pdev="$(physical_gpu_for_slot "$gpu")"
  local logf="$LOGDIR/qwen_combo_${TAG}_${ds}_${algo}_${TS}.log"
  local runner=( "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$algo"
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval
    --llm-backends qwen-sft6 --qwen-backend hf
    --qwen-model-path "$BASE_MODEL"
    --qwen-adapter-path "$ADAPTER_PATH"
    --llm-max-tokens 768
    --llm-response-format rationale_json
    --output-dir "$OUTPUT_ROOT"
    --all-datasets --include-datasets "$ds"
    --time-limit "$TIME_LIMIT"
    --per-run-time-limit "$PER_RUN_TIME_LIMIT"
    --dataset-retries "$DATASET_RETRIES"
    --optimizer "$OPT_BASE"
    --prompt-variant "$PROMPT_VARIANT"
    --hybrid-variant "$HYBRID_VARIANT"
    "${RESUME_FLAG[@]}"
  )
  [[ "$DRY_RUN" == "1" ]] && runner+=( --llm-mock )
  log "  gpu=$gpu (phys $pdev) $TAG $tt/$ds/$algo"
  CUDA_VISIBLE_DEVICES="$pdev" "${runner[@]}" >>"$logf" 2>&1
}

run_gpu_pool() {
  local -n _jobs=$1
  declare -A SLOT_PID=()
  local queue_idx=0 total=${#_jobs[@]}
  local gpu slot pid job tt ds algo

  log "queue: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU map=$EVAL_GPU_MAP"

  while [[ $queue_idx -lt $total ]] || {
      local any=0
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
        local count=0 s
        for ((s=0; s<JOBS_PER_GPU; s++)); do
          is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
        done
        [[ $count -lt $JOBS_PER_GPU ]] || continue
        if [[ $count -ge 1 ]]; then
          local free pdev
          pdev="$(physical_gpu_for_slot "$gpu")"
          free=$(gpu_free_mb "$pdev")
          [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
        fi
        job="${_jobs[$queue_idx]}"
        queue_idx=$((queue_idx+1))
        IFS='|' read -r tt ds algo <<<"$job"
        log "dispatch $queue_idx/$total gpu=$gpu phys=$(physical_gpu_for_slot "$gpu") $ds/$algo"
        ( run_one_task "$tt" "$ds" "$algo" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
}

"$PYTHON_LLM" - <<PY
from argparse import Namespace
from experiment_utils import (
    load_prompt_variants,
    _resolve_prompt_variant,
    load_hybrid_variants,
    apply_prompt_settings,
    apply_hybrid_settings,
)
pv = load_prompt_variants()
row = _resolve_prompt_variant(pv["$PROMPT_VARIANT"], pv)
assert row.get("experience_self_gate") == "fanova", row.get("experience_self_gate")
assert row.get("experience_retrieve_ranker") == "surrogate", row.get("experience_retrieve_ranker")
assert bool(row.get("experience_retrieve_until_self")) is True
assert row.get("prompt_blocks", {}).get("experience_retrieve") is True
assert row.get("prompt_blocks", {}).get("experience_self") is True
args = Namespace(llm_history_window=80)
apply_prompt_settings(args, "$PROMPT_VARIANT")
assert args.llm_experience_retrieve_stop_trial == 50, args.llm_experience_retrieve_stop_trial
assert args.llm_experience_self_gate == "fanova"
hv = load_hybrid_variants()
assert "$HYBRID_VARIANT" in hv
hrow = hv["$HYBRID_VARIANT"]
assert hrow.get("hybrid_selector") == "pucb"
assert float(hrow.get("pucb_p1")) == 0.9
assert float(hrow.get("pucb_p_end")) == 0.5
assert hrow.get("pucb_q") == "arm_best"
apply_hybrid_settings(args, "$HYBRID_VARIANT")
assert args.pucb_p_end == 0.5
print("ok", "$PROMPT_VARIANT", "gate=fanova", "rs_stop", args.llm_experience_retrieve_stop_trial)
print("ok hybrid", "$HYBRID_VARIANT", "pucb", "p1", args.pucb_p1, "p_end", args.pucb_p_end)
PY

jobs=()
for t in "${TASKS[@]}"; do
  jobs+=("$t")
done

log "=== Qwen combo proxy-Fanova + RSWarm + HPUCB_arm_p05 ==="
log "tag=$TAG"
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU map=$EVAL_GPU_MAP"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "PUCB LLM 0.9→0.5  TPE 0.1→0.5  RS retrieve until trial 50  proxy Fanova F>=0.10"

run_gpu_pool jobs

log "=== Qwen combo proxy-Fanova + RSWarm + HPUCB_arm_p05 done ==="
