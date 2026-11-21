#!/usr/bin/env bash
# After the current Qwen XTPE eval finishes, run two sequential 40-task arms
# on the same model / eval set as the senior report:
#   1) P5plusA5T50X50 + HPUCB_arm_p05
#        PUCB arm-best Q, LLM ω 0.9→0.5, TPE ω 0.1→0.5
#   2) P5plusA5T50X50RSWarm + HPUCB_arm_p05
#        same PUCB + RS similar-task retrieve only until trial 50 (P5+ this-run start)
#
#   bash examples/benchmark/run_p5plus_qwen_hpucb_p05_and_rswarm.sh
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
OPT_BASE="${OPT_BASE:-ptllm_tpe_qwen}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-20}"
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-60}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_hpucb_p05_rswarm_${TS}.log"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  log "ERROR missing adapter $ADAPTER_PATH/adapter_model.safetensors"
  exit 1
fi

"$PYTHON_LLM" - <<PY
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, load_hybrid_variants, apply_prompt_settings, apply_hybrid_settings
from argparse import Namespace

pv = load_prompt_variants()
x50 = _resolve_prompt_variant(pv["P5plusA5T50X50"], pv)
assert (x50.get("experience_explore_mode") or "off") == "fifty_fifty"
assert int(x50.get("experience_self_start_trial") or 0) == 50

warm = _resolve_prompt_variant(pv["P5plusA5T50X50RSWarm"], pv)
assert warm.get("prompt_blocks", {}).get("experience_retrieve") is True
assert warm.get("experience_retrieve_ranker") == "surrogate"
assert bool(warm.get("experience_retrieve_until_self")) is True

args = Namespace(llm_history_window=80)
apply_prompt_settings(args, "P5plusA5T50X50RSWarm")
assert args.llm_experience_retrieve_stop_trial == 50, args.llm_experience_retrieve_stop_trial

hv = load_hybrid_variants()
row = hv["HPUCB_arm_p05"]
assert row["hybrid_selector"] == "pucb"
assert float(row["pucb_p1"]) == 0.9
assert float(row["pucb_p_end"]) == 0.5
assert row["pucb_q"] == "arm_best"
apply_hybrid_settings(args, "HPUCB_arm_p05")
assert args.pucb_p_end == 0.5
print("ok P5plusA5T50X50 / P5plusA5T50X50RSWarm / HPUCB_arm_p05")
print("ok RSWarm stop_trial", args.llm_experience_retrieve_stop_trial)
PY

xtpe_still_running() {
  pgrep -f '[r]un_p5plus_qwen_fanova_full_and_tpe_xor.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5TRdyFanovaFullX50' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5T50XTPE' >/dev/null 2>&1 && return 0
  return 1
}

if [[ "$DRY_RUN" != "1" ]] && xtpe_still_running; then
  log "waiting for current FanovaFull/XTPE eval to finish"
  while xtpe_still_running; do
    n_py=$(pgrep -cf '[P]5plusA5T50XTPE' || true)
    log "  still running XTPE python jobs≈${n_py}"
    sleep "$WAIT_POLL_SEC"
  done
  log "XTPE eval finished — starting HPUCB_arm_p05 + RSWarm"
else
  log "XTPE eval not running (or DRY_RUN) — starting immediately"
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

EVAL_GPU_MAP="${EVAL_GPU_MAP:-}"

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
  local prompt="$1" hybrid="$2" tt="$3" ds="$4" algo="$5" gpu="$6"
  local pdev
  pdev="$(physical_gpu_for_slot "$gpu")"
  local tag="${OPT_BASE}__${prompt}__${hybrid}"
  local logf="$LOGDIR/qwen_${tag}_${ds}_${algo}_${TS}.log"
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
    --prompt-variant "$prompt"
    --hybrid-variant "$hybrid"
    "${RESUME_FLAG[@]}"
  )
  [[ "$DRY_RUN" == "1" ]] && runner+=( --llm-mock )
  log "  gpu=$gpu (phys $pdev) $tag $tt/$ds/$algo"
  CUDA_VISIBLE_DEVICES="$pdev" "${runner[@]}" >>"$logf" 2>&1
}

run_gpu_pool() {
  local -n _jobs=$1
  local label="$2"
  declare -A SLOT_PID=()
  local queue_idx=0 total=${#_jobs[@]}
  local gpu slot pid job prompt hybrid tt ds algo

  log "$label queue: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

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
        IFS='|' read -r prompt hybrid tt ds algo <<<"$job"
        log "dispatch $label $queue_idx/$total gpu=$gpu $prompt $hybrid $ds/$algo"
        ( run_one_task "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
}

log "=== Qwen HPUCB_arm_p05 + RSWarm ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "PUCB LLM 0.9→0.5  TPE 0.1→0.5  RS retrieve until trial 50"

for spec in "P5plusA5T50X50|HPUCB_arm_p05" "P5plusA5T50X50RSWarm|HPUCB_arm_p05"; do
  IFS='|' read -r prompt hybrid <<<"$spec"
  jobs=()
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    jobs+=("$prompt|$hybrid|$tt|$ds|$algo")
  done
  log "--- ARM $prompt / $hybrid (${#jobs[@]} tasks) ---"
  run_gpu_pool jobs "$prompt"
  log "--- ARM $prompt complete ---"
done

log "=== Qwen HPUCB_arm_p05 + RSWarm done ==="
