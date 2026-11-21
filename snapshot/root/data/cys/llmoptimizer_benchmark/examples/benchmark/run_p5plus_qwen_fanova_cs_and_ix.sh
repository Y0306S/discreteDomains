#!/usr/bin/env bash
# After the current combo (proxy Fanova + RSWarm + HPUCB_arm_p05) finishes,
# run two sequential official-fANOVA follow-ups on the same Qwen eval:
#   1) P5plusA5TRdyFanovaFullCsX50 + H2
#        1+3: real ConfigSpace + 1D main effects + calibrated keep
#   2) P5plusA5TRdyFanovaFullIxX50 + H2
#        2+3: real ConfigSpace + total (main+interaction) importance + calibrated keep
# XTPE is NOT launched (backup only; P5plusA5T50XTPELate waits until trial 200).
#
# Leaves GPU 0 free: 3 GPUs × 3 workers on physical 1,2,3.
#
#   bash examples/benchmark/run_p5plus_qwen_fanova_cs_and_ix.sh
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
NUM_GPUS="${NUM_GPUS:-3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-20}"
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_GPU_MAP="${EVAL_GPU_MAP:-1,2,3}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-60}"
export FANOVA_PYTHON="${FANOVA_PYTHON:-/root/venvs/fanova/bin/python}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_fanova_cs_ix_${TS}.log"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  log "ERROR missing adapter $ADAPTER_PATH/adapter_model.safetensors"
  exit 1
fi
if [[ ! -x "$FANOVA_PYTHON" ]]; then
  log "ERROR missing FANOVA_PYTHON=$FANOVA_PYTHON"
  exit 1
fi

"$PYTHON_LLM" - <<PY
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, load_hybrid_variants, apply_prompt_settings
from argparse import Namespace
pv = load_prompt_variants()
for pid, gate in (
    ("P5plusA5TRdyFanovaFullCsX50", "fanova_full_cs"),
    ("P5plusA5TRdyFanovaFullIxX50", "fanova_full_ix"),
):
    row = _resolve_prompt_variant(pv[pid], pv)
    got = row.get("experience_self_gate") or "clock"
    assert got == gate, (pid, got, gate)
    ee = row.get("experience_explore_mode") or "off"
    assert ee == "fifty_fifty", (pid, ee)
    print("ok", pid, "gate=", got, "explore=", ee)
late = _resolve_prompt_variant(pv["P5plusA5T50XTPELate"], pv)
assert late.get("experience_explore_mode") == "tpe"
assert int(late.get("experience_explore_tpe_min_trial") or 0) == 200
args = Namespace(llm_history_window=80)
apply_prompt_settings(args, "P5plusA5T50XTPELate")
assert args.llm_experience_explore_tpe_min_trial == 200
assert "H2" in load_hybrid_variants()
print("ok XTPELate min_trial", args.llm_experience_explore_tpe_min_trial)
print("ok FANOVA_PYTHON=", "$FANOVA_PYTHON")
PY

combo_still_running() {
  pgrep -f '[r]un_p5plus_qwen_combo_fanova_rswarm_hpucb_arm_p05.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5TRdyFanovaX50RSWarm' >/dev/null 2>&1 && return 0
  return 1
}

if [[ "$DRY_RUN" != "1" ]] && combo_still_running; then
  log "waiting for current combo (Fanova+RSWarm+HPUCB_arm_p05) to finish"
  while combo_still_running; do
    n_py=$(pgrep -cf '[P]5plusA5TRdyFanovaX50RSWarm' || true)
    log "  still running combo python jobs≈${n_py}"
    sleep "$WAIT_POLL_SEC"
  done
  log "combo finished — starting FanovaFullCs then FanovaFullIx"
else
  log "combo not running (or DRY_RUN) — starting Fanova CS/Ix arms"
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

  log "$label queue: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU map=$EVAL_GPU_MAP"

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

log "=== Qwen FanovaFullCs (1+3) then FanovaFullIx (2+3) ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU map=$EVAL_GPU_MAP"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "FANOVA_PYTHON=$FANOVA_PYTHON  XTPE not launched (Late backup min_trial=200)"

for spec in "P5plusA5TRdyFanovaFullCsX50|H2" "P5plusA5TRdyFanovaFullIxX50|H2"; do
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

log "=== Qwen FanovaFullCs + FanovaFullIx done ==="
