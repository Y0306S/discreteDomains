#!/usr/bin/env bash
# After FanovaFullCs + FanovaFullIx finish, run two sequential Qwen evals:
#   1) P5plusA5TRdyFanovaX50 + HPUCB_arm_late05
#        shipping PUCB 0.9→0.1 until ~trial 50, then clamp ω_llm ≥ 0.5 so
#        Fanova-filtered prompts actually get late LLM turns
#   2) P5plusA5T50X50 + HPUCB_arm_H0split
#        HPUCB_arm for ~40 trials, then Bernoulli 50/50 (H0) for 41–80
# Leaves GPU 0 free: 3 GPUs × 3 workers on physical 1,2,3.
#
#   bash examples/benchmark/run_p5plus_qwen_late_floor_then_h0split.sh
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
MASTER_LOG="$LOGDIR/qwen_late_floor_h0split_${TS}.log"

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

"$PYTHON_LLM" - <<'PY'
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, load_hybrid_variants, apply_hybrid_settings
from argparse import Namespace
from mindware.components.optimizers.hybrid_selector import HybridSelector
import numpy as np

pv = load_prompt_variants()
row = _resolve_prompt_variant(pv["P5plusA5TRdyFanovaX50"], pv)
assert (row.get("experience_self_gate") or "clock") == "fanova", row.get("experience_self_gate")

hv = load_hybrid_variants()
assert "HPUCB_arm_late05" in hv and "HPUCB_arm_H0split" in hv
late = hv["HPUCB_arm_late05"]
assert float(late["pucb_floor"]) == 0.5
assert abs(float(late["pucb_floor_after_frac"]) - 0.6) < 1e-9
assert late.get("pucb_p_end") is None
spl = hv["HPUCB_arm_H0split"]
assert abs(float(spl["pucb_switch_after_frac"]) - 0.47) < 1e-9
assert float(spl["llm_prob"]) == 0.5

sel = HybridSelector(
    mode="pucb",
    pucb_p1=0.9,
    pucb_floor=0.5,
    pucb_floor_after_frac=0.6,
    rng=np.random.RandomState(0),
)
assert abs(sel.omega("llm", 0, 76) - 0.9) < 1e-9
assert abs(sel.omega("llm", 46, 76) - 0.5) < 1e-9
args = Namespace()
apply_hybrid_settings(args, "HPUCB_arm_late05")
assert args.pucb_floor == 0.5
apply_hybrid_settings(args, "HPUCB_arm_H0split")
assert abs(args.pucb_switch_after_frac - 0.47) < 1e-9
print("ok late05 floor", late["pucb_floor"], "after_frac", late["pucb_floor_after_frac"])
print("ok H0split switch_frac", spl["pucb_switch_after_frac"])
PY

cs_ix_still_running() {
  pgrep -f '[r]un_p5plus_qwen_fanova_cs_and_ix.sh' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5TRdyFanovaFullCsX50' >/dev/null 2>&1 && return 0
  pgrep -f '[P]5plusA5TRdyFanovaFullIxX50' >/dev/null 2>&1 && return 0
  return 1
}

if [[ "$DRY_RUN" != "1" ]] && cs_ix_still_running; then
  log "waiting for FanovaFullCs / FanovaFullIx eval to finish"
  while cs_ix_still_running; do
    n_cs=$(pgrep -cf '[P]5plusA5TRdyFanovaFullCsX50' || true)
    n_ix=$(pgrep -cf '[P]5plusA5TRdyFanovaFullIxX50' || true)
    log "  still running CS python≈${n_cs}  Ix python≈${n_ix}"
    sleep "$WAIT_POLL_SEC"
  done
  log "Fanova CS/Ix finished — starting late-floor then H0-split"
else
  log "Fanova CS/Ix not running (or DRY_RUN) — starting late-floor then H0-split"
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

log "=== Qwen Fanova+late-floor then HPUCB→H0 split ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU map=$EVAL_GPU_MAP"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"

for spec in "P5plusA5TRdyFanovaX50|HPUCB_arm_late05" "P5plusA5T50X50|HPUCB_arm_H0split"; do
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

log "=== Qwen late-floor + H0-split done ==="
