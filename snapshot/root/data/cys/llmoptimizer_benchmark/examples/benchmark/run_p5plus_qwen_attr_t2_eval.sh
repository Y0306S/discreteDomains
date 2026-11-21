#!/usr/bin/env bash
# Qwen LoRA s1b_attr_t2 partial-edit sweep — same 39 configs as gpt_p5_p5plus_summary.csv.
#
# MODE=seq_then_parallel (default):
#   1) SEQ_GPT_VARIANTS one config at a time (40 tasks × 80 trials)
#   2) remaining configs mixed in one GPU pool (like the GPT dynamic-gates sweep)
#
#   bash examples/benchmark/run_p5plus_qwen_attr_t2_eval.sh
#   JOBS_PER_GPU=3 NUM_GPUS=4 bash ...
#   MODE=sequential START_RANK=1 END_RANK=39 bash ...
#   DRY_RUN=1 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
SUMMARY_CSV="${SUMMARY_CSV:-benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus/gpt_p5_p5plus_summary.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
PER_RUN_TIME_LIMIT="${PER_RUN_TIME_LIMIT:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-20}"
START_RANK="${START_RANK:-1}"
END_RANK="${END_RANK:-999}"
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
MODE="${MODE:-seq_then_parallel}"
SEQ_GPT_VARIANTS="${SEQ_GPT_VARIANTS:-ptllm_tpe__P5plusA5T50X50__H0,ptllm_tpe__P5plusA5T50X50__HPUCB_arm}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_p5plus_attr_t2_${TS}.log"

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
  local rank="$1" gpt_variant="$2" opt_base="$3" prompt="$4" hybrid="$5" tt="$6" ds="$7" algo="$8" gpu="$9"
  local pdev
  pdev="$(physical_gpu_for_slot "$gpu")"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local logf="$LOGDIR/qwen_attr_t2_r${rank}_${tag}_${ds}_${algo}_${TS}.log"
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
    --optimizer "$opt_base"
    --prompt-variant "$prompt"
  )
  if [[ -n "$hybrid" ]]; then
    runner+=( --hybrid-variant "$hybrid" )
  fi
  runner+=( "${RESUME_FLAG[@]}" )
  [[ "$DRY_RUN" == "1" ]] && runner+=( --llm-mock )
  log "  gpu=$gpu (phys $pdev) rank=$rank $tag $tt/$ds/$algo"
  CUDA_VISIBLE_DEVICES="$pdev" "${runner[@]}" >>"$logf" 2>&1
}

run_gpu_pool() {
  local -n _jobs=$1
  local label="$2"
  declare -A SLOT_PID=()
  local queue_idx=0 total=${#_jobs[@]}
  local gpu slot pid job rank gpt_variant opt_base prompt hybrid tt ds algo

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
        IFS='|' read -r rank gpt_variant opt_base prompt hybrid tt ds algo <<<"$job"
        log "dispatch $label $queue_idx/$total gpu=$gpu r$rank $opt_base $prompt $hybrid $ds/$algo"
        ( run_one_task "$rank" "$gpt_variant" "$opt_base" "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
}

config_jobs() {
  local rank="$1" gpt_variant="$2" opt_base="$3" prompt="$4" hybrid="$5"
  local t tt ds algo
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    echo "$rank|$gpt_variant|$opt_base|$prompt|$hybrid|$tt|$ds|$algo"
  done
}

mapfile -t CONFIG_LINES < <("$PYTHON_LLM" - <<PY
import csv
from pathlib import Path

def parse_gpt_variant(v):
    parts = v.split("__")
    base = parts[0]
    if base == "ptllm_tpe":
        opt = "ptllm_tpe_qwen"
    elif base == "ptllm":
        opt = "ptllm_qwen"
    else:
        raise ValueError(f"unknown base {base!r} in {v!r}")
    prompt = parts[1] if len(parts) > 1 else "P5"
    hybrid = parts[2] if len(parts) > 2 else ""
    return opt, prompt, hybrid

rows = list(csv.DictReader(Path("$SUMMARY_CSV").open()))
for i, row in enumerate(rows, start=1):
    v = row["variant"].strip()
    opt, prompt, hybrid = parse_gpt_variant(v)
    print(f"{i}|{v}|{opt}|{prompt}|{hybrid}")
PY
)

IFS=',' read -ra SEQ_LIST <<<"$SEQ_GPT_VARIANTS"
declare -A SEQ_SET=()
for v in "${SEQ_LIST[@]}"; do
  v="${v// /}"
  [[ -n "$v" ]] && SEQ_SET["$v"]=1
done

log "=== Qwen s1b_attr_t2 P5+ sweep mode=$MODE ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "seq_variants=$SEQ_GPT_VARIANTS"

run_one_config() {
  local rank="$1" gpt_variant="$2" opt_base="$3" prompt="$4" hybrid="$5"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local jobs=()
  mapfile -t jobs < <(config_jobs "$rank" "$gpt_variant" "$opt_base" "$prompt" "$hybrid")
  log "--- RANK $rank / ${#CONFIG_LINES[@]}: $gpt_variant -> $tag (${#jobs[@]} tasks) ---"
  run_gpu_pool jobs "rank$rank"
  log "--- RANK $rank complete ---"
}

if [[ "$MODE" == "sequential" ]]; then
  for line in "${CONFIG_LINES[@]}"; do
    IFS='|' read -r rank gpt_variant opt_base prompt hybrid <<<"$line"
    if (( rank < START_RANK || rank > END_RANK )); then
      continue
    fi
    run_one_config "$rank" "$gpt_variant" "$opt_base" "$prompt" "$hybrid"
  done
else
  # Phase 1: named configs one at a time, CSV order among the named set.
  for want in "${SEQ_LIST[@]}"; do
    want="${want// /}"
    [[ -n "$want" ]] || continue
    found=0
    for line in "${CONFIG_LINES[@]}"; do
      IFS='|' read -r rank gpt_variant opt_base prompt hybrid <<<"$line"
      if [[ "$gpt_variant" == "$want" ]]; then
        run_one_config "$rank" "$gpt_variant" "$opt_base" "$prompt" "$hybrid"
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      log "ERROR seq variant not in summary CSV: $want"
      exit 1
    fi
  done

  # Phase 2: remaining configs mixed (all tasks) in one GPU pool.
  mix_jobs=()
  for line in "${CONFIG_LINES[@]}"; do
    IFS='|' read -r rank gpt_variant opt_base prompt hybrid <<<"$line"
    if (( rank < START_RANK || rank > END_RANK )); then
      continue
    fi
    [[ -n "${SEQ_SET[$gpt_variant]:-}" ]] && continue
    mapfile -t _one < <(config_jobs "$rank" "$gpt_variant" "$opt_base" "$prompt" "$hybrid")
    mix_jobs+=("${_one[@]}")
  done
  log "--- PARALLEL rest: ${#mix_jobs[@]} jobs (configs x tasks, seq arms excluded) ---"
  run_gpu_pool mix_jobs "parallel_rest"
fi

log "=== Qwen s1b_attr_t2 P5+ sweep done ==="
