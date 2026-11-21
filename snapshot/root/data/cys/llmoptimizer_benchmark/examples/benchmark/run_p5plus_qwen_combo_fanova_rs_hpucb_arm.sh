#!/usr/bin/env bash
# Kitchen-sink combo on Qwen LoRA s1b_attr_t2:
#   prompt  = P5plusA5TRdyFanovaX50RS  (A5 + Fanova gate + X50 + RS surrogate)
#   hybrid  = HPUCB_arm
#   model   = llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2
#
#   bash examples/benchmark/run_p5plus_qwen_combo_fanova_rs_hpucb_arm.sh
#   JOBS_PER_GPU=3 NUM_GPUS=4 bash ...
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
PROMPT_VARIANT="${PROMPT_VARIANT:-P5plusA5TRdyFanovaX50RS}"
HYBRID_VARIANT="${HYBRID_VARIANT:-HPUCB_arm}"
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
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TAG="${OPT_BASE}__${PROMPT_VARIANT}__${HYBRID_VARIANT}"
MASTER_LOG="$LOGDIR/qwen_combo_fanova_rs_hpucb_arm_${TS}.log"

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

  log "queue: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

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
        log "dispatch $queue_idx/$total gpu=$gpu $ds/$algo"
        ( run_one_task "$tt" "$ds" "$algo" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
}

# Sanity: resolve prompt + hybrid before burning GPU time
"$PYTHON_LLM" - <<PY
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, load_hybrid_variants
pv = load_prompt_variants()
row = _resolve_prompt_variant(pv["$PROMPT_VARIANT"], pv)
assert row.get("experience_self_gate") == "fanova", row.get("experience_self_gate")
assert row.get("experience_retrieve_ranker") == "surrogate", row.get("experience_retrieve_ranker")
assert row.get("prompt_blocks", {}).get("experience_retrieve") is True
assert row.get("prompt_blocks", {}).get("experience_self") is True
hv = load_hybrid_variants()
assert "$HYBRID_VARIANT" in hv
print("ok", "$PROMPT_VARIANT", "blocks=", sorted(row["prompt_blocks"]), "gate=", row["experience_self_gate"], "ranker=", row["experience_retrieve_ranker"])
print("ok hybrid", "$HYBRID_VARIANT", hv["$HYBRID_VARIANT"].get("hybrid_selector"), hv["$HYBRID_VARIANT"].get("pucb_q"))
PY

jobs=()
for t in "${TASKS[@]}"; do
  jobs+=("$t")
done

log "=== Qwen combo Fanova+RS+HPUCB_arm ==="
log "tag=$TAG"
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "bank=history_data/experience_bank_v2_holdout40.jsonl"

run_gpu_pool jobs

log "=== Qwen combo Fanova+RS+HPUCB_arm done ==="
#!/usr/bin/env bash
# Kitchen-sink combo on Qwen LoRA s1b_attr_t2:
#   prompt  = P5plusA5TRdyFanovaX50RS  (A5 + Fanova gate + X50 + RS surrogate)
#   hybrid  = HPUCB_arm
#   model   = llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2
#
#   bash examples/benchmark/run_p5plus_qwen_combo_fanova_rs_hpucb_arm.sh
#   JOBS_PER_GPU=3 NUM_GPUS=4 bash ...
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
PROMPT_VARIANT="${PROMPT_VARIANT:-P5plusA5TRdyFanovaX50RS}"
HYBRID_VARIANT="${HYBRID_VARIANT:-HPUCB_arm}"
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
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TAG="${OPT_BASE}__${PROMPT_VARIANT}__${HYBRID_VARIANT}"
MASTER_LOG="$LOGDIR/qwen_combo_fanova_rs_hpucb_arm_${TS}.log"

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

  log "queue: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

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
        log "dispatch $queue_idx/$total gpu=$gpu $ds/$algo"
        ( run_one_task "$tt" "$ds" "$algo" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
}

# Sanity: resolve prompt + hybrid before burning GPU time
"$PYTHON_LLM" - <<PY
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, load_hybrid_variants
pv = load_prompt_variants()
row = _resolve_prompt_variant(pv["$PROMPT_VARIANT"], pv)
assert row.get("experience_self_gate") == "fanova", row.get("experience_self_gate")
assert row.get("experience_retrieve_ranker") == "surrogate", row.get("experience_retrieve_ranker")
assert row.get("prompt_blocks", {}).get("experience_retrieve") is True
assert row.get("prompt_blocks", {}).get("experience_self") is True
hv = load_hybrid_variants()
assert "$HYBRID_VARIANT" in hv
print("ok", "$PROMPT_VARIANT", "blocks=", sorted(row["prompt_blocks"]), "gate=", row["experience_self_gate"], "ranker=", row["experience_retrieve_ranker"])
print("ok hybrid", "$HYBRID_VARIANT", hv["$HYBRID_VARIANT"].get("hybrid_selector"), hv["$HYBRID_VARIANT"].get("pucb_q"))
PY

jobs=()
for t in "${TASKS[@]}"; do
  jobs+=("$t")
done

log "=== Qwen combo Fanova+RS+HPUCB_arm ==="
log "tag=$TAG"
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "bank=history_data/experience_bank_v2_holdout40.jsonl"

run_gpu_pool jobs

log "=== Qwen combo Fanova+RS+HPUCB_arm done ==="
#!/usr/bin/env bash
# Kitchen-sink combo on Qwen LoRA s1b_attr_t2:
#   prompt  = P5plusA5TRdyFanovaX50RS  (A5 + Fanova gate + X50 + RS surrogate)
#   hybrid  = HPUCB_arm
#   model   = llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2
#
#   bash examples/benchmark/run_p5plus_qwen_combo_fanova_rs_hpucb_arm.sh
#   JOBS_PER_GPU=3 NUM_GPUS=4 bash ...
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
PROMPT_VARIANT="${PROMPT_VARIANT:-P5plusA5TRdyFanovaX50RS}"
HYBRID_VARIANT="${HYBRID_VARIANT:-HPUCB_arm}"
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
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TAG="${OPT_BASE}__${PROMPT_VARIANT}__${HYBRID_VARIANT}"
MASTER_LOG="$LOGDIR/qwen_combo_fanova_rs_hpucb_arm_${TS}.log"

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

  log "queue: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

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
        log "dispatch $queue_idx/$total gpu=$gpu $ds/$algo"
        ( run_one_task "$tt" "$ds" "$algo" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
}

# Sanity: resolve prompt + hybrid before burning GPU time
"$PYTHON_LLM" - <<PY
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, load_hybrid_variants
pv = load_prompt_variants()
row = _resolve_prompt_variant(pv["$PROMPT_VARIANT"], pv)
assert row.get("experience_self_gate") == "fanova", row.get("experience_self_gate")
assert row.get("experience_retrieve_ranker") == "surrogate", row.get("experience_retrieve_ranker")
assert row.get("prompt_blocks", {}).get("experience_retrieve") is True
assert row.get("prompt_blocks", {}).get("experience_self") is True
hv = load_hybrid_variants()
assert "$HYBRID_VARIANT" in hv
print("ok", "$PROMPT_VARIANT", "blocks=", sorted(row["prompt_blocks"]), "gate=", row["experience_self_gate"], "ranker=", row["experience_retrieve_ranker"])
print("ok hybrid", "$HYBRID_VARIANT", hv["$HYBRID_VARIANT"].get("hybrid_selector"), hv["$HYBRID_VARIANT"].get("pucb_q"))
PY

jobs=()
for t in "${TASKS[@]}"; do
  jobs+=("$t")
done

log "=== Qwen combo Fanova+RS+HPUCB_arm ==="
log "tag=$TAG"
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "bank=history_data/experience_bank_v2_holdout40.jsonl"

run_gpu_pool jobs

log "=== Qwen combo Fanova+RS+HPUCB_arm done ==="
#!/usr/bin/env bash
# Kitchen-sink combo on Qwen LoRA s1b_attr_t2:
#   prompt  = P5plusA5TRdyFanovaX50RS  (A5 + Fanova gate + X50 + RS surrogate)
#   hybrid  = HPUCB_arm
#   model   = llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2
#
#   bash examples/benchmark/run_p5plus_qwen_combo_fanova_rs_hpucb_arm.sh
#   JOBS_PER_GPU=3 NUM_GPUS=4 bash ...
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
PROMPT_VARIANT="${PROMPT_VARIANT:-P5plusA5TRdyFanovaX50RS}"
HYBRID_VARIANT="${HYBRID_VARIANT:-HPUCB_arm}"
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
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
TAG="${OPT_BASE}__${PROMPT_VARIANT}__${HYBRID_VARIANT}"
MASTER_LOG="$LOGDIR/qwen_combo_fanova_rs_hpucb_arm_${TS}.log"

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

  log "queue: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

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
        log "dispatch $queue_idx/$total gpu=$gpu $ds/$algo"
        ( run_one_task "$tt" "$ds" "$algo" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
}

# Sanity: resolve prompt + hybrid before burning GPU time
"$PYTHON_LLM" - <<PY
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, load_hybrid_variants
pv = load_prompt_variants()
row = _resolve_prompt_variant(pv["$PROMPT_VARIANT"], pv)
assert row.get("experience_self_gate") == "fanova", row.get("experience_self_gate")
assert row.get("experience_retrieve_ranker") == "surrogate", row.get("experience_retrieve_ranker")
assert row.get("prompt_blocks", {}).get("experience_retrieve") is True
assert row.get("prompt_blocks", {}).get("experience_self") is True
hv = load_hybrid_variants()
assert "$HYBRID_VARIANT" in hv
print("ok", "$PROMPT_VARIANT", "blocks=", sorted(row["prompt_blocks"]), "gate=", row["experience_self_gate"], "ranker=", row["experience_retrieve_ranker"])
print("ok hybrid", "$HYBRID_VARIANT", hv["$HYBRID_VARIANT"].get("hybrid_selector"), hv["$HYBRID_VARIANT"].get("pucb_q"))
PY

jobs=()
for t in "${TASKS[@]}"; do
  jobs+=("$t")
done

log "=== Qwen combo Fanova+RS+HPUCB_arm ==="
log "tag=$TAG"
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS"
log "bank=history_data/experience_bank_v2_holdout40.jsonl"

run_gpu_pool jobs

log "=== Qwen combo Fanova+RS+HPUCB_arm done ==="
