#!/usr/bin/env bash
# Eval suite on approved 40 held-out (dataset, algorithm) pairs.
#
# Arms (after DPO B–H training finishes, or run standalone):
#   Classical: smac, tpe, cmaes
#   SFT LoRA:  s1a_ep7p5 / s1b_ep7p5 × P5/H0 × (ptllm_smac + ptllm_tpe)
#   DPO LoRA:  A–H × P5/H0 × (ptllm_smac + ptllm_tpe)
#
# Usage:
#   bash examples/benchmark/run_mltest20_algo40_eval.sh
#   STAGE=classical|sft|dpo|all WAIT_DPO=0 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_CLASS="${PYTHON_CLASS:-/root/miniconda3/envs/openbox_eval/bin/python}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
CLASS_PARALLEL="${CLASS_PARALLEL:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-20}"
HYBRID="${HYBRID:-H0}"
PROMPT_VARIANT="${PROMPT_VARIANT:-P5}"
# Space-separated LLM optimizers for LoRA stages (default: smac + tpe)
LLM_OPTIMIZERS="${LLM_OPTIMIZERS:-ptllm_smac_qwen ptllm_tpe_qwen}"
STAGE="${STAGE:-all}"   # classical | sft | dpo | s1a_s2_cg | all
EVAL_RESUME="${EVAL_RESUME:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/algo40_eval_${TS}.log"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

mapfile -t TASK_LINES < <("$PYTHON_CLASS" - <<PY
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
log "Loaded ${#TASKS[@]} tasks from $TASKS_JSON"

RESUME_FLAG=()
[[ "$EVAL_RESUME" == "1" ]] && RESUME_FLAG=(--resume)

# ---------- classical ----------
run_classical() {
  local out="$OUTPUT_ROOT/classical"
  mkdir -p "$out"
  log "CLASSICAL start -> $out (parallel=$CLASS_PARALLEL)"
  local jobs=()
  local t tt ds algo opt
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    for opt in smac tpe cmaes; do
      jobs+=("$tt|$ds|$algo|$opt")
    done
  done
  log "CLASSICAL ${#jobs[@]} jobs"
  local job
  for job in "${jobs[@]}"; do
    IFS='|' read -r tt ds algo opt <<<"$job"
    while [[ $(jobs -rp | wc -l) -ge $CLASS_PARALLEL ]]; do sleep 5; done
    (
      echo "[$(date -Is)] START $opt $tt/$ds/$algo" >>"$MASTER_LOG"
      "$PYTHON_CLASS" examples/benchmark/run_toy_categorical_optimizers.py \
        --algorithm "$algo" \
        --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval \
        --optimizer "$opt" \
        "${RESUME_FLAG[@]}" \
        --output-dir "$out" \
        --all-datasets --include-datasets "$ds" \
        --time-limit "$TIME_LIMIT" --dataset-retries "$DATASET_RETRIES" \
        >>"$LOGDIR/classical_${opt}_${ds}_${TS}.log" 2>&1
      echo "[$(date -Is)] DONE $opt $tt/$ds/$algo" >>"$MASTER_LOG"
    ) &
  done
  wait
  log "CLASSICAL done"
}

# ---------- LLM parallel pool ----------
declare -A ADAPTER_FOR=(
  [s1a_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1a_ep7p5_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1a_ep7p5_r3]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1b_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5"
  [dpo_A]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_A"
  [dpo_B]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_B"
  [dpo_C]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_C"
  [dpo_D]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5_dpo_D"
  [dpo_E]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_E"
  [dpo_F]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_F"
  [dpo_G]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_G"
  [dpo_G_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_G"
  [dpo_H]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_H"
  [s1a_s2_C]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_C"
  [s1a_s2_G]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G"
  [s1a_s2_D1]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1"
  [s1a_s2_D1_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1"
  [s1a_s2_D1m]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1m"
  [s1a_s2_D2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D2"
  [s1a_s2_D3]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D3"
  [s1a_s2_O1]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O1"
  [s1a_s2_O3]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O3"
  [s1a_s2_O4]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O4"
  [s1a_r0a]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0a"
  [s1a_r0b]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0b"
  [s1a_r0c]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0c"
  [s1a_r0d]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0d"
  [s1a_a_boseed_global]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_a_boseed_global"
  [s1a_a_boseed_dsseq]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_a_boseed_dsseq"
  [s1a_b_prefix_grpo]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_grpo"
  [s1a_b_prefix_dsseq]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq"
  [s1a_b_prefix_dsseq_hg8]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_hg8"
  [s1a_b_prefix_dsseq_m32]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_m32"
  [dpo_rationale_C]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_C"
  [dpo_rationale_G]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G"
  [dpo_rationale_G_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G"
)
declare -A RESPFMT_FOR=(
  [s1a_ep7p5]=json_only
  [s1a_ep7p5_r2]=json_only
  [s1a_ep7p5_r3]=json_only
  [s1b_ep7p5]=rationale_json
  [dpo_A]=rationale_json
  [dpo_B]=rationale_json
  [dpo_C]=rationale_json
  [dpo_D]=rationale_json
  [dpo_E]=rationale_json
  [dpo_F]=rationale_json
  [dpo_G]=rationale_json
  [dpo_G_r2]=rationale_json
  [dpo_H]=rationale_json
  [s1a_s2_C]=json_only
  [s1a_s2_G]=json_only
  [s1a_s2_D1]=json_only
  [s1a_s2_D1_r2]=json_only
  [s1a_s2_D1m]=json_only
  [s1a_s2_D2]=json_only
  [s1a_s2_D3]=json_only
  [s1a_s2_O1]=json_only
  [s1a_s2_O3]=json_only
  [s1a_s2_O4]=json_only
  [s1a_r0a]=json_only
  [s1a_r0b]=json_only
  [s1a_r0c]=json_only
  [s1a_r0d]=json_only
  [s1a_a_boseed_global]=json_only
  [s1a_a_boseed_dsseq]=json_only
  [s1a_b_prefix_grpo]=json_only
  [s1a_b_prefix_dsseq]=json_only
  [s1a_b_prefix_dsseq_hg8]=json_only
  [s1a_b_prefix_dsseq_m32]=json_only
  [dpo_rationale_C]=rationale_json
  [dpo_rationale_G]=rationale_json
  [dpo_rationale_G_r2]=rationale_json
)

EVAL_GPU_MAP="${EVAL_GPU_MAP:-}"  # e.g. 0,1 or 2,3 — slot index -> physical GPU id
JOBS_PER_GPU_MAP="${JOBS_PER_GPU_MAP:-}"  # e.g. 2,3,3 — per-slot worker cap (defaults to JOBS_PER_GPU)

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

jobs_per_gpu_for_slot() {
  local slot="$1"
  if [[ -n "$JOBS_PER_GPU_MAP" ]]; then
    IFS=',' read -ra _jmap <<<"$JOBS_PER_GPU_MAP"
    echo "${_jmap[$slot]:-$JOBS_PER_GPU}"
  else
    echo "$JOBS_PER_GPU"
  fi
}

run_llm_job() {
  local model="$1" ds="$2" algo="$3" opt="$4" gpu="$5"
  local pdev
  pdev="$(physical_gpu_for_slot "$gpu")"
  local adapter="${ADAPTER_FOR[$model]}"
  local resp="${RESPFMT_FOR[$model]}"
  local out="${OUTPUT_ROOT}_p5_${model}"
  local logf="$LOGDIR/llm_${model}_${opt}_${HYBRID}_${ds}_${TS}.log"
  mkdir -p "$out"
  log "LLM gpu=$gpu (phys $pdev) model=$model opt=$opt $ds/$algo"
  CUDA_VISIBLE_DEVICES="$pdev" "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval \
    --llm-backends qwen-sft6 --qwen-backend hf \
    --qwen-model-path "$BASE_MODEL" \
    --qwen-adapter-path "$adapter" \
    --llm-max-tokens 768 \
    --llm-response-format "$resp" \
    --output-dir "$out" \
    --all-datasets --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" --dataset-retries "$DATASET_RETRIES" \
    --optimizer "$opt" \
    --prompt-variant "$PROMPT_VARIANT" \
    --hybrid-variant "$HYBRID" \
    "${RESUME_FLAG[@]}" >>"$logf" 2>&1
}

run_llm_queue() {
  local -n models_ref=$1
  local label="$2"
  local lora_jobs=()
  local model t tt ds algo opt
  read -r -a _llm_opts <<<"$LLM_OPTIMIZERS"
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    for model in "${models_ref[@]}"; do
      for opt in "${_llm_opts[@]}"; do
        lora_jobs+=("$model|$ds|$algo|$opt")
      done
    done
  done
  log "$label queue: ${#lora_jobs[@]} jobs (models×[${LLM_OPTIMIZERS}]×40) P${PROMPT_VARIANT#P}/${HYBRID}"

  # require adapters
  for model in "${models_ref[@]}"; do
    local w="${ADAPTER_FOR[$model]}/adapter_model.safetensors"
    if [[ ! -f "$w" ]]; then
      log "ERROR missing adapter $w — skip model $model"
    fi
  done

  declare -A SLOT_PID=()
  local queue_idx=0 total=${#lora_jobs[@]}
  local gpu slot pid job jpg

  while [[ $queue_idx -lt $total ]] || {
      local any=0
      for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        jpg="$(jobs_per_gpu_for_slot "$gpu")"
        for ((slot=0; slot<jpg; slot++)); do
          is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && any=1
        done
      done
      [[ $any -eq 1 ]]
    }; do
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      jpg="$(jobs_per_gpu_for_slot "$gpu")"
      for ((slot=0; slot<jpg; slot++)); do
        pid="${SLOT_PID[${gpu}_${slot}]:-}"
        if is_alive "$pid"; then continue; fi
        SLOT_PID[${gpu}_${slot}]=
        [[ $queue_idx -ge $total ]] && continue
        # packing: slot0 always if free; slot1+ needs free VRAM
        local count=0 s
        for ((s=0; s<jpg; s++)); do
          is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
        done
        [[ $count -lt $jpg ]] || continue
        if [[ $count -ge 1 ]]; then
          local free pdev
          pdev="$(physical_gpu_for_slot "$gpu")"
          free=$(gpu_free_mb "$pdev")
          [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
        fi
        job="${lora_jobs[$queue_idx]}"
        queue_idx=$((queue_idx+1))
        IFS='|' read -r model ds algo opt <<<"$job"
        if [[ ! -f "${ADAPTER_FOR[$model]}/adapter_model.safetensors" ]]; then
          log "SKIP missing $model ($queue_idx/$total)"
          continue
        fi
        log "dispatch $queue_idx/$total gpu=$gpu $model $opt $ds"
        ( run_llm_job "$model" "$ds" "$algo" "$opt" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
  log "$label done"
}

run_sft() {
  local models=(s1a_ep7p5 s1b_ep7p5)
  run_llm_queue models "SFT"
}

run_dpo() {
  local models=(dpo_A dpo_B dpo_C dpo_D dpo_E dpo_F dpo_G dpo_H)
  run_llm_queue models "DPO"
}

run_s1a_s2_cg() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  for model in s1a_s2_C s1a_s2_G; do
    local -a one=("$model")
    run_llm_queue one "S1A_S2_${model}"
  done
}

run_s1a_s2_c() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_s2_C)
  run_llm_queue one "S1A_S2_C"
}

run_s1a_s2_g() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_s2_G)
  run_llm_queue one "S1A_S2_G"
}

run_s1a_s2_conservative() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local models=(s1a_s2_D1 s1a_s2_D2 s1a_s2_D3 s1a_s2_O1 s1a_s2_O3 s1a_s2_O4)
  for model in "${models[@]}"; do
    local -a one=("$model")
    run_llm_queue one "S1A_S2_${model}"
  done
}

run_s1a_s2_o3() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_s2_O3)
  run_llm_queue one "S1A_S2_O3"
}

run_s1a_s2_micro() {
  # Half-epoch micros: ORPO-O3 + DPO-D1m — same queue so eval packs in parallel
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_s2_O3 s1a_s2_D1m)
  run_llm_queue models "S1A_S2_MICRO"
}

run_s1a_r0_light() {
  # REINFORCE lights R0a (T=0.7) + R0b (T=1.0) after on-policy update
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_r0a s1a_r0b)
  run_llm_queue models "S1A_R0_LIGHT"
}

run_s1a_r0_medium() {
  # REINFORCE mediums R0c (T=0.7) + R0d (T=1.0) on picked-32
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_r0c s1a_r0d)
  run_llm_queue models "S1A_R0_MEDIUM"
}

run_s1a_setup_ab() {
  # BO-seeded Setup A (global + dsseq) and prefix Setup B (grpo + dsseq)
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_a_boseed_global s1a_a_boseed_dsseq s1a_b_prefix_grpo s1a_b_prefix_dsseq)
  run_llm_queue models "S1A_SETUP_AB"
}

run_s1a_b_prefix_dsseq_hg8() {
  # Setup B dsseq mid-ckpt (top-8 high-growth) and/or full medium-32 final
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_b_prefix_dsseq_hg8)
  run_llm_queue models "S1A_B_PREFIX_DSSEQ_HG8"
}

run_s1a_b_prefix_dsseq_m32() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_b_prefix_dsseq_m32)
  run_llm_queue models "S1A_B_PREFIX_DSSEQ_M32"
}

run_s1a_b_prefix_dsseq_hg8_m32() {
  # Both mid (hg8) and final (medium32) from one train run
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_b_prefix_dsseq_hg8 s1a_b_prefix_dsseq_m32)
  run_llm_queue models "S1A_B_PREFIX_DSSEQ_HG8_M32"
}

run_from_manifest() {
  # MANIFEST_JSON: [{"id":"model_id","adapter":"/path","resp":"json_only"}, ...]
  local mf="${MANIFEST_JSON:-}"
  if [[ -z "$mf" || ! -f "$mf" ]]; then
    log "ERROR: STAGE=from_manifest requires MANIFEST_JSON pointing to a JSON list"
    exit 1
  fi
  LLM_OPTIMIZERS="${LLM_OPTIMIZERS:-ptllm_tpe_qwen}"
  PROMPT_VARIANT="${PROMPT_VARIANT:-P5}"
  HYBRID="${HYBRID:-H0}"
  local -a models=()
  while IFS= read -r mid; do
    [[ -n "$mid" ]] || continue
    models+=("$mid")
    local adapter resp
    adapter="$("$PYTHON_CLASS" -c "import json; m=json.load(open('$mf')); print(next(x['adapter'] for x in m if x['id']=='$mid'))")"
    resp="$("$PYTHON_CLASS" -c "import json; m=json.load(open('$mf')); print(next(x.get('resp','json_only') for x in m if x['id']=='$mid'))")"
    ADAPTER_FOR["$mid"]="$adapter"
    RESPFMT_FOR["$mid"]="$resp"
  done < <("$PYTHON_CLASS" -c "import json; print('\n'.join(x['id'] for x in json.load(open('$mf'))))")
  if [[ ${#models[@]} -eq 0 ]]; then
    log "ERROR: empty manifest $mf"
    exit 1
  fi
  run_llm_queue models "FROM_MANIFEST"
}

run_s1b_dpo_rationale_cg() {
  # s1b rationale-aligned DPO-C + ORPO-G (p5_s2_dpo_rationale)
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  for model in dpo_rationale_C dpo_rationale_G; do
    local -a one=("$model")
    run_llm_queue one "S1B_DPO_RATIONALE_${model}"
  done
}

# Replication: same s1a adapter, fresh output dir (use SEED=2 vs original seed=1)
run_s1a_ep7p5_tpe_r2() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_ep7p5_r2)
  run_llm_queue one "S1A_EP7P5_TPE_R2"
}

# Replication: same s1a adapter, fresh output dir (use SEED=3 vs original seed=1)
run_s1a_ep7p5_tpe_r3() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_ep7p5_r3)
  run_llm_queue one "S1A_EP7P5_TPE_R3"
}

# Seed=2 replications for leaderboard top arms (same adapters, fresh dirs)
run_seed2_top3() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  for model in dpo_G_r2 s1a_s2_D1_r2 dpo_rationale_G_r2; do
    local -a one=("$model")
    run_llm_queue one "SEED2_${model}"
  done
}

log "=== algo40 eval STAGE=$STAGE HYBRID=$HYBRID PROMPT=$PROMPT_VARIANT LLM_OPTIMIZERS=$LLM_OPTIMIZERS n_trials=$N_TRIALS SEED=$SEED ==="

case "$STAGE" in
  classical) run_classical ;;
  sft) run_sft ;;
  dpo) run_dpo ;;
  s1a_s2_cg) run_s1a_s2_cg ;;
  s1a_s2_c) run_s1a_s2_c ;;
  s1a_s2_g) run_s1a_s2_g ;;
  s1a_s2_conservative) run_s1a_s2_conservative ;;
  s1a_s2_o3) run_s1a_s2_o3 ;;
  s1a_s2_micro) run_s1a_s2_micro ;;
  s1a_r0_light) run_s1a_r0_light ;;
  s1a_r0_medium) run_s1a_r0_medium ;;
  s1a_setup_ab) run_s1a_setup_ab ;;
  s1a_b_prefix_dsseq_hg8) run_s1a_b_prefix_dsseq_hg8 ;;
  s1a_b_prefix_dsseq_m32) run_s1a_b_prefix_dsseq_m32 ;;
  s1a_b_prefix_dsseq_hg8_m32) run_s1a_b_prefix_dsseq_hg8_m32 ;;
  from_manifest) run_from_manifest ;;
  s1b_dpo_rationale_cg) run_s1b_dpo_rationale_cg ;;
  s1a_ep7p5_tpe_r2) run_s1a_ep7p5_tpe_r2 ;;
  s1a_ep7p5_tpe_r3) run_s1a_ep7p5_tpe_r3 ;;
  seed2_top3) run_seed2_top3 ;;
  all)
    # classical on CPU in parallel with SFT on GPU
    run_classical &
    class_pid=$!
    run_sft
    wait "$class_pid" || true
    run_dpo
    ;;
  *) log "unknown STAGE=$STAGE"; exit 1 ;;
esac

log "=== algo40 eval complete ==="
exit 0
#!/usr/bin/env bash
# Eval suite on approved 40 held-out (dataset, algorithm) pairs.
#
# Arms (after DPO B–H training finishes, or run standalone):
#   Classical: smac, tpe, cmaes
#   SFT LoRA:  s1a_ep7p5 / s1b_ep7p5 × P5/H0 × (ptllm_smac + ptllm_tpe)
#   DPO LoRA:  A–H × P5/H0 × (ptllm_smac + ptllm_tpe)
#
# Usage:
#   bash examples/benchmark/run_mltest20_algo40_eval.sh
#   STAGE=classical|sft|dpo|all WAIT_DPO=0 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_CLASS="${PYTHON_CLASS:-/root/miniconda3/envs/openbox_eval/bin/python}"
PYTHON_LLM="${PYTHON_LLM:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
CLASS_PARALLEL="${CLASS_PARALLEL:-4}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-20}"
HYBRID="${HYBRID:-H0}"
PROMPT_VARIANT="${PROMPT_VARIANT:-P5}"
# Space-separated LLM optimizers for LoRA stages (default: smac + tpe)
LLM_OPTIMIZERS="${LLM_OPTIMIZERS:-ptllm_smac_qwen ptllm_tpe_qwen}"
STAGE="${STAGE:-all}"   # classical | sft | dpo | s1a_s2_cg | all
EVAL_RESUME="${EVAL_RESUME:-1}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/algo40_eval_${TS}.log"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

mapfile -t TASK_LINES < <("$PYTHON_CLASS" - <<PY
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
log "Loaded ${#TASKS[@]} tasks from $TASKS_JSON"

RESUME_FLAG=()
[[ "$EVAL_RESUME" == "1" ]] && RESUME_FLAG=(--resume)

# ---------- classical ----------
run_classical() {
  local out="$OUTPUT_ROOT/classical"
  mkdir -p "$out"
  log "CLASSICAL start -> $out (parallel=$CLASS_PARALLEL)"
  local jobs=()
  local t tt ds algo opt
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    for opt in smac tpe cmaes; do
      jobs+=("$tt|$ds|$algo|$opt")
    done
  done
  log "CLASSICAL ${#jobs[@]} jobs"
  local job
  for job in "${jobs[@]}"; do
    IFS='|' read -r tt ds algo opt <<<"$job"
    while [[ $(jobs -rp | wc -l) -ge $CLASS_PARALLEL ]]; do sleep 5; done
    (
      echo "[$(date -Is)] START $opt $tt/$ds/$algo" >>"$MASTER_LOG"
      "$PYTHON_CLASS" examples/benchmark/run_toy_categorical_optimizers.py \
        --algorithm "$algo" \
        --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval \
        --optimizer "$opt" \
        "${RESUME_FLAG[@]}" \
        --output-dir "$out" \
        --all-datasets --include-datasets "$ds" \
        --time-limit "$TIME_LIMIT" --dataset-retries "$DATASET_RETRIES" \
        >>"$LOGDIR/classical_${opt}_${ds}_${TS}.log" 2>&1
      echo "[$(date -Is)] DONE $opt $tt/$ds/$algo" >>"$MASTER_LOG"
    ) &
  done
  wait
  log "CLASSICAL done"
}

# ---------- LLM parallel pool ----------
declare -A ADAPTER_FOR=(
  [s1a_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1a_ep7p5_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1a_ep7p5_r3]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5"
  [s1b_ep7p5]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5"
  [dpo_A]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_A"
  [dpo_B]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_B"
  [dpo_C]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_C"
  [dpo_D]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5_dpo_D"
  [dpo_E]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_E"
  [dpo_F]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_F"
  [dpo_G]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_G"
  [dpo_G_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_G"
  [dpo_H]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_H"
  [s1a_s2_C]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_C"
  [s1a_s2_G]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_G"
  [s1a_s2_D1]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1"
  [s1a_s2_D1_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1"
  [s1a_s2_D1m]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D1m"
  [s1a_s2_D2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D2"
  [s1a_s2_D3]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_D3"
  [s1a_s2_O1]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O1"
  [s1a_s2_O3]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O3"
  [s1a_s2_O4]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_dpo_O4"
  [s1a_r0a]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0a"
  [s1a_r0b]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0b"
  [s1a_r0c]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0c"
  [s1a_r0d]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_r0d"
  [s1a_a_boseed_global]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_a_boseed_global"
  [s1a_a_boseed_dsseq]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_a_boseed_dsseq"
  [s1a_b_prefix_grpo]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_grpo"
  [s1a_b_prefix_dsseq]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq"
  [s1a_b_prefix_dsseq_hg8]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_hg8"
  [s1a_b_prefix_dsseq_m32]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_m32"
  [dpo_rationale_C]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_C"
  [dpo_rationale_G]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G"
  [dpo_rationale_G_r2]="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_dpo_rationale_G"
)
declare -A RESPFMT_FOR=(
  [s1a_ep7p5]=json_only
  [s1a_ep7p5_r2]=json_only
  [s1a_ep7p5_r3]=json_only
  [s1b_ep7p5]=rationale_json
  [dpo_A]=rationale_json
  [dpo_B]=rationale_json
  [dpo_C]=rationale_json
  [dpo_D]=rationale_json
  [dpo_E]=rationale_json
  [dpo_F]=rationale_json
  [dpo_G]=rationale_json
  [dpo_G_r2]=rationale_json
  [dpo_H]=rationale_json
  [s1a_s2_C]=json_only
  [s1a_s2_G]=json_only
  [s1a_s2_D1]=json_only
  [s1a_s2_D1_r2]=json_only
  [s1a_s2_D1m]=json_only
  [s1a_s2_D2]=json_only
  [s1a_s2_D3]=json_only
  [s1a_s2_O1]=json_only
  [s1a_s2_O3]=json_only
  [s1a_s2_O4]=json_only
  [s1a_r0a]=json_only
  [s1a_r0b]=json_only
  [s1a_r0c]=json_only
  [s1a_r0d]=json_only
  [s1a_a_boseed_global]=json_only
  [s1a_a_boseed_dsseq]=json_only
  [s1a_b_prefix_grpo]=json_only
  [s1a_b_prefix_dsseq]=json_only
  [s1a_b_prefix_dsseq_hg8]=json_only
  [s1a_b_prefix_dsseq_m32]=json_only
  [dpo_rationale_C]=rationale_json
  [dpo_rationale_G]=rationale_json
  [dpo_rationale_G_r2]=rationale_json
)

EVAL_GPU_MAP="${EVAL_GPU_MAP:-}"  # e.g. 0,1 or 2,3 — slot index -> physical GPU id
JOBS_PER_GPU_MAP="${JOBS_PER_GPU_MAP:-}"  # e.g. 2,3,3 — per-slot worker cap (defaults to JOBS_PER_GPU)

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

jobs_per_gpu_for_slot() {
  local slot="$1"
  if [[ -n "$JOBS_PER_GPU_MAP" ]]; then
    IFS=',' read -ra _jmap <<<"$JOBS_PER_GPU_MAP"
    echo "${_jmap[$slot]:-$JOBS_PER_GPU}"
  else
    echo "$JOBS_PER_GPU"
  fi
}

run_llm_job() {
  local model="$1" ds="$2" algo="$3" opt="$4" gpu="$5"
  local pdev
  pdev="$(physical_gpu_for_slot "$gpu")"
  local adapter="${ADAPTER_FOR[$model]}"
  local resp="${RESPFMT_FOR[$model]}"
  local out="${OUTPUT_ROOT}_p5_${model}"
  local logf="$LOGDIR/llm_${model}_${opt}_${HYBRID}_${ds}_${TS}.log"
  mkdir -p "$out"
  log "LLM gpu=$gpu (phys $pdev) model=$model opt=$opt $ds/$algo"
  CUDA_VISIBLE_DEVICES="$pdev" "$PYTHON_LLM" examples/benchmark/run_toy_categorical_optimizers.py \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval \
    --llm-backends qwen-sft6 --qwen-backend hf \
    --qwen-model-path "$BASE_MODEL" \
    --qwen-adapter-path "$adapter" \
    --llm-max-tokens 768 \
    --llm-response-format "$resp" \
    --output-dir "$out" \
    --all-datasets --include-datasets "$ds" \
    --time-limit "$TIME_LIMIT" --dataset-retries "$DATASET_RETRIES" \
    --optimizer "$opt" \
    --prompt-variant "$PROMPT_VARIANT" \
    --hybrid-variant "$HYBRID" \
    "${RESUME_FLAG[@]}" >>"$logf" 2>&1
}

run_llm_queue() {
  local -n models_ref=$1
  local label="$2"
  local lora_jobs=()
  local model t tt ds algo opt
  read -r -a _llm_opts <<<"$LLM_OPTIMIZERS"
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    for model in "${models_ref[@]}"; do
      for opt in "${_llm_opts[@]}"; do
        lora_jobs+=("$model|$ds|$algo|$opt")
      done
    done
  done
  log "$label queue: ${#lora_jobs[@]} jobs (models×[${LLM_OPTIMIZERS}]×40) P${PROMPT_VARIANT#P}/${HYBRID}"

  # require adapters
  for model in "${models_ref[@]}"; do
    local w="${ADAPTER_FOR[$model]}/adapter_model.safetensors"
    if [[ ! -f "$w" ]]; then
      log "ERROR missing adapter $w — skip model $model"
    fi
  done

  declare -A SLOT_PID=()
  local queue_idx=0 total=${#lora_jobs[@]}
  local gpu slot pid job jpg

  while [[ $queue_idx -lt $total ]] || {
      local any=0
      for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        jpg="$(jobs_per_gpu_for_slot "$gpu")"
        for ((slot=0; slot<jpg; slot++)); do
          is_alive "${SLOT_PID[${gpu}_${slot}]:-}" && any=1
        done
      done
      [[ $any -eq 1 ]]
    }; do
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      jpg="$(jobs_per_gpu_for_slot "$gpu")"
      for ((slot=0; slot<jpg; slot++)); do
        pid="${SLOT_PID[${gpu}_${slot}]:-}"
        if is_alive "$pid"; then continue; fi
        SLOT_PID[${gpu}_${slot}]=
        [[ $queue_idx -ge $total ]] && continue
        # packing: slot0 always if free; slot1+ needs free VRAM
        local count=0 s
        for ((s=0; s<jpg; s++)); do
          is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count+1))
        done
        [[ $count -lt $jpg ]] || continue
        if [[ $count -ge 1 ]]; then
          local free pdev
          pdev="$(physical_gpu_for_slot "$gpu")"
          free=$(gpu_free_mb "$pdev")
          [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
        fi
        job="${lora_jobs[$queue_idx]}"
        queue_idx=$((queue_idx+1))
        IFS='|' read -r model ds algo opt <<<"$job"
        if [[ ! -f "${ADAPTER_FOR[$model]}/adapter_model.safetensors" ]]; then
          log "SKIP missing $model ($queue_idx/$total)"
          continue
        fi
        log "dispatch $queue_idx/$total gpu=$gpu $model $opt $ds"
        ( run_llm_job "$model" "$ds" "$algo" "$opt" "$gpu" ) &
        SLOT_PID[${gpu}_${slot}]=$!
      done
    done
    sleep "$POLL_SEC"
  done
  wait
  log "$label done"
}

run_sft() {
  local models=(s1a_ep7p5 s1b_ep7p5)
  run_llm_queue models "SFT"
}

run_dpo() {
  local models=(dpo_A dpo_B dpo_C dpo_D dpo_E dpo_F dpo_G dpo_H)
  run_llm_queue models "DPO"
}

run_s1a_s2_cg() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  for model in s1a_s2_C s1a_s2_G; do
    local -a one=("$model")
    run_llm_queue one "S1A_S2_${model}"
  done
}

run_s1a_s2_c() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_s2_C)
  run_llm_queue one "S1A_S2_C"
}

run_s1a_s2_g() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_s2_G)
  run_llm_queue one "S1A_S2_G"
}

run_s1a_s2_conservative() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local models=(s1a_s2_D1 s1a_s2_D2 s1a_s2_D3 s1a_s2_O1 s1a_s2_O3 s1a_s2_O4)
  for model in "${models[@]}"; do
    local -a one=("$model")
    run_llm_queue one "S1A_S2_${model}"
  done
}

run_s1a_s2_o3() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_s2_O3)
  run_llm_queue one "S1A_S2_O3"
}

run_s1a_s2_micro() {
  # Half-epoch micros: ORPO-O3 + DPO-D1m — same queue so eval packs in parallel
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_s2_O3 s1a_s2_D1m)
  run_llm_queue models "S1A_S2_MICRO"
}

run_s1a_r0_light() {
  # REINFORCE lights R0a (T=0.7) + R0b (T=1.0) after on-policy update
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_r0a s1a_r0b)
  run_llm_queue models "S1A_R0_LIGHT"
}

run_s1a_r0_medium() {
  # REINFORCE mediums R0c (T=0.7) + R0d (T=1.0) on picked-32
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_r0c s1a_r0d)
  run_llm_queue models "S1A_R0_MEDIUM"
}

run_s1a_setup_ab() {
  # BO-seeded Setup A (global + dsseq) and prefix Setup B (grpo + dsseq)
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_a_boseed_global s1a_a_boseed_dsseq s1a_b_prefix_grpo s1a_b_prefix_dsseq)
  run_llm_queue models "S1A_SETUP_AB"
}

run_s1a_b_prefix_dsseq_hg8() {
  # Setup B dsseq mid-ckpt (top-8 high-growth) and/or full medium-32 final
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_b_prefix_dsseq_hg8)
  run_llm_queue models "S1A_B_PREFIX_DSSEQ_HG8"
}

run_s1a_b_prefix_dsseq_m32() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_b_prefix_dsseq_m32)
  run_llm_queue models "S1A_B_PREFIX_DSSEQ_M32"
}

run_s1a_b_prefix_dsseq_hg8_m32() {
  # Both mid (hg8) and final (medium32) from one train run
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a models=(s1a_b_prefix_dsseq_hg8 s1a_b_prefix_dsseq_m32)
  run_llm_queue models "S1A_B_PREFIX_DSSEQ_HG8_M32"
}

run_from_manifest() {
  # MANIFEST_JSON: [{"id":"model_id","adapter":"/path","resp":"json_only"}, ...]
  local mf="${MANIFEST_JSON:-}"
  if [[ -z "$mf" || ! -f "$mf" ]]; then
    log "ERROR: STAGE=from_manifest requires MANIFEST_JSON pointing to a JSON list"
    exit 1
  fi
  LLM_OPTIMIZERS="${LLM_OPTIMIZERS:-ptllm_tpe_qwen}"
  PROMPT_VARIANT="${PROMPT_VARIANT:-P5}"
  HYBRID="${HYBRID:-H0}"
  local -a models=()
  while IFS= read -r mid; do
    [[ -n "$mid" ]] || continue
    models+=("$mid")
    local adapter resp
    adapter="$("$PYTHON_CLASS" -c "import json; m=json.load(open('$mf')); print(next(x['adapter'] for x in m if x['id']=='$mid'))")"
    resp="$("$PYTHON_CLASS" -c "import json; m=json.load(open('$mf')); print(next(x.get('resp','json_only') for x in m if x['id']=='$mid'))")"
    ADAPTER_FOR["$mid"]="$adapter"
    RESPFMT_FOR["$mid"]="$resp"
  done < <("$PYTHON_CLASS" -c "import json; print('\n'.join(x['id'] for x in json.load(open('$mf'))))")
  if [[ ${#models[@]} -eq 0 ]]; then
    log "ERROR: empty manifest $mf"
    exit 1
  fi
  run_llm_queue models "FROM_MANIFEST"
}

run_s1b_dpo_rationale_cg() {
  # s1b rationale-aligned DPO-C + ORPO-G (p5_s2_dpo_rationale)
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  for model in dpo_rationale_C dpo_rationale_G; do
    local -a one=("$model")
    run_llm_queue one "S1B_DPO_RATIONALE_${model}"
  done
}

# Replication: same s1a adapter, fresh output dir (use SEED=2 vs original seed=1)
run_s1a_ep7p5_tpe_r2() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_ep7p5_r2)
  run_llm_queue one "S1A_EP7P5_TPE_R2"
}

# Replication: same s1a adapter, fresh output dir (use SEED=3 vs original seed=1)
run_s1a_ep7p5_tpe_r3() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  local -a one=(s1a_ep7p5_r3)
  run_llm_queue one "S1A_EP7P5_TPE_R3"
}

# Seed=2 replications for leaderboard top arms (same adapters, fresh dirs)
run_seed2_top3() {
  LLM_OPTIMIZERS="ptllm_tpe_qwen"
  PROMPT_VARIANT=P5
  HYBRID=H0
  for model in dpo_G_r2 s1a_s2_D1_r2 dpo_rationale_G_r2; do
    local -a one=("$model")
    run_llm_queue one "SEED2_${model}"
  done
}

log "=== algo40 eval STAGE=$STAGE HYBRID=$HYBRID PROMPT=$PROMPT_VARIANT LLM_OPTIMIZERS=$LLM_OPTIMIZERS n_trials=$N_TRIALS SEED=$SEED ==="

case "$STAGE" in
  classical) run_classical ;;
  sft) run_sft ;;
  dpo) run_dpo ;;
  s1a_s2_cg) run_s1a_s2_cg ;;
  s1a_s2_c) run_s1a_s2_c ;;
  s1a_s2_g) run_s1a_s2_g ;;
  s1a_s2_conservative) run_s1a_s2_conservative ;;
  s1a_s2_o3) run_s1a_s2_o3 ;;
  s1a_s2_micro) run_s1a_s2_micro ;;
  s1a_r0_light) run_s1a_r0_light ;;
  s1a_r0_medium) run_s1a_r0_medium ;;
  s1a_setup_ab) run_s1a_setup_ab ;;
  s1a_b_prefix_dsseq_hg8) run_s1a_b_prefix_dsseq_hg8 ;;
  s1a_b_prefix_dsseq_m32) run_s1a_b_prefix_dsseq_m32 ;;
  s1a_b_prefix_dsseq_hg8_m32) run_s1a_b_prefix_dsseq_hg8_m32 ;;
  from_manifest) run_from_manifest ;;
  s1b_dpo_rationale_cg) run_s1b_dpo_rationale_cg ;;
  s1a_ep7p5_tpe_r2) run_s1a_ep7p5_tpe_r2 ;;
  s1a_ep7p5_tpe_r3) run_s1a_ep7p5_tpe_r3 ;;
  seed2_top3) run_seed2_top3 ;;
  all)
    # classical on CPU in parallel with SFT on GPU
    run_classical &
    class_pid=$!
    run_sft
    wait "$class_pid" || true
    run_dpo
    ;;
  *) log "unknown STAGE=$STAGE"; exit 1 ;;
esac

log "=== algo40 eval complete ==="
