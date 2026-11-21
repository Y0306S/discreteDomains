#!/usr/bin/env bash
# Landscape RS mesh on tasks_30, Qwen s1b_attr_t2, HPUCB_arm 90→10:
#   RS10 gpsim 0.5  |  RS3 gpsim 0.5  |  RS10 gpsim 0.3  |  RS5 gpsim 0.3
#
#   bash examples/benchmark/run_p5plus_qwen_rs_gpsim_mesh_tasks30.sh
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_30.json}"
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
EVAL_GPU_MAP="${EVAL_GPU_MAP:-0,1,2,3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-18000}"
POLL_SEC="${POLL_SEC:-20}"
EVAL_RESUME="${EVAL_RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/qwen_rs_gpsim_mesh_tasks30_${TS}.log"
HYBRID="${HYBRID:-HPUCB_arm}"

mkdir -p "$LOGDIR" "$OUTPUT_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" ]]; then
  log "ERROR missing adapter $ADAPTER_PATH/adapter_model.safetensors"
  exit 1
fi

"$PYTHON_LLM" - <<PY
from experiment_utils import load_prompt_variants, _resolve_prompt_variant, apply_prompt_settings, apply_hybrid_settings
from argparse import Namespace
from mindware.components.optimizers.hybrid_selector import HybridSelector
from mindware.components.experience.retrieve import (
    ExperienceBank, retrieve_for_run, _load_landscape_cache,
)
from pathlib import Path
import numpy as np

pv = load_prompt_variants()
for pid, stop, floor in [
    ("P5plusA5T50X50RS10f05", 11, 0.5),
    ("P5plusA5T50X50RS3f05", 4, 0.5),
    ("P5plusA5T50X50RS10f03", 11, 0.3),
    ("P5plusA5T50X50RS5f03", 6, 0.3),
]:
    row = _resolve_prompt_variant(pv[pid], pv)
    assert row.get("prompt_blocks", {}).get("experience_retrieve") is True
    assert row.get("experience_retrieve_ranker") == "surrogate"
    assert int(row.get("experience_retrieve_stop_trial") or 0) == stop, (pid, row.get("experience_retrieve_stop_trial"))
    assert abs(float(row.get("experience_retrieve_spearman_floor")) - floor) < 1e-9, (pid, row.get("experience_retrieve_spearman_floor"))
    args = Namespace(llm_history_window=80)
    apply_prompt_settings(args, pid)
    apply_hybrid_settings(args, "HPUCB_arm")
    assert abs(float(args.llm_experience_retrieve_spearman_floor) - floor) < 1e-9
    sel = HybridSelector(mode="pucb", pucb_p1=0.9, pucb_p_end=args.pucb_p_end, pucb_c2=1.4, rng=np.random.RandomState(0))
    assert abs(sel.omega("llm", 0, 80) - 0.9) < 1e-9
    assert abs(sel.omega("llm", 80, 80) - 0.1) < 1e-9
    print(f"ok prompt {pid} stop={stop} gpsim={floor} pucb 0.9→0.1")

cache = _load_landscape_cache("cls")
assert cache is not None
assert cache.pred("catboost", "ada") is not None, "new landscape cache must include eval target ada/catboost"
bank_path = Path("history_data/experience_bank_v2_holdout40.jsonl")
bank = ExperienceBank.load(bank_path)
block, hits = retrieve_for_run(
    bank,
    dataset_id="ada",
    algorithm_id="catboost",
    task_type="classification",
    dataset_profile=None,
    successes=[],
    k=3,
    start_trial=1,
    rerank_after=1,
    current_trial=1,
    stop_trial=11,
    ranker="surrogate",
    spearman_floor=0.5,
)
hits05 = hits
sims = [h.features.get("surrogate_landscape_sim") for h in hits]
srcs = [h.features.get("landscape_source") for h in hits]
assert hits, "RS must return neighbors at trial 1"
assert any(s is not None for s in sims), f"expected landscape sim at trial 1, got {[(h.exp_id, h.features) for h in hits[:3]]}"
assert srcs and srcs[0] == "cache", srcs
print(f"ok trial-1 landscape retrieve n={len(hits)} sims={sims} source={srcs[0]} floor0.5")
_, hits03 = retrieve_for_run(
    bank, dataset_id="ada", algorithm_id="catboost", task_type="classification",
    dataset_profile=None, successes=[], k=3, start_trial=1, rerank_after=1,
    current_trial=1, stop_trial=11, ranker="surrogate", spearman_floor=0.3,
)
print(f"ok floor0.3 n={len(hits03)} vs floor0.5 n={len(hits05)}")
print("ok landscape RS wiring")
PY

mapfile -t TASK_LINES < <("$PYTHON_LLM" - <<PY
import json
from pathlib import Path
tasks = json.loads(Path("$TASKS_JSON").read_text())
assert len(tasks) == 30, len(tasks)
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
  log "DRY_RUN: n_trials=$N_TRIALS single task yeast/lightgbm"
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

log "=== Qwen landscape RS gpsim mesh on tasks_30 ==="
log "output=$OUTPUT_ROOT tasks=${#TASKS[@]} gpus=$NUM_GPUS (phys ${EVAL_GPU_MAP}) jobs_per_gpu=$JOBS_PER_GPU"
log "adapter=$ADAPTER_PATH seed=$SEED n_trials=$N_TRIALS hybrid=$HYBRID 90→10"
log "arms: RS10f05 RS3f05 RS10f03 RS5f03"

jobs=()
for spec in \
  "P5plusA5T50X50RS10f05|$HYBRID" \
  "P5plusA5T50X50RS3f05|$HYBRID" \
  "P5plusA5T50X50RS10f03|$HYBRID" \
  "P5plusA5T50X50RS5f03|$HYBRID"; do
  IFS='|' read -r prompt hybrid <<<"$spec"
  for t in "${TASKS[@]}"; do
    IFS='|' read -r tt ds algo <<<"$t"
    jobs+=("$prompt|$hybrid|$tt|$ds|$algo")
  done
done
log "total jobs=${#jobs[@]} (4 arms × ${#TASKS[@]} tasks)"
run_gpu_pool jobs "RSgpsim-mesh"
log "=== Qwen landscape RS gpsim mesh done ==="
