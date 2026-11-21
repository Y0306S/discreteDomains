#!/usr/bin/env bash
# Top-up missing v5 GRPO eval datasets (RS3f05v5 + RS5f05v5, HPUCB_arm + ptllm_tpe_qwen).
# Uses openbox_qwen_eval (GPU PyTorch) — NOT openbox_eval (CPU-only).
#
# Usage:
#   bash examples/benchmark/wait_eval_grpo_rsmix_v12_v5_topup.sh
#   DRY_RUN=1 bash ...   # print jobs only
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

PY_EVAL="${PY_EVAL:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
[[ -x "$PY_EVAL" ]] || { echo "missing $PY_EVAL"; exit 1; }

BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
OUTPUT_BASE="${OUTPUT_BASE:-benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12}"
LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-10800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
POLL_SEC="${POLL_SEC:-20}"
DRY_RUN="${DRY_RUN:-0}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOG_ROOT/eval_grpo_rsmix_v12_v5_topup_${TS}.log"
JOBLIST="$LOG_ROOT/eval_grpo_rsmix_v12_v5_topup_${TS}.jobs.tsv"
mkdir -p "$LOG_ROOT" "$(dirname "$OUTPUT_BASE")"

log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

"$PY_EVAL" - <<PY >"$JOBLIST"
import json
from pathlib import Path

OUTPUT_BASE = "${OUTPUT_BASE}"
tasks = json.loads(Path("benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text())
unique_ds = sorted({(t["task_type"], t["dataset"]) for t in tasks})

def ok(p: Path) -> bool:
    try:
        json.loads(p.read_text())
        return True
    except Exception:
        return False

prompts = ("P5plusA5T50X50RS3f05v5", "P5plusA5T50X50RS5f05v5")
rows = []
seen = set()
for ckpt in (8, 16, 32):
    mid = f"s1b_attr_t2_b_prefix_grpo_rsmix_v5_pareto32_v12_ckpt{ckpt}"
    root = Path(f"benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_p5_{mid}")
    adapter = (
        f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2"
        f"_b_prefix_grpo_rsmix_v5_pareto32_v12/ckpt{ckpt}"
    )
    for pr in prompts:
        for t in tasks:
            tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
            sj = root / tt / ds / "summary.json"
            if sj.is_file() and ok(sj):
                continue
            key = (mid, pr, tt, ds, algo)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                "\t".join(
                    [
                        mid,
                        str(ckpt),
                        pr,
                        tt,
                        ds,
                        algo,
                        adapter,
                        f"{OUTPUT_BASE}_p5_{mid}",
                    ]
                )
            )
print("\n".join(rows))
PY

n_jobs=$(wc -l <"$JOBLIST" | tr -d ' ')
log "=== v5 GRPO top-up START ($n_jobs jobs) ==="
log "python=$PY_EVAL jobs=$n_jobs gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"
log "joblist=$JOBLIST"

if [[ "$n_jobs" == "0" ]]; then
  log "nothing to do"
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  cat "$JOBLIST" | tee -a "$PIPE"
  exit 0
fi

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null | head -1 | tr -d ' '
}

is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

run_one() {
  local job_line="$1"
  local gpu="$2"
  local model ckpt prompt tt ds algo adapter out
  IFS=$'\t' read -r model ckpt prompt tt ds algo adapter out <<<"$job_line"
  local logf="$LOG_ROOT/v5_topup_${TS}_${model}_${prompt}_${ds}.log"
  log "RUN gpu=$gpu $model $prompt $tt/$ds/$algo"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY_EVAL" examples/benchmark/run_toy_categorical_optimizers.py \
    --task-type "$tt" \
    --dataset "$ds" \
    --algorithm "$algo" \
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval \
    --llm-backends qwen-sft6 --qwen-backend hf \
    --qwen-model-path "$BASE_MODEL" \
    --qwen-adapter-path "$adapter" \
    --llm-max-tokens 768 \
    --llm-response-format json_only \
    --output-dir "$out" \
    --time-limit "$TIME_LIMIT" --dataset-retries "$DATASET_RETRIES" \
    --optimizer ptllm_tpe_qwen \
    --prompt-variant "$prompt" \
    --hybrid-variant HPUCB_arm \
    --resume >>"$logf" 2>&1
}

mapfile -t JOBS <"$JOBLIST"
declare -A SLOT_PID=()
failures=0
queue_idx=0
total=${#JOBS[@]}

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
      if [[ -n "$pid" ]]; then
        if wait "$pid"; then
          log "OK finished gpu=$gpu slot=$slot pid=$pid"
        else
          log "FAIL gpu=$gpu slot=$slot pid=$pid"
          failures=$((failures + 1))
        fi
        SLOT_PID[${gpu}_${slot}]=
      fi
      [[ $queue_idx -ge $total ]] && continue
      count=0
      for ((s=0; s<JOBS_PER_GPU; s++)); do
        is_alive "${SLOT_PID[${gpu}_${s}]:-}" && count=$((count + 1))
      done
      [[ $count -lt $JOBS_PER_GPU ]] || continue
      if [[ $count -ge 1 ]]; then
        free=$(gpu_free_mb "$gpu")
        [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
      fi
      job="${JOBS[$queue_idx]}"
      queue_idx=$((queue_idx + 1))
      log "dispatch $queue_idx/$total gpu=$gpu: $job"
      (
        run_one "$job" "$gpu"
      ) &
      SLOT_PID[${gpu}_${slot}]=$!
    done
  done
  sleep "$POLL_SEC"
done

log "=== verify completion ==="
remaining=$("$PY_EVAL" - <<'PY'
import json
from pathlib import Path

tasks = json.loads(Path("benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json").read_text())
unique_ds = sorted({(t["task_type"], t["dataset"]) for t in tasks})

def ok(p):
    try:
        json.loads(p.read_text())
        return True
    except Exception:
        return False

prompts = ("P5plusA5T50X50RS3f05v5", "P5plusA5T50X50RS5f05v5")
missing = 0
for ckpt in (8, 16, 32):
    mid = f"s1b_attr_t2_b_prefix_grpo_rsmix_v5_pareto32_v12_ckpt{ckpt}"
    root = Path(f"benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_p5_{mid}")
    for pr in prompts:
        for tt, ds in unique_ds:
            sj = root / tt / ds / "summary.json"
            if not (sj.is_file() and ok(sj)):
                missing += 1
                print(f"MISSING {mid} {pr} {tt}/{ds}")
print(f"REMAINING {missing}")
PY
)
log "$remaining"

if [[ "$failures" -gt 0 ]]; then
  log "=== v5 top-up DONE with $failures job failure(s) ==="
  exit 1
fi

log "=== rebuild tables ==="
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== v5 GRPO top-up DONE ==="
log "pipe=$PIPE"
echo "$PIPE"
