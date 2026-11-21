#!/usr/bin/env bash
# Pareto-32 Setup-B rs_mix RL:
#   1) Collect rollouts for RS v3 / v4 / v5 (prefixes 3,10,50,60,70,80)
#   2) Train b_prefix_dsseq once per version; save ckpt8 / ckpt16 / ckpt32
#   3) Eval all 9 adapters via STAGE=from_manifest
#
# Usage:
#   bash examples/benchmark/wait_b_prefix_dsseq_rsmix_pareto32_ckpt_sweep.sh
#   SKIP_EVAL=1 bash ...          # stop after train
#   RS_VERSIONS="v3" bash ...     # only one RS version
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
PY_EVAL="${PY_EVAL:-$EVAL_ENV/bin/python}"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/picked_32_pareto_tasks.json}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/picked_32_pareto_train_order.json}"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/b_prefix_dsseq_rsmix_pareto32_ckpt81632_manifest.json}"

RS_VERSIONS="${RS_VERSIONS:-v3 v4 v5}"
PREFIXES="${PREFIXES:-3 10 50 60 70 80}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMP="${TEMP:-1.0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"

# Expected: 32 tasks × 6 prefixes × 4 samples = 768
MIN_RECORDS="${MIN_RECORDS:-768}"
SKIP_ROLLOUTS="${SKIP_ROLLOUTS:-0}"
# Default: collect-only (no train/eval). Set SKIP_TRAIN=0 SKIP_EVAL=0 to resume later.
SKIP_TRAIN="${SKIP_TRAIN:-1}"
SKIP_EVAL="${SKIP_EVAL:-1}"

TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"  # one GPU per RS version train
CHECKPOINT_AT="${CHECKPOINT_AT:-8,16,32}"
LR="${LR:-1e-6}"
KL_BETA="${KL_BETA:-0.05}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/b_prefix_dsseq_rsmix_pareto32_ckpt_sweep_${TS}.log"
mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

count_records() {
  local root="$1"
  "$PY_TRAIN" - <<PY
from pathlib import Path
root = Path("$root")
n = 0
if root.exists():
    for p in root.rglob("rollouts_*.jsonl"):
        n += sum(1 for ln in p.open() if ln.strip())
print(n)
PY
}

if [[ ! -f "$INIT/adapter_model.safetensors" ]]; then
  log "ERROR missing init adapter: $INIT"
  exit 1
fi
if [[ ! -f "$TASKS_JSON" ]]; then
  log "ERROR missing tasks: $TASKS_JSON"
  exit 1
fi
if [[ ! -f "$ORDER" ]]; then
  log "ERROR missing order: $ORDER"
  exit 1
fi

log "=== b_prefix_dsseq rs_mix pareto32 ckpt sweep ==="
log "RS_VERSIONS=$RS_VERSIONS CHECKPOINT_AT=$CHECKPOINT_AT SKIP_ROLLOUTS=$SKIP_ROLLOUTS SKIP_TRAIN=$SKIP_TRAIN SKIP_EVAL=$SKIP_EVAL"
log "TASKS=$TASKS_JSON ORDER=$ORDER"

# ---------- Phase 1: rollouts ----------
if [[ "$SKIP_ROLLOUTS" != "1" ]]; then
  for ver in $RS_VERSIONS; do
    OUT_ROOT="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4_T${TEMP/./p}_pareto32"
    n=$(count_records "$OUT_ROOT")
    if (( n >= MIN_RECORDS )); then
      log "SKIP rollouts rs=$ver already have records=$n (>= $MIN_RECORDS) in $OUT_ROOT"
      continue
    fi
    log "=== rollouts rs=$ver -> $OUT_ROOT (have $n / need $MIN_RECORDS) ==="
    RS_VERSION="$ver" \
      TASKS_JSON="$TASKS_JSON" \
      OUT_ROOT="$OUT_ROOT" \
      PREFIXES="$PREFIXES" \
      N_SAMPLES="$N_SAMPLES" \
      TEMP="$TEMP" \
      ADAPTER="$INIT" \
      GPU_IDS="$GPU_IDS" \
      JOBS_PER_GPU="$JOBS_PER_GPU" \
      PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
      bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh | tee -a "$PIPE"
    n=$(count_records "$OUT_ROOT")
    log "rollouts rs=$ver done records=$n"
    if (( n < MIN_RECORDS )); then
      log "ERROR rs=$ver records=$n < $MIN_RECORDS"
      exit 1
    fi
  done
else
  log "SKIP_ROLLOUTS=1"
fi

# ---------- Phase 2: train (3 parallel) ----------
declare -a TRAIN_PIDS=()
declare -a TRAIN_VERS=()
if [[ "$SKIP_TRAIN" != "1" ]]; then
  export PATH="$LF_ENV/bin:$PATH"
  read -ra TGPU_ARR <<<"$TRAIN_GPUS"
  i=0
  for ver in $RS_VERSIONS; do
    BUF="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4_T${TEMP/./p}_pareto32"
    OUT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_rsmix_${ver}_pareto32"
    gpu="${TGPU_ARR[$((i % ${#TGPU_ARR[@]}))]}"
    i=$((i + 1))

    # already have all ckpts?
    if [[ -f "$OUT/ckpt8/adapter_model.safetensors" && -f "$OUT/ckpt16/adapter_model.safetensors" && -f "$OUT/ckpt32/adapter_model.safetensors" ]]; then
      log "SKIP train rs=$ver — ckpt8/16/32 already present in $OUT"
      continue
    fi

    n=$(count_records "$BUF")
    if (( n < MIN_RECORDS )); then
      log "ERROR buffer $BUF records=$n < $MIN_RECORDS — cannot train"
      exit 1
    fi

    log "=== train rs=$ver GPU=$gpu buffer=$BUF out=$OUT ckpt@$CHECKPOINT_AT ==="
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
        --buffer-root "$BUF" \
        --init-adapter "$INIT" \
        --output-dir "$OUT" \
        --checkpoint-root "$OUT" \
        --checkpoint-at "$CHECKPOINT_AT" \
        --dataset-order-file "$ORDER" \
        --strict-dataset-order \
        --advantage-mode per_dataset \
        --dataset-sequential \
        --lr "$LR" --kl-beta "$KL_BETA" --grad-accum "$GRAD_ACCUM" --seed 1
    ) >>"$PIPE" 2>&1 &
    TRAIN_PIDS+=("$!")
    TRAIN_VERS+=("$ver")
  done

  fail=0
  for j in "${!TRAIN_PIDS[@]}"; do
    pid="${TRAIN_PIDS[$j]}"
    ver="${TRAIN_VERS[$j]}"
    if wait "$pid"; then
      log "train rs=$ver OK pid=$pid"
    else
      log "ERROR train rs=$ver FAILED pid=$pid"
      fail=1
    fi
  done
  if (( fail )); then
    log "ERROR: one or more trains failed"
    exit 1
  fi
else
  log "SKIP_TRAIN=1"
fi

# ---------- Phase 3: write 9-adapter manifest + eval ----------
log "=== write eval manifest -> $MANIFEST ==="
"$PY_TRAIN" - <<PY
import json
from pathlib import Path
versions = "$(echo $RS_VERSIONS)".split()
ckpts = [int(x) for x in "$CHECKPOINT_AT".split(",") if x.strip()]
rows = []
for ver in versions:
    root = Path(f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_prefix_dsseq_rsmix_{ver}_pareto32")
    for k in ckpts:
        mid = f"s1a_b_prefix_dsseq_rsmix_{ver}_pareto32_ckpt{k}"
        adapter = root / f"ckpt{k}"
        # fallback: final dir if ckpt32 missing but output-dir has adapter
        if k == max(ckpts) and not (adapter / "adapter_model.safetensors").exists():
            if (root / "adapter_model.safetensors").exists():
                adapter = root
        rows.append({
            "id": mid,
            "adapter": str(adapter),
            "resp": "json_only",
            "rs_version": ver,
            "ckpt": k,
            "recipe": "b_prefix_dsseq",
            "tasks": "picked_32_pareto",
            "schedule": "rs_mix",
        })
missing = [r for r in rows if not (Path(r["adapter"]) / "adapter_model.safetensors").exists()]
Path("$MANIFEST").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} adapters -> $MANIFEST")
if missing:
    print("MISSING adapters:")
    for r in missing:
        print(" ", r["id"], r["adapter"])
    raise SystemExit(1)
PY

if [[ "$SKIP_EVAL" == "1" ]]; then
  log "SKIP_EVAL=1 — manifest ready at $MANIFEST"
  log "To eval later:"
  log "  MANIFEST_JSON=$MANIFEST STAGE=from_manifest HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen bash examples/benchmark/run_mltest20_algo40_eval.sh"
  log "pipe=$PIPE"
  exit 0
fi

log "=== algo40 eval all 9 adapters ==="
NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}" PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
  STAGE=from_manifest MANIFEST_JSON="$MANIFEST" \
  HYBRID=H0 PROMPT_VARIANT=P5 LLM_OPTIMIZERS=ptllm_tpe_qwen \
  SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
  PYTHON_LLM="$PY_EVAL" \
  PYTHON_CLASS="${PYTHON_CLASS:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_eval/bin/python}" \
  bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1

log "=== rebuild drop20pct tables ==="
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm >>"$PIPE" 2>&1 || true
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== complete ==="
log "manifest=$MANIFEST"
log "pipe=$PIPE"
echo "$PIPE"
