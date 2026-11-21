#!/usr/bin/env bash
# Pareto-32 Setup-B rs_mix v12 + GRPO:
#   1) v3: full 12-prefix collect (or skip if done)
#      v4/v5: rs_warm only (0,5,10) then merge shared k!=0,5,10 from v3 buffer
#   2) Train b_prefix_grpo once per version from s1b_ep7p5_attr_t2; ckpt8/16/32
#   3) Eval all 9 adapters via STAGE=from_manifest
#
# Usage:
#   bash examples/benchmark/wait_b_prefix_grpo_rsmix_v12_pareto32_ckpt_sweep.sh
#   SKIP_EVAL=1 bash ...
#   SKIP_TRAIN=1 bash ...          # collect only
#   RS_VERSIONS="v3" bash ...
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
PY_EVAL="${PY_EVAL:-$EVAL_ENV/bin/python}"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/picked_32_pareto_tasks.json}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/picked_32_pareto_train_order.json}"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12_pareto32_ckpt81632_manifest.json}"

RS_VERSIONS="${RS_VERSIONS:-v3 v4 v5}"
PREFIXES="${PREFIXES:-0 5 10 20 30 40 50 60 70 80 90 100}"
RS_PREFIXES="${RS_PREFIXES:-0 5 10}"
N_SAMPLES="${N_SAMPLES:-4}"
N_SAMPLES_MAX="${N_SAMPLES_MAX:-10}"
TEMP="${TEMP:-1.0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"

# Expected floor: 32 tasks × 12 prefixes × 4 samples = 1536 (extensions add more)
MIN_RECORDS="${MIN_RECORDS:-1536}"
# rs_warm-only collect floor: 32 × 3 prefixes × 4 samples
MIN_RS_RECORDS="${MIN_RS_RECORDS:-384}"
SKIP_ROLLOUTS="${SKIP_ROLLOUTS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"
CHECKPOINT_AT="${CHECKPOINT_AT:-8,16,32}"
LR="${LR:-1e-6}"
KL_BETA="${KL_BETA:-0.05}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/b_prefix_grpo_rsmix_v12_pareto32_ckpt_sweep_${TS}.log"
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

count_rs_tasks() {
  local root="$1"
  local min_per_task="$2"
  "$PY_TRAIN" - <<PY
import json
from pathlib import Path
root = Path("$root")
need = int("$min_per_task")
n = 0
if root.exists():
    for task in root.iterdir():
        if not task.is_dir():
            continue
        rec = 0
        for p in task.rglob("rollouts_*.jsonl"):
            rec += sum(1 for ln in p.open() if ln.strip())
        if rec >= need:
            n += 1
print(n)
PY
}

expected_task_count() {
  "$PY_TRAIN" - <<PY
import json
from pathlib import Path
tasks = json.loads(Path("$TASKS_JSON").read_text())
print(len(tasks))
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

log "=== b_prefix_grpo rs_mix v12 pareto32 ckpt sweep ==="
log "INIT=$INIT RS_VERSIONS=$RS_VERSIONS CHECKPOINT_AT=$CHECKPOINT_AT"
log "SKIP_ROLLOUTS=$SKIP_ROLLOUTS SKIP_TRAIN=$SKIP_TRAIN SKIP_EVAL=$SKIP_EVAL"
log "TASKS=$TASKS_JSON ORDER=$ORDER PREFIXES=$PREFIXES n=$N_SAMPLES max=$N_SAMPLES_MAX"

# ---------- Phase 1: rollouts ----------
V3_ROOT="benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T${TEMP/./p}_pareto32_v12"
N_TASKS="$(expected_task_count)"
RS_EXPECT_PER_TASK="$(( $(wc -w <<<"$RS_PREFIXES") * N_SAMPLES ))"
if [[ "$SKIP_ROLLOUTS" != "1" ]]; then
  for ver in $RS_VERSIONS; do
    OUT_ROOT="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12"
    n=$(count_records "$OUT_ROOT")
    if (( n >= MIN_RECORDS )); then
      log "SKIP rollouts rs=$ver already have records=$n (>= $MIN_RECORDS) in $OUT_ROOT"
      continue
    fi

    if [[ "$ver" == "v3" ]]; then
      log "=== rollouts rs=v3 full mix -> $OUT_ROOT (have $n / need $MIN_RECORDS) ==="
      RS_VERSION="$ver" \
        TASKS_JSON="$TASKS_JSON" \
        OUT_ROOT="$OUT_ROOT" \
        PREFIXES="$PREFIXES" \
        N_SAMPLES="$N_SAMPLES" \
        N_SAMPLES_MAX="$N_SAMPLES_MAX" \
        TEMP="$TEMP" \
        ADAPTER="$INIT" \
        GPU_IDS="$GPU_IDS" \
        JOBS_PER_GPU="$JOBS_PER_GPU" \
        PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
        EXPECT_PER_TASK="$(( $(wc -w <<<"$PREFIXES") * N_SAMPLES ))" \
        bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh | tee -a "$PIPE"
    else
      if (( $(count_records "$V3_ROOT") < MIN_RECORDS )); then
        log "ERROR need completed v3 buffer at $V3_ROOT before rs=$ver merge"
        exit 1
      fi
      RS_ROOT="benchmark_data/rl_rollouts/B_rsmix_${ver}_rswarm_T${TEMP/./p}_pareto32_v12"
      rs_n=$(count_records "$RS_ROOT")
      rs_tasks=$(count_rs_tasks "$RS_ROOT" "$RS_EXPECT_PER_TASK")
      if (( rs_tasks < N_TASKS )); then
        log "=== rollouts rs=$ver rs_warm only -> $RS_ROOT (tasks $rs_tasks/$N_TASKS records=$rs_n) ==="
        RS_VERSION="$ver" \
          TASKS_JSON="$TASKS_JSON" \
          OUT_ROOT="$RS_ROOT" \
          PREFIXES="$RS_PREFIXES" \
          N_SAMPLES="$N_SAMPLES" \
          N_SAMPLES_MAX="$N_SAMPLES_MAX" \
          TEMP="$TEMP" \
          ADAPTER="$INIT" \
          GPU_IDS="$GPU_IDS" \
          JOBS_PER_GPU="$JOBS_PER_GPU" \
          PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
          EXPECT_PER_TASK="$(( $(wc -w <<<"$RS_PREFIXES") * N_SAMPLES ))" \
          bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh | tee -a "$PIPE"
        rs_n=$(count_records "$RS_ROOT")
        rs_tasks=$(count_rs_tasks "$RS_ROOT" "$RS_EXPECT_PER_TASK")
        if (( rs_tasks < N_TASKS )); then
          log "ERROR rs=$ver rs_warm tasks=$rs_tasks/$N_TASKS records=$rs_n"
          exit 1
        fi
      else
        log "SKIP rs_warm collect rs=$ver tasks=$rs_tasks/$N_TASKS records=$rs_n in $RS_ROOT"
      fi
      log "=== merge rs=$ver shared from v3 + rs_warm from $RS_ROOT -> $OUT_ROOT ==="
      "$PY_TRAIN" examples/benchmark/merge_rsmix_v12_shared_from_v3.py \
        --shared-from "$V3_ROOT" \
        --rs-from "$RS_ROOT" \
        --out "$OUT_ROOT" \
        --rs-version "$ver" | tee -a "$PIPE"
    fi

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

# ---------- Phase 2: train (3 parallel GRPO) ----------
declare -a TRAIN_PIDS=()
declare -a TRAIN_VERS=()
if [[ "$SKIP_TRAIN" != "1" ]]; then
  export PATH="$LF_ENV/bin:$PATH"
  read -ra TGPU_ARR <<<"$TRAIN_GPUS"
  i=0
  for ver in $RS_VERSIONS; do
    BUF="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12"
    OUT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_${ver}_pareto32_v12"
    gpu="${TGPU_ARR[$((i % ${#TGPU_ARR[@]}))]}"
    i=$((i + 1))

    if [[ -f "$OUT/ckpt8/adapter_model.safetensors" && -f "$OUT/ckpt16/adapter_model.safetensors" && -f "$OUT/ckpt32/adapter_model.safetensors" ]]; then
      log "SKIP train rs=$ver — ckpt8/16/32 already present in $OUT"
      continue
    fi

    n=$(count_records "$BUF")
    if (( n < MIN_RECORDS )); then
      log "ERROR buffer $BUF records=$n < $MIN_RECORDS — cannot train"
      exit 1
    fi

    log "=== train rs=$ver GPU=$gpu buffer=$BUF out=$OUT mode=grpo_group ckpt@$CHECKPOINT_AT ==="
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
        --buffer-root "$BUF" \
        --init-adapter "$INIT" \
        --output-dir "$OUT" \
        --checkpoint-root "$OUT" \
        --checkpoint-at "$CHECKPOINT_AT" \
        --dataset-order-file "$ORDER" \
        --strict-dataset-order \
        --advantage-mode grpo_group \
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
    root = Path(f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12")
    for k in ckpts:
        mid = f"s1b_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12_ckpt{k}"
        adapter = root / f"ckpt{k}"
        if k == max(ckpts) and not (adapter / "adapter_model.safetensors").exists():
            if (root / "adapter_model.safetensors").exists():
                adapter = root
        rows.append({
            "id": mid,
            "adapter": str(adapter),
            "resp": "json_only",
            "rs_version": ver,
            "ckpt": k,
            "recipe": "b_prefix_grpo",
            "init": "s1b_ep7p5_attr_t2",
            "tasks": "picked_32_pareto",
            "schedule": "rs_mix_v12",
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
