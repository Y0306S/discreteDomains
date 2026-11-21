#!/usr/bin/env bash
# Pareto-32 Setup-B rs_mix v13 + GRPO:
#   Prefix schedule (21 jobs/dataset):
#     0,5,10 rs_warm | 20-45 normal | 50-100 explore+exploit each
#   adaptive train groups (3+1); ckpt8/16/32 per v3/v4/v5
#
# Usage:
#   bash examples/benchmark/wait_b_prefix_grpo_rsmix_v13_pareto32_ckpt_sweep.sh
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
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v13_pareto32_ckpt81632_manifest.json}"

RS_VERSIONS="${RS_VERSIONS:-v3 v4 v5}"
PREFIXES="${PREFIXES:-0 5 10 20 25 30 35 40 45 50 60 70 80 90 100}"
RS_PREFIXES="${RS_PREFIXES:-0 5 10}"
N_SAMPLES="${N_SAMPLES:-4}"
N_SAMPLES_MAX="${N_SAMPLES_MAX:-10}"
TEMP="${TEMP:-1.0}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
COLLECT_SCHEDULE="${COLLECT_SCHEDULE:-rs_mix_v13}"
TRAIN_MEMBER_POLICY="${TRAIN_MEMBER_POLICY:-adaptive}"

# 32 tasks × 21 jobs × 4 samples
MIN_RECORDS="${MIN_RECORDS:-2688}"
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
PIPE="$LOG_ROOT/b_prefix_grpo_rsmix_v13_pareto32_ckpt_sweep_${TS}.log"
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

log "=== b_prefix_grpo rs_mix v13 pareto32 ckpt sweep ==="
log "INIT=$INIT COLLECT_SCHEDULE=$COLLECT_SCHEDULE TRAIN_MEMBER_POLICY=$TRAIN_MEMBER_POLICY"
log "SKIP_ROLLOUTS=$SKIP_ROLLOUTS SKIP_TRAIN=$SKIP_TRAIN SKIP_EVAL=$SKIP_EVAL"

V3_ROOT="benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T${TEMP/./p}_pareto32_v13"
N_TASKS="$(expected_task_count)"
RS_EXPECT_PER_TASK="$(( $(wc -w <<<"$RS_PREFIXES") * N_SAMPLES ))"

if [[ "$SKIP_ROLLOUTS" != "1" ]]; then
  for ver in $RS_VERSIONS; do
    OUT_ROOT="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v13"
    n=$(count_records "$OUT_ROOT")
    if (( n >= MIN_RECORDS )); then
      log "SKIP rollouts rs=$ver records=$n (>= $MIN_RECORDS) in $OUT_ROOT"
      continue
    fi

    if [[ "$ver" == "v3" ]]; then
      log "=== rollouts rs=v3 full v13 mix -> $OUT_ROOT (have $n / need $MIN_RECORDS) ==="
      RS_VERSION="$ver" \
        TASKS_JSON="$TASKS_JSON" \
        OUT_ROOT="$OUT_ROOT" \
        PREFIXES="$PREFIXES" \
        COLLECT_SCHEDULE="$COLLECT_SCHEDULE" \
        TRAIN_MEMBER_POLICY="$TRAIN_MEMBER_POLICY" \
        N_SAMPLES="$N_SAMPLES" \
        N_SAMPLES_MAX="$N_SAMPLES_MAX" \
        TEMP="$TEMP" \
        ADAPTER="$INIT" \
        GPU_IDS="$GPU_IDS" \
        JOBS_PER_GPU="$JOBS_PER_GPU" \
        PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
        EXPECT_PER_TASK="$(( 21 * N_SAMPLES ))" \
        bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh | tee -a "$PIPE"
    else
      if (( $(count_records "$V3_ROOT") < MIN_RECORDS )); then
        log "ERROR need completed v3 v13 buffer at $V3_ROOT before rs=$ver merge"
        exit 1
      fi
      RS_ROOT="benchmark_data/rl_rollouts/B_rsmix_${ver}_rswarm_T${TEMP/./p}_pareto32_v13"
      rs_n=$(count_records "$RS_ROOT")
      rs_tasks=$(count_rs_tasks "$RS_ROOT" "$RS_EXPECT_PER_TASK")
      if (( rs_tasks < N_TASKS )); then
        log "=== rollouts rs=$ver rs_warm only -> $RS_ROOT (tasks $rs_tasks/$N_TASKS) ==="
        RS_VERSION="$ver" \
          TASKS_JSON="$TASKS_JSON" \
          OUT_ROOT="$RS_ROOT" \
          PREFIXES="$RS_PREFIXES" \
          COLLECT_SCHEDULE="rs_mix" \
          TRAIN_MEMBER_POLICY="$TRAIN_MEMBER_POLICY" \
          N_SAMPLES="$N_SAMPLES" \
          N_SAMPLES_MAX="$N_SAMPLES_MAX" \
          TEMP="$TEMP" \
          ADAPTER="$INIT" \
          GPU_IDS="$GPU_IDS" \
          JOBS_PER_GPU="$JOBS_PER_GPU" \
          PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
          EXPECT_PER_TASK="$RS_EXPECT_PER_TASK" \
          bash examples/benchmark/run_p5plus_rs_mix_prefix_rollouts.sh | tee -a "$PIPE"
      else
        log "SKIP rs_warm collect rs=$ver tasks=$rs_tasks/$N_TASKS in $RS_ROOT"
      fi
      log "=== merge rs=$ver v13 shared from v3 -> $OUT_ROOT ==="
      "$PY_TRAIN" examples/benchmark/merge_rsmix_v13_shared_from_v3.py \
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

if [[ "$TRAIN_MEMBER_POLICY" == "adaptive" && "$SKIP_TRAIN" != "1" ]]; then
  if [[ -d "$V3_ROOT" ]]; then
    log "=== retag v3 buffer train_member -> adaptive ==="
    "$PY_TRAIN" examples/benchmark/retag_rsmix_train_members.py \
      --buffer-root "$V3_ROOT" --policy adaptive --seed 1 | tee -a "$PIPE"
  fi
fi

declare -a TRAIN_PIDS=()
declare -a TRAIN_VERS=()
if [[ "$SKIP_TRAIN" != "1" ]]; then
  export PATH="$LF_ENV/bin:$PATH"
  read -ra TGPU_ARR <<<"$TRAIN_GPUS"
  i=0
  for ver in $RS_VERSIONS; do
    BUF="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v13"
    OUT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_${ver}_pareto32_v13"
    gpu="${TGPU_ARR[$((i % ${#TGPU_ARR[@]}))]}"
    i=$((i + 1))

    if [[ -f "$OUT/ckpt8/adapter_model.safetensors" && -f "$OUT/ckpt16/adapter_model.safetensors" && -f "$OUT/ckpt32/adapter_model.safetensors" ]]; then
      log "SKIP train rs=$ver — ckpts present in $OUT"
      continue
    fi

    log "=== train rs=$ver GPU=$gpu buffer=$BUF out=$OUT policy=$TRAIN_MEMBER_POLICY grpo_group ==="
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
    if wait "${TRAIN_PIDS[$j]}"; then
      log "train ${TRAIN_VERS[$j]} OK"
    else
      log "ERROR train ${TRAIN_VERS[$j]} FAILED"
      fail=1
    fi
  done
  (( fail == 0 )) || exit 1
else
  log "SKIP_TRAIN=1"
fi

log "=== write manifest -> $MANIFEST ==="
"$PY_TRAIN" - <<PY | tee -a "$PIPE"
import json
from pathlib import Path
versions = "$(echo $RS_VERSIONS)".split()
ckpts = [int(x) for x in "$CHECKPOINT_AT".split(",") if x.strip()]
rows = []
for ver in versions:
    root = Path(f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v13")
    for k in ckpts:
        mid = f"s1b_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v13_ckpt{k}"
        adapter = root / f"ckpt{k}"
        rows.append({
            "id": mid,
            "adapter": str(adapter),
            "resp": "json_only",
            "rs_version": ver,
            "ckpt": k,
            "recipe": "b_prefix_grpo",
            "init": "s1b_ep7p5_attr_t2",
            "tasks": "picked_32_pareto",
            "schedule": "rs_mix_v13",
        })
missing = [r for r in rows if not (Path(r["adapter"]) / "adapter_model.safetensors").exists()]
Path("$MANIFEST").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> $MANIFEST")
if missing:
    for r in missing:
        print("MISSING", r["id"], r["adapter"])
    raise SystemExit(1)
PY

if [[ "$SKIP_EVAL" == "1" ]]; then
  log "SKIP_EVAL=1 manifest=$MANIFEST pipe=$PIPE"
  exit 0
fi

OUTPUT_BASE="benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v13"
FULL_MANIFEST="$MANIFEST"

run_eval_batch() {
  local ver="$1" prompt="$2"
  local mf="benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v13_p5plus_${prompt}_manifest_${ver}.json"
  "$PY_TRAIN" - <<PY
import json
from pathlib import Path
rows = [r for r in json.loads(Path("$FULL_MANIFEST").read_text()) if r.get("rs_version") == "$ver"]
Path("$mf").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> $mf")
PY
  log "=== eval v13 $ver prompt=$prompt ==="
  TASKS_JSON=benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json \
    OUTPUT_ROOT="$OUTPUT_BASE" NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}" \
    PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
    STAGE=from_manifest MANIFEST_JSON="$mf" \
    HYBRID=HPUCB_arm PROMPT_VARIANT="$prompt" LLM_OPTIMIZERS=ptllm_tpe_qwen \
    SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
    PYTHON_LLM="$PY_EVAL" PYTHON_CLASS="$PY_EVAL" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

run_eval_batch v3 P5plusA5T50X50RS3f05
run_eval_batch v3 P5plusA5T50X50RS5f05
run_eval_batch v3 P5plusA5T50X50RS10f03
run_eval_batch v4 P5plusA5T50X50RS3f05v4
run_eval_batch v4 P5plusA5T50X50RS5f05v4
run_eval_batch v4 P5plusA5T50X50RS10f05v4
run_eval_batch v5 P5plusA5T50X50RS3f05v5
run_eval_batch v5 P5plusA5T50X50RS5f05v5
run_eval_batch v5 P5plusA5T50X50RS10f05v5

log "=== rebuild tables ==="
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== v13 complete ==="
log "manifest=$MANIFEST pipe=$PIPE"
