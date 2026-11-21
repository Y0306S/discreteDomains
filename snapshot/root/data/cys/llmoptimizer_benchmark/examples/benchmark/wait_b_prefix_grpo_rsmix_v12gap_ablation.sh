#!/usr/bin/env bash
# v12gap ablation — same harness as v12, buffer = v12 (12 jobs) + 9 gap jobs.
#
# Phases (like v12):  COLLECT all → TRAIN all → EVAL all
#   Collect: gap jobs on v3 (version-agnostic), merge v12gap for v3/v4/v5
#   Train:   v3/v4/v5 in parallel on GPU 0,1,2 from attr_t2 scratch, ckpt 8/16/32
#   Eval:    9 adapters × RS3/RS5/RS10 (v12 baseline evals not re-run)
#
# Usage:
#   bash examples/benchmark/wait_b_prefix_grpo_rsmix_v12gap_ablation.sh
#   SKIP_GAP_COLLECT=1 SKIP_MERGE=1 bash ...   # train+eval only
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
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12gap_ckpt81632_manifest.json}"

RS_VERSIONS="${RS_VERSIONS:-v3 v4 v5}"
TEMP="${TEMP:-1.0}"
GAP_COLLECT_GPUS="${GAP_COLLECT_GPUS:-0,1,2,3}"
TRAIN_MEMBER_POLICY="${TRAIN_MEMBER_POLICY:-adaptive}"

GAP_FROM="${GAP_FROM:-benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T1p0_pareto32_v13}"
GAP_PATCH_ROOT="${GAP_PATCH_ROOT:-benchmark_data/rl_rollouts/B_rsmix_v3_gap_patch_T1p0_pareto32}"
MIN_RECORDS="${MIN_RECORDS:-2688}"
N_GAP_JOBS="${N_GAP_JOBS:-9}"
N_TOTAL_JOBS="${N_TOTAL_JOBS:-21}"

SKIP_GAP_COLLECT="${SKIP_GAP_COLLECT:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"
CHECKPOINT_AT="${CHECKPOINT_AT:-8,16,32}"
LR="${LR:-1e-6}"
KL_BETA="${KL_BETA:-0.05}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/b_prefix_grpo_rsmix_v12gap_ablation_${TS}.log"
mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

gap_jobs_missing() {
  "$PY_TRAIN" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "examples/benchmark")
from p5plus_rl_collect_schedule import V13_GAP_JOBS
gap = set((int(k), sc) for k, sc in V13_GAP_JOBS)
need = len(gap)
roots = [Path("$GAP_FROM"), Path("$GAP_PATCH_ROOT")]
missing = []
for t in json.loads(Path("$TASKS_JSON").read_text()):
    tt, ds, algo = t["task_type"], t["dataset"], t["algorithm"]
    task = f"{tt}_{ds}_{algo}"
    have = set()
    for root in roots:
        tdir = root / task
        if not tdir.is_dir():
            continue
        for p in tdir.rglob("rollouts_*.jsonl"):
            for ln in p.open():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                key = (int(r["prefix_k"]), r.get("scenario"))
                if key in gap:
                    have.add(key)
    if len(have) < need:
        missing.append(f"{tt}|{ds}|{algo}|{len(have)}")
for m in missing:
    print(m)
PY
}

verify_v12_sources() {
  local ver fail=0
  for ver in $RS_VERSIONS; do
    local v12="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12"
    if [[ ! -d "$v12" ]]; then
      log "ERROR missing v12 buffer for rs=$ver: $v12"
      fail=1
    fi
  done
  (( fail == 0 )) || exit 1
}

verify_v12gap_buffers() {
  "$PY_TRAIN" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "examples/benchmark")
from p5plus_rl_collect_schedule import V13_MIX_JOBS
need_jobs = set((int(k), sc) for k, sc in V13_MIX_JOBS)
need_records = int("$MIN_RECORDS")
versions = "$(echo $RS_VERSIONS)".split()
fail = False
for ver in versions:
    root = Path(f"benchmark_data/rl_rollouts/B_rsmix_{ver}_n4to10_T${TEMP/./p}_pareto32_v12gap")
    if not root.is_dir():
        print(f"ERROR missing v12gap buffer rs={ver}: {root}")
        fail = True
        continue
    total = 0
    bad_tasks = []
    for task in sorted(root.iterdir()):
        if not task.is_dir():
            continue
        have = set()
        rec = 0
        for p in task.rglob("rollouts_*.jsonl"):
            for ln in p.open():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                have.add((int(r["prefix_k"]), r.get("scenario")))
                rec += 1
        total += rec
        if have != need_jobs:
            bad_tasks.append((task.name, len(have), sorted(have - need_jobs)[:3]))
    print(f"rs={ver} records={total} bad_tasks={len(bad_tasks)}")
    if bad_tasks:
        for name, n, extra in bad_tasks[:5]:
            print(f"  BAD {name} jobs={n}")
        fail = True
    if total < need_records:
        print(f"ERROR rs={ver} records={total} < {need_records}")
        fail = True
raise SystemExit(1 if fail else 0)
PY
}

patch_gap_into_buffer() {
  local out="$1"
  [[ -d "$GAP_PATCH_ROOT" ]] || return 0
  "$PY_TRAIN" - <<PY | tee -a "$PIPE"
import json
from pathlib import Path
import sys
sys.path.insert(0, "examples/benchmark")
from merge_rsmix_v12gap_buffer import GAP_JOBS, _read_lines, _job_key
out = Path("$out")
patch = Path("$GAP_PATCH_ROOT")
added = 0
for task in sorted(patch.iterdir()):
    if not task.is_dir():
        continue
    dest = out / task.name / "rollouts_merged.jsonl"
    if not dest.is_file():
        continue
    lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
    existing = {_job_key(json.loads(ln)) for ln in lines}
    for src in sorted(task.rglob("rollouts_*.jsonl")):
        for ln in _read_lines(src, GAP_JOBS):
            key = _job_key(json.loads(ln))
            if key in existing:
                continue
            lines.append(ln)
            existing.add(key)
            added += 1
    dest.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print(f"patched {added} gap records into {out}")
PY
}

# ===================== Phase 1: COLLECT (all versions) =====================
log "========== PHASE 1: COLLECT v3/v4/v5 v12gap buffers =========="
verify_v12_sources

if [[ "$SKIP_GAP_COLLECT" != "1" ]]; then
  mapfile -t MISSING < <(gap_jobs_missing || true)
  if ((${#MISSING[@]} > 0)); then
    log "=== gap collect rs=v3 missing_tasks=${#MISSING[@]} (rs_mix_v13_gap) ==="
    IFS=',' read -ra CGPUS <<<"$GAP_COLLECT_GPUS"
    gi=0
    for spec in "${MISSING[@]}"; do
      IFS='|' read -r tt ds algo have <<<"$spec"
      gpu="${CGPUS[$((gi % ${#CGPUS[@]}))]}"
      gi=$((gi + 1))
      outdir="$GAP_PATCH_ROOT/${tt}_${ds}_${algo}"
      log "gap patch gpu=$gpu $tt/$ds/$algo have=$have/$N_GAP_JOBS -> $outdir"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY_EVAL" examples/benchmark/run_p5_rl_bo_prefix_b.py \
        --task-type "$tt" --dataset "$ds" --algorithm "$algo" \
        --metric "$( [[ "$tt" == cls ]] && echo bal_acc || echo mse )" \
        --collect-schedule rs_mix_v13_gap \
        --train-member-policy "$TRAIN_MEMBER_POLICY" \
        --rs-version v3 --n-samples 4 --n-samples-max 10 \
        --bo-history-root history_data/history_data --seed 1 --time-limit 21600 \
        --output-dir "$outdir" \
        --qwen-backend hf \
        --qwen-model-path /root/eb-public/huggingface-models/Qwen/Qwen3.5-9B \
        --qwen-adapter-path "$INIT" \
        --llm-response-format json_only --llm-max-tokens 768 --llm-temperature 1.0 \
        >>"$PIPE" 2>&1 &
    done
    wait
    mapfile -t STILL < <(gap_jobs_missing || true)
    if ((${#STILL[@]} > 0)); then
      log "ERROR gap collect incomplete: ${STILL[*]}"
      exit 1
    fi
    log "gap collect complete (all 32 tasks × $N_GAP_JOBS jobs)"
  else
    log "gap collect already complete (32/32 tasks)"
  fi
else
  log "SKIP_GAP_COLLECT=1"
fi

if [[ "$SKIP_MERGE" != "1" ]]; then
  for ver in $RS_VERSIONS; do
    V12="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12"
    OUT="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12gap"
    log "=== merge v12gap rs=$ver (v12 rs_warm + shared gap from v3) -> $OUT ==="
    "$PY_TRAIN" examples/benchmark/merge_rsmix_v12gap_buffer.py \
      --v12-from "$V12" --gap-from "$GAP_FROM" --out "$OUT" | tee -a "$PIPE"
    patch_gap_into_buffer "$OUT"
    log "=== retag adaptive rs=$ver ==="
    "$PY_TRAIN" examples/benchmark/retag_rsmix_train_members.py \
      --buffer-root "$OUT" --policy adaptive --seed 1 | tee -a "$PIPE"
  done
else
  log "SKIP_MERGE=1"
fi

log "=== verify v12gap buffers before train ==="
verify_v12gap_buffers
log "PHASE 1 COLLECT done"

# ===================== Phase 2: TRAIN (all versions at once) =====================
log "========== PHASE 2: TRAIN v3/v4/v5 parallel (from attr_t2 scratch) =========="

if [[ "$SKIP_TRAIN" != "1" ]]; then
  export PATH="$LF_ENV/bin:$PATH"
  read -ra TGPU_ARR <<<"$TRAIN_GPUS"
  declare -a TRAIN_PIDS=()
  declare -a TRAIN_VERS=()
  i=0
  for ver in $RS_VERSIONS; do
    BUF="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12gap"
    OUT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_${ver}_pareto32_v12gap"
    gpu="${TGPU_ARR[$((i % ${#TGPU_ARR[@]}))]}"
    i=$((i + 1))
    rm -rf "$OUT"
    log "=== train rs=$ver GPU=$gpu buf=$BUF out=$OUT ckpts=$CHECKPOINT_AT ==="
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
  log "PHASE 2 TRAIN done"
else
  log "SKIP_TRAIN=1"
fi

log "=== write manifest -> $MANIFEST ==="
"$PY_TRAIN" - <<PY | tee -a "$PIPE"
import json
from pathlib import Path
versions = "$(echo $RS_VERSIONS)".split()
ckpts = [int(x) for x in "$CHECKPOINT_AT".split(",") if x.strip()]
prompts_by_ver = {
    "v3": ["P5plusA5T50X50RS3f05", "P5plusA5T50X50RS5f05", "P5plusA5T50X50RS10f03"],
    "v4": ["P5plusA5T50X50RS3f05v4", "P5plusA5T50X50RS5f05v4", "P5plusA5T50X50RS10f05v4"],
    "v5": ["P5plusA5T50X50RS3f05v5", "P5plusA5T50X50RS5f05v5", "P5plusA5T50X50RS10f05v5"],
}
rows = []
for ver in versions:
    root = Path(f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12gap")
    for k in ckpts:
        mid = f"s1b_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12gap_ckpt{k}"
        rows.append({
            "id": mid,
            "adapter": str(root / f"ckpt{k}"),
            "resp": "json_only",
            "rs_version": ver,
            "ckpt": k,
            "recipe": "b_prefix_grpo_v12gap",
            "init": "s1b_ep7p5_attr_t2",
            "train_tasks": "picked_32_pareto",
            "eval_tasks": "tasks_40",
            "eval_prompts": prompts_by_ver[ver],
            "schedule": "v12_plus_gap9",
        })
Path("$MANIFEST").write_text(json.dumps(rows, indent=2) + "\n")
missing = [r for r in rows if not (Path(r["adapter"]) / "adapter_model.safetensors").exists()]
print(f"wrote {len(rows)} -> $MANIFEST")
if missing:
    for r in missing:
        print("MISSING", r["id"])
    raise SystemExit(1)
PY

# ===================== Phase 3: EVAL (all at once) =====================
if [[ "$SKIP_EVAL" == "1" ]]; then
  log "SKIP_EVAL=1 pipe=$PIPE"
  exit 0
fi

log "========== PHASE 3: EVAL all v12gap adapters =========="
OUTPUT_BASE="benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12gap"
FULL_MANIFEST="$MANIFEST"

run_eval_batch() {
  local ver="$1" prompt="$2"
  local mf="benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12gap_${prompt}_manifest_${ver}.json"
  "$PY_TRAIN" - <<PY
import json
from pathlib import Path
rows = [r for r in json.loads(Path("$FULL_MANIFEST").read_text()) if r.get("rs_version") == "$ver"]
Path("$mf").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> $mf")
PY
  log "=== eval v12gap $ver prompt=$prompt ==="
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

log "=== v12gap ablation complete ==="
log "manifest=$MANIFEST pipe=$PIPE"
