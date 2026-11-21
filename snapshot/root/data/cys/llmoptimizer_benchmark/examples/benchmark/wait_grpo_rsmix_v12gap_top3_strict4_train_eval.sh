#!/usr/bin/env bash
# Offline 21-point GRPO for top-3 v12 adapters (v4 ckpt32, v4 ckpt16, v3 ckpt8).
#
# Phases:  COLLECT gap (strict_first4, incremental) → MERGE v12+gap → TRAIN ×3 → EVAL
#
# Collect is incremental:
#   - strict4 patch: only tasks missing 9 gap jobs in GAP_PATCH_ROOT
#   - optional any-gap mode: only tasks missing gap across strict4 + v13 + v12gap fallbacks
#
# Top-3 only (not v5). Merge builds v3 + v4 v12gap_strict4 buffers (shared gap, version rs_warm).
#
# Usage:
#   PREFLIGHT_ONLY=1 bash examples/benchmark/wait_grpo_rsmix_v12gap_top3_strict4_train_eval.sh
#   bash examples/benchmark/wait_grpo_rsmix_v12gap_top3_strict4_train_eval.sh
#   SKIP_GAP_COLLECT=1 SKIP_MERGE=1 bash ...   # train+eval only (buffers must exist)
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
cd "$REPO"

LOG_ROOT="${LOG_ROOT:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
LF_ENV="${LF_ENV:-/root/miniconda3/envs/llamafactory_qwen35}"
EVAL_ENV="${EVAL_ENV:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval}"
PY_TRAIN="${PY_TRAIN:-$LF_ENV/bin/python}"
PY_EVAL="${PY_EVAL:-$EVAL_ENV/bin/python}"
[[ -x "$PY_TRAIN" ]] || PY_TRAIN="/root/data/cys/blockdata_archive/miniconda3/envs/llamafactory_qwen35/bin/python"
[[ -x "$PY_EVAL" ]] || PY_EVAL="/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python"

INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
ORDER="${ORDER:-benchmark_data/rl_task_selection/picked_32_pareto_train_order.json}"
TASKS_JSON="${TASKS_JSON:-benchmark_data/rl_task_selection/picked_32_pareto_tasks.json}"
MANIFEST="${MANIFEST:-benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12gap_top3_strict4_manifest.json}"
EVAL_TASKS_JSON="${EVAL_TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_30.json}"

TEMP="${TEMP:-1.0}"
RS_VERSIONS_MERGE="${RS_VERSIONS_MERGE:-v3 v4}"
GAP_PATCH_ROOT="${GAP_PATCH_ROOT:-benchmark_data/rl_rollouts/B_rsmix_v3_gap_patch_T1p0_pareto32_strict4}"
GAP_FALLBACK_V13="${GAP_FALLBACK_V13:-benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T1p0_pareto32_v13}"
V12GAP_SUFFIX="${V12GAP_SUFFIX:-v12gap_strict4}"

# adaptive_4 in docs = strict_first4 in code (first 4 samples only; no adaptive 3+1 inject)
TRAIN_MEMBER_POLICY="${TRAIN_MEMBER_POLICY:-strict_first4}"
RETAG_POLICY="${RETAG_POLICY:-strict_first4}"
TRAIN_STRICT_FIRST_N="${TRAIN_STRICT_FIRST_N:-4}"
COLLECT_MODE="${COLLECT_MODE:-strict4_only}"   # strict4_only | any_missing (fallback gap is adaptive, not strict4)
GAP_COLLECT_GPUS="${GAP_COLLECT_GPUS:-0,1,2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-0 1 2}"

LR="${LR:-1e-6}"
KL_BETA="${KL_BETA:-0.05}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"

N_GAP_JOBS="${N_GAP_JOBS:-9}"
SKIP_GAP_COLLECT="${SKIP_GAP_COLLECT:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
PIPE="$LOG_ROOT/grpo_rsmix_v12gap_top3_strict4_${TS}.log"
mkdir -p "$LOG_ROOT"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

# ver|ckpt|init_adapter|rs_version|buffer_rs|best_eval_prompt|out_suffix
declare -a TOP3=(
  "v4|32|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32|v4|v4|P5plusA5T50X50RS5f05v4|v12gap_strict4_top3"
  "v4|16|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt16|v4|v4|P5plusA5T50X50RS3f05v4|v12gap_strict4_top3"
  "v3|8|/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v3_pareto32_v12/ckpt8|v3|v3|P5plusA5T50X50RS5f05|v12gap_strict4_top3"
)

OUTPUT_BASE="${OUTPUT_BASE:-benchmark_data/real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12gap_top3_strict4}"

log "=== v12gap top-3 strict4 offline pipeline ==="
log "COLLECT_MODE=$COLLECT_MODE collect=$TRAIN_MEMBER_POLICY retag=$RETAG_POLICY train_strict_first_n=$TRAIN_STRICT_FIRST_N MAX_SEQ_LEN=$MAX_SEQ_LEN"
log "GAP_PATCH_ROOT=$GAP_PATCH_ROOT V12GAP_SUFFIX=$V12GAP_SUFFIX"
log "SKIP_GAP_COLLECT=$SKIP_GAP_COLLECT SKIP_MERGE=$SKIP_MERGE SKIP_TRAIN=$SKIP_TRAIN SKIP_EVAL=$SKIP_EVAL"
log "pipe=$PIPE"

preflight() {
  "$PY_TRAIN" examples/benchmark/rsmix_v12gap_gap_status.py \
    --strict4-patch "$GAP_PATCH_ROOT" --order "$ORDER" | tee -a "$PIPE"
}

list_collect_tasks() {
  local flag="--tasks-needing-strict4-collect"
  if [[ "$COLLECT_MODE" == "any_missing" ]]; then
    flag="--tasks-needing-any-gap-collect"
  fi
  "$PY_TRAIN" examples/benchmark/rsmix_v12gap_gap_status.py \
    --strict4-patch "$GAP_PATCH_ROOT" --order "$ORDER" "$flag"
}

verify_v12_sources() {
  local ver fail=0
  for ver in $RS_VERSIONS_MERGE; do
    local v12="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12"
    if [[ ! -d "$v12" ]]; then
      log "ERROR missing v12 buffer rs=$ver: $v12"
      fail=1
    fi
  done
  (( fail == 0 )) || exit 1
}

verify_merged_buffers() {
  local ver fail=0
  for ver in $RS_VERSIONS_MERGE; do
    local buf="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_${V12GAP_SUFFIX}"
    if [[ ! -d "$buf" ]]; then
      log "ERROR missing merged buffer rs=$ver: $buf"
      fail=1
      continue
    fi
    "$PY_TRAIN" - <<PY | tee -a "$PIPE"
import json, sys
from pathlib import Path
sys.path.insert(0, "examples/benchmark")
from p5plus_rl_collect_schedule import V13_MIX_JOBS
need = set((int(k), sc) for k, sc in V13_MIX_JOBS)
root = Path("$buf")
bad = []
for task in sorted(root.iterdir()):
    if not task.is_dir():
        continue
    have = set()
    merged = task / "rollouts_merged.jsonl"
    paths = [merged] if merged.is_file() else sorted(task.rglob("rollouts_*.jsonl"))
    for p in paths:
        for ln in p.read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            have.add((int(r["prefix_k"]), r.get("scenario")))
    if have != need:
        bad.append((task.name, len(have)))
if bad:
    print(f"ERROR rs=$ver incomplete tasks={len(bad)} sample={bad[:3]}")
    raise SystemExit(1)
print(f"OK rs=$ver buffer=$buf tasks=32 jobs=21")
PY
  done
  (( fail == 0 )) || exit 1
}

patch_gap_sources_into_buffer() {
  local out="$1"
  "$PY_TRAIN" - <<PY | tee -a "$PIPE"
import json
from pathlib import Path
import sys
sys.path.insert(0, "examples/benchmark")
from merge_rsmix_v12gap_buffer import GAP_JOBS, _read_lines, _job_key

out = Path("$out")
sources = [
    Path("$GAP_PATCH_ROOT"),
    Path("$GAP_FALLBACK_V13"),
    Path("benchmark_data/rl_rollouts/B_rsmix_v3_n4to10_T1p0_pareto32_v12gap"),
]
added = 0
for task in sorted(out.iterdir()):
    if not task.is_dir():
        continue
    dest = task / "rollouts_merged.jsonl"
    if not dest.is_file():
        continue
    lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
    existing = {_job_key(json.loads(ln)) for ln in lines}
    for src_root in sources:
        src_task = src_root / task.name
        if not src_task.is_dir():
            continue
        for src in sorted(src_task.rglob("rollouts_*.jsonl")):
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

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  preflight
  log "collect tasks ($COLLECT_MODE):"
  list_collect_tasks | tee -a "$PIPE" || true
  log "PREFLIGHT_ONLY=1 — not launching collect/train/eval"
  exit 0
fi

# ===================== Phase 1: COLLECT (incremental gap) =====================
log "========== PHASE 1: COLLECT gap (strict_first4, incremental) =========="
verify_v12_sources
preflight

if [[ "$SKIP_GAP_COLLECT" != "1" ]]; then
  mapfile -t COLLECT_TASKS < <(list_collect_tasks || true)
  if ((${#COLLECT_TASKS[@]} > 0)); then
    log "=== gap collect mode=$COLLECT_MODE tasks=${#COLLECT_TASKS[@]} policy=$TRAIN_MEMBER_POLICY (throttled) ==="
    GAP_COLLECT_GPUS="$GAP_COLLECT_GPUS" \
      GAP_JOBS_PER_GPU="${GAP_JOBS_PER_GPU:-1}" \
      GAP_PATCH_ROOT="$GAP_PATCH_ROOT" \
      TRAIN_MEMBER_POLICY="$TRAIN_MEMBER_POLICY" \
      COLLECT_MODE="$COLLECT_MODE" \
      PIPE="$PIPE" \
      bash examples/benchmark/run_rsmix_v12gap_collect_throttled.sh >>"$PIPE" 2>&1
    log "gap collect complete"
  else
    log "gap collect skipped — all tasks already have gap jobs in patch (mode=$COLLECT_MODE)"
  fi
else
  log "SKIP_GAP_COLLECT=1"
fi

if [[ "$SKIP_MERGE" != "1" ]]; then
  for ver in $RS_VERSIONS_MERGE; do
    V12="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_v12"
    OUT="benchmark_data/rl_rollouts/B_rsmix_${ver}_n4to10_T${TEMP/./p}_pareto32_${V12GAP_SUFFIX}"
    log "=== merge $V12GAP_SUFFIX rs=$ver (v12 + gap fallbacks) -> $OUT ==="
    "$PY_TRAIN" examples/benchmark/merge_rsmix_v12gap_buffer.py \
      --v12-from "$V12" --gap-from "$GAP_FALLBACK_V13" --out "$OUT" | tee -a "$PIPE"
    patch_gap_sources_into_buffer "$OUT"
    log "=== retag $RETAG_POLICY rs=$ver ==="
    "$PY_TRAIN" examples/benchmark/retag_rsmix_train_members.py \
      --buffer-root "$OUT" --policy "$RETAG_POLICY" --seed 1 | tee -a "$PIPE"
  done
else
  log "SKIP_MERGE=1"
fi

verify_merged_buffers
log "PHASE 1 COLLECT+MERGE done"

# ===================== Phase 2: TRAIN (top-3 parallel) =====================
log "========== PHASE 2: TRAIN top-3 offline GRPO (21 jobs) =========="

if [[ "$SKIP_TRAIN" != "1" ]]; then
  export PATH="$LF_ENV/bin:$PATH"
  read -ra TGPU_ARR <<<"$TRAIN_GPUS"
  declare -a PIDS=()
  declare -a LABELS=()
  i=0
  for spec in "${TOP3[@]}"; do
    IFS='|' read -r ver ckpt init rs buf_rs prompt suffix <<<"$spec"
    BUF="benchmark_data/rl_rollouts/B_rsmix_${buf_rs}_n4to10_T${TEMP/./p}_pareto32_${V12GAP_SUFFIX}"
    OUT="/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_${ver}_pareto32_v12_ckpt${ckpt}_${suffix}"
    gpu="${TGPU_ARR[$((i % ${#TGPU_ARR[@]}))]}"
    i=$((i + 1))
    if [[ ! -f "$init/adapter_model.safetensors" ]]; then
      log "ERROR missing init $init"
      exit 1
    fi
    rm -rf "$OUT"
    log "=== train $ver ckpt$ckpt GPU=$gpu init=$init buf=$BUF out=$OUT ==="
    (
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY_TRAIN" examples/benchmark/train_p5_reinforce.py \
        --buffer-root "$BUF" \
        --strict-first-n "$TRAIN_STRICT_FIRST_N" \
        --init-adapter "$init" \
        --output-dir "$OUT" \
        --dataset-order-file "$ORDER" \
        --strict-dataset-order \
        --advantage-mode grpo_group \
        --dataset-sequential \
        --max-seq-len "$MAX_SEQ_LEN" \
        --lr "$LR" --kl-beta "$KL_BETA" --grad-accum "$GRAD_ACCUM" --seed 1
    ) >>"$PIPE" 2>&1 &
    PIDS+=("$!")
    LABELS+=("${ver}_ckpt${ckpt}_${suffix}")
  done
  fail=0
  for j in "${!PIDS[@]}"; do
    if wait "${PIDS[$j]}"; then
      log "train OK ${LABELS[$j]}"
    else
      log "ERROR train FAILED ${LABELS[$j]}"
      fail=1
    fi
  done
  (( fail == 0 )) || exit 1
else
  log "SKIP_TRAIN=1"
fi

log "=== write manifest -> $MANIFEST ==="
"$PY_TRAIN" - <<'PY' | tee -a "$PIPE"
import json
from pathlib import Path

suffix = "v12gap_strict4_top3"
specs = [
    ("v4", 32, "P5plusA5T50X50RS5f05v4"),
    ("v4", 16, "P5plusA5T50X50RS3f05v4"),
    ("v3", 8, "P5plusA5T50X50RS5f05"),
]
rows = []
for ver, ckpt, prompt in specs:
    mid = f"s1b_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12_ckpt{ckpt}_{suffix}"
    adapter = Path(
        f"/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_{ver}_pareto32_v12_ckpt{ckpt}_{suffix}"
    )
    if not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing adapter {adapter}")
    rows.append({
        "id": mid,
        "adapter": str(adapter),
        "resp": "json_only",
        "rs_version": ver,
        "ckpt": ckpt,
        "recipe": suffix,
        "init": f"grpo_rsmix_{ver}_ckpt{ckpt}",
        "tasks": "picked_32_pareto",
        "schedule": "rs_mix_v12gap_strict4_top3",
        "train_member_policy_collect": "strict_first4",
        "train_member_policy_retag": "strict_first4",
        "train_member_policy_train": "strict_first_n_4",
        "n_jobs_per_task": 21,
        "best_prompt": prompt,
    })
mf = Path("benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12gap_top3_strict4_manifest.json")
mf.write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> {mf}")
PY

if [[ "$SKIP_EVAL" == "1" ]]; then
  log "SKIP_EVAL=1 manifest=$MANIFEST pipe=$PIPE"
  exit 0
fi

# ===================== Phase 3: EVAL =====================
# Eval on physical GPUs 0..N-1 (N=4 → GPUs 0,1,2,3; 2 workers/GPU).
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:-4}"
EVAL_JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}"
export EVAL_NUM_GPUS EVAL_JOBS_PER_GPU
log "========== PHASE 3: EVAL top-3 (best RS prompt each) =========="
log "EVAL physical GPUs 0-$((EVAL_NUM_GPUS - 1)) NUM_GPUS=$EVAL_NUM_GPUS JOBS_PER_GPU=$EVAL_JOBS_PER_GPU"

run_model_eval() {
  local model_id="$1" prompt="$2"
  local mf="benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12gap_top3_strict4_${model_id}_${prompt}.json"
  "$PY_TRAIN" - <<PY
import json
from pathlib import Path
rows = [r for r in json.loads(Path("$MANIFEST").read_text()) if r.get("id") == "$model_id"]
Path("$mf").write_text(json.dumps(rows, indent=2) + "\n")
print(f"wrote {len(rows)} -> $mf")
if len(rows) != 1:
    raise SystemExit(1)
PY
  log "=== eval $model_id prompt=$prompt ==="
  TASKS_JSON="$EVAL_TASKS_JSON" \
    OUTPUT_ROOT="$OUTPUT_BASE" NUM_GPUS="${EVAL_NUM_GPUS:-4}" JOBS_PER_GPU="${EVAL_JOBS_PER_GPU:-2}" \
    PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}" \
    STAGE=from_manifest MANIFEST_JSON="$mf" \
    HYBRID=HPUCB_arm PROMPT_VARIANT="$prompt" LLM_OPTIMIZERS=ptllm_tpe_qwen \
    SEED=1 EVAL_RESUME=1 TIME_LIMIT=10800 \
    PYTHON_LLM="$PY_EVAL" PYTHON_CLASS="$PY_EVAL" \
    bash examples/benchmark/run_mltest20_algo40_eval.sh >>"$PIPE" 2>&1
}

while IFS= read -r line; do
  id=$(python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['id'])" <<<"$line")
  prompt=$(python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['best_prompt'])" <<<"$line")
  run_model_eval "$id" "$prompt"
done < <("$PY_TRAIN" - <<'PY'
import json
from pathlib import Path
for r in json.loads(Path("benchmark_data/rl_task_selection/b_prefix_grpo_rsmix_v12gap_top3_strict4_manifest.json").read_text()):
    print(json.dumps(r))
PY
)

log "=== rebuild tables ==="
"$PY_EVAL" examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only >>"$PIPE" 2>&1 || true

log "=== v12gap top-3 strict4 DONE ==="
log "manifest=$MANIFEST output_base=$OUTPUT_BASE pipe=$PIPE"
