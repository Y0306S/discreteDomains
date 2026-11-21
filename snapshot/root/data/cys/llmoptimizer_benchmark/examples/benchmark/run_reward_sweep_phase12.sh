#!/usr/bin/env bash
# Phase 1: reward sweep on Setup B light16 (dsseq), ckpts at 12/14/16, then algo40 eval.
# Phase 2 (after pick): continue best arm from ckpt16 → next8, ckpts 18/20/22/24, eval.
#
# Locked defaults:
#   arms: hybrid, dbest, abs, mix_strong
#   init: s1a_ep7p5
#   selection: highest mean_balanced_norm on seed1 no_smac pooled table
#   Phase2 always resumes from winner's ckpt16
#
# Usage:
#   PHASE=1 bash examples/benchmark/run_reward_sweep_phase12.sh
#   PHASE=2 BEST_ARM=hybrid bash examples/benchmark/run_reward_sweep_phase12.sh
#   PHASE=all bash ...   # phase1 then auto-pick BEST_ARM then phase2
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

PHASE="${PHASE:-1}"
ARMS="${ARMS:-hybrid dbest abs mix_strong}"
CKPTS_P1="${CKPTS_P1:-12,14,16}"
CKPTS_P2="${CKPTS_P2:-18,20,22,24}"
INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
SRC_B_LIGHT="${SRC_B_LIGHT:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_light16}"
SRC_B_MED="${SRC_B_MED:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_medium32}"
ORDER_L16="${ORDER_L16:-benchmark_data/rl_task_selection/light_16.json}"
ORDER_NEXT8="${ORDER_NEXT8:-benchmark_data/rl_task_selection/light16_then_next8_next8_only.json}"
ARMS_FILE="${ARMS_FILE:-benchmark_data/rl_task_selection/reward_sweep_arms.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
BUFFER_ROOT="${BUFFER_ROOT:-benchmark_data/rl_rollouts}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/reward_sweep_phase12_${TS}.log"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
BEST_ARM="${BEST_ARM:-}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

reweight_arm() {
  local arm="$1" src="$2" order="$3" dst_name="$4"
  local dst="$BUFFER_ROOT/$dst_name"
  log "reweight arm=$arm -> $dst"
  python3 examples/benchmark/reweight_rl_buffer.py \
    --src-root "$src" \
    --dst-root "$dst" \
    --arm "$arm" \
    --arms-file "$ARMS_FILE" \
    --task-order-file "$order" \
    --overwrite
}

train_arm_phase1() {
  local arm="$1"
  local gpu="${2:-0}"
  local buf="$BUFFER_ROOT/B_rw_${arm}_light16"
  local ckpt_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_${arm}"
  local final="$ckpt_root/final"
  local arm_log="$LOGDIR/reward_sweep_train_${arm}_${TS}.log"
  mkdir -p "$ckpt_root"
  log "PHASE1 train arm=$arm gpu=$gpu ckpts=$CKPTS_P1 log=$arm_log"
  # One physical GPU per arm via CUDA_VISIBLE_DEVICES; train always uses cuda:0 inside.
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$buf" \
    --init-adapter "$INIT" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER_L16" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at "$CKPTS_P1" \
    --checkpoint-root "$ckpt_root" \
    --device cuda:0 \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    >"$arm_log" 2>&1
  local rc=$?
  log "PHASE1 train arm=$arm finished rc=$rc"
  return "$rc"
}

write_manifest_phase1() {
  local out="$1"
  python3 - <<PY
import json
from pathlib import Path
arms = "$ARMS".split()
ckpts = [int(x) for x in "$CKPTS_P1".split(",") if x.strip()]
root = Path("$WEIGHT_ROOT")
rows = []
for arm in arms:
  for c in ckpts:
    mid = f"s1a_b_rw_{arm}_ckpt{c}"
    adapter = root / f"llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_{arm}" / f"ckpt{c}"
    if not (adapter / "adapter_model.safetensors").exists():
      raise SystemExit(f"missing adapter {adapter}")
    rows.append({"id": mid, "adapter": str(adapter), "resp": "json_only"})
Path("$out").write_text(json.dumps(rows, indent=2))
print(f"wrote {len(rows)} models -> $out")
PY
}

eval_manifest() {
  local mf="$1"
  local tag="$2"
  log "eval $tag manifest=$mf"
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$mf" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  log "regen tables after $tag"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
}

pick_best_arm() {
  python3 - <<'PY'
import csv
from pathlib import Path
p = Path("benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank/mean_norm_table_pooled_drop20pct_seed1.csv")
rows = list(csv.DictReader(p.open()))
best = None
for r in rows:
    v = r.get("variant", "")
    if "s1a_b_rw_" not in v or "_ckpt" not in v:
        continue
    # variant like ptllm_tpe_qwen_s1a_b_rw_hybrid_ckpt12__P5__H0
    mid = v.replace("ptllm_tpe_qwen_", "").replace("__P5__H0", "")
    # s1a_b_rw_hybrid_ckpt12
    try:
        arm = mid.split("s1a_b_rw_", 1)[1].rsplit("_ckpt", 1)[0]
        ckpt = int(mid.rsplit("_ckpt", 1)[1])
    except Exception:
        continue
    if ckpt not in (12, 14, 16):
        continue
    bal = float(r["mean_balanced_norm"])
    if best is None or bal > best[0]:
        best = (bal, arm, ckpt, mid, r.get("display_name"))
if best is None:
    raise SystemExit("no s1a_b_rw_* ckpt{12,14,16} rows in table")
print(f"{best[1]}\t{best[2]}\t{best[0]:.6f}\t{best[3]}")
PY
}

train_arm_phase2() {
  local arm="$1"
  local buf="$BUFFER_ROOT/B_rw_${arm}_next8"
  local p1_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_${arm}"
  local init="$p1_root/ckpt16"
  local ckpt_root="$p1_root"  # write ckpt18..24 alongside
  local final="$p1_root/final_p2"
  [[ -f "$init/adapter_model.safetensors" ]] || { log "ERROR missing $init"; exit 1; }
  log "PHASE2 continue arm=$arm from ckpt16 -> $CKPTS_P2"
  "$PYTHON_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$buf" \
    --init-adapter "$init" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER_NEXT8" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at "$CKPTS_P2" \
    --checkpoint-root "$ckpt_root" \
    --checkpoint-label-offset 16 \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    2>&1 | tee -a "$PIPE"
}

write_manifest_phase2() {
  local arm="$1"
  local out="$2"
  python3 - <<PY
import json
from pathlib import Path
arm = "$arm"
ckpts = [int(x) for x in "$CKPTS_P2".split(",") if x.strip()]
root = Path("$WEIGHT_ROOT") / f"llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_{arm}"
rows = []
for c in ckpts:
    mid = f"s1a_b_rw_{arm}_ckpt{c}"
    adapter = root / f"ckpt{c}"
    if not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing adapter {adapter}")
    rows.append({"id": mid, "adapter": str(adapter), "resp": "json_only"})
Path("$out").write_text(json.dumps(rows, indent=2))
print(f"wrote {len(rows)} models -> $out")
PY
}

phase1() {
  log "=== PHASE 1 start arms=[$ARMS] (parallel across GPUs) ==="
  for arm in $ARMS; do
    reweight_arm "$arm" "$SRC_B_LIGHT" "$ORDER_L16" "B_rw_${arm}_light16"
  done
  local -a pids=()
  local -a arm_list=()
  local gpu=0
  local -a arm_arr=($ARMS)
  if (( ${#arm_arr[@]} > NUM_GPUS )); then
    log "ERROR: ${#arm_arr[@]} arms > NUM_GPUS=$NUM_GPUS"
    exit 1
  fi
  for arm in "${arm_arr[@]}"; do
    arm_list+=("$arm")
    train_arm_phase1 "$arm" "$gpu" &
    pids+=("$!")
    log "launched arm=$arm pid=${pids[-1]} gpu=$gpu"
    gpu=$((gpu + 1))
  done
  local fail=0
  local i=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      log "ERROR train failed arm=${arm_list[$i]} pid=$pid"
      fail=1
    else
      log "OK train arm=${arm_list[$i]} pid=$pid"
    fi
    i=$((i + 1))
  done
  if (( fail != 0 )); then
    log "ERROR phase1 train had failures; aborting eval"
    exit 1
  fi
  local mf="$LOGDIR/reward_sweep_phase1_manifest_${TS}.json"
  write_manifest_phase1 "$mf"
  eval_manifest "$mf" "phase1"
  log "=== PHASE 1 done ==="
}

phase2() {
  local arm="${BEST_ARM:-}"
  if [[ -z "$arm" ]]; then
    log "auto-picking BEST_ARM from seed1 table..."
    local pick
    pick="$(pick_best_arm)"
    arm="$(echo "$pick" | cut -f1)"
    log "picked: $pick"
    BEST_ARM="$arm"
  fi
  log "=== PHASE 2 start BEST_ARM=$arm ==="
  reweight_arm "$arm" "$SRC_B_MED" "$ORDER_NEXT8" "B_rw_${arm}_next8"
  train_arm_phase2 "$arm"
  local mf="$LOGDIR/reward_sweep_phase2_manifest_${TS}.json"
  write_manifest_phase2 "$arm" "$mf"
  eval_manifest "$mf" "phase2"
  log "=== PHASE 2 done BEST_ARM=$arm ==="
  log "Final pick among ckpt{12,14,16,18,20,22,24} for arm=$arm by mean_balanced_norm in seed1 table."
}

case "$PHASE" in
  1) phase1 ;;
  2) phase2 ;;
  all) phase1; phase2 ;;
  *) echo "PHASE must be 1|2|all"; exit 1 ;;
esac

log "ALL COMPLETE phase=$PHASE pipe=$PIPE"
#!/usr/bin/env bash
# Phase 1: reward sweep on Setup B light16 (dsseq), ckpts at 12/14/16, then algo40 eval.
# Phase 2 (after pick): continue best arm from ckpt16 → next8, ckpts 18/20/22/24, eval.
#
# Locked defaults:
#   arms: hybrid, dbest, abs, mix_strong
#   init: s1a_ep7p5
#   selection: highest mean_balanced_norm on seed1 no_smac pooled table
#   Phase2 always resumes from winner's ckpt16
#
# Usage:
#   PHASE=1 bash examples/benchmark/run_reward_sweep_phase12.sh
#   PHASE=2 BEST_ARM=hybrid bash examples/benchmark/run_reward_sweep_phase12.sh
#   PHASE=all bash ...   # phase1 then auto-pick BEST_ARM then phase2
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"

PHASE="${PHASE:-1}"
ARMS="${ARMS:-hybrid dbest abs mix_strong}"
CKPTS_P1="${CKPTS_P1:-12,14,16}"
CKPTS_P2="${CKPTS_P2:-18,20,22,24}"
INIT="${INIT:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5}"
SRC_B_LIGHT="${SRC_B_LIGHT:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_light16}"
SRC_B_MED="${SRC_B_MED:-benchmark_data/rl_rollouts/B_prefix10to100_n4_T1p0_medium32}"
ORDER_L16="${ORDER_L16:-benchmark_data/rl_task_selection/light_16.json}"
ORDER_NEXT8="${ORDER_NEXT8:-benchmark_data/rl_task_selection/light16_then_next8_next8_only.json}"
ARMS_FILE="${ARMS_FILE:-benchmark_data/rl_task_selection/reward_sweep_arms.json}"
WEIGHT_ROOT="${WEIGHT_ROOT:-/root/data/cys/blockdata_archive/model-weights}"
BUFFER_ROOT="${BUFFER_ROOT:-benchmark_data/rl_rollouts}"
LOGDIR="${LOGDIR:-/root/data/cys/blockdata_archive/LLMOptimizer/logs/bolt_qwen9b_finetune}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
PIPE="$LOGDIR/reward_sweep_phase12_${TS}.log"
PYTHON_TRAIN="${PYTHON_TRAIN:-/root/miniconda3/envs/llamafactory_qwen35/bin/python}"
BEST_ARM="${BEST_ARM:-}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-15000}"
EVAL_RESUME="${EVAL_RESUME:-1}"
SEED="${SEED:-1}"

mkdir -p "$LOGDIR"
log() { echo "[$(date -Is)] $*" | tee -a "$PIPE"; }

reweight_arm() {
  local arm="$1" src="$2" order="$3" dst_name="$4"
  local dst="$BUFFER_ROOT/$dst_name"
  log "reweight arm=$arm -> $dst"
  python3 examples/benchmark/reweight_rl_buffer.py \
    --src-root "$src" \
    --dst-root "$dst" \
    --arm "$arm" \
    --arms-file "$ARMS_FILE" \
    --task-order-file "$order" \
    --overwrite
}

train_arm_phase1() {
  local arm="$1"
  local gpu="${2:-0}"
  local buf="$BUFFER_ROOT/B_rw_${arm}_light16"
  local ckpt_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_${arm}"
  local final="$ckpt_root/final"
  local arm_log="$LOGDIR/reward_sweep_train_${arm}_${TS}.log"
  mkdir -p "$ckpt_root"
  log "PHASE1 train arm=$arm gpu=$gpu ckpts=$CKPTS_P1 log=$arm_log"
  # One physical GPU per arm via CUDA_VISIBLE_DEVICES; train always uses cuda:0 inside.
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$buf" \
    --init-adapter "$INIT" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER_L16" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at "$CKPTS_P1" \
    --checkpoint-root "$ckpt_root" \
    --device cuda:0 \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    >"$arm_log" 2>&1
  local rc=$?
  log "PHASE1 train arm=$arm finished rc=$rc"
  return "$rc"
}

write_manifest_phase1() {
  local out="$1"
  python3 - <<PY
import json
from pathlib import Path
arms = "$ARMS".split()
ckpts = [int(x) for x in "$CKPTS_P1".split(",") if x.strip()]
root = Path("$WEIGHT_ROOT")
rows = []
for arm in arms:
  for c in ckpts:
    mid = f"s1a_b_rw_{arm}_ckpt{c}"
    adapter = root / f"llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_{arm}" / f"ckpt{c}"
    if not (adapter / "adapter_model.safetensors").exists():
      raise SystemExit(f"missing adapter {adapter}")
    rows.append({"id": mid, "adapter": str(adapter), "resp": "json_only"})
Path("$out").write_text(json.dumps(rows, indent=2))
print(f"wrote {len(rows)} models -> $out")
PY
}

eval_manifest() {
  local mf="$1"
  local tag="$2"
  log "eval $tag manifest=$mf"
  NUM_GPUS="$NUM_GPUS" JOBS_PER_GPU="$JOBS_PER_GPU" PACK_MIN_FREE_MB="$PACK_MIN_FREE_MB" \
    STAGE=from_manifest MANIFEST_JSON="$mf" HYBRID=H0 PROMPT_VARIANT=P5 \
    LLM_OPTIMIZERS=ptllm_tpe_qwen SEED="$SEED" EVAL_RESUME="$EVAL_RESUME" TIME_LIMIT=10800 \
    bash examples/benchmark/run_mltest20_algo40_eval.sh 2>&1 | tee -a "$PIPE"
  log "regen tables after $tag"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm 2>&1 | tee -a "$PIPE"
  python3 examples/benchmark/plot_mltest20_algo40_avg_rank.py --rgs-pooled-trial-norm --seed1-only 2>&1 | tee -a "$PIPE"
}

pick_best_arm() {
  python3 - <<'PY'
import csv
from pathlib import Path
p = Path("benchmark_data/real_hpo_v2_mltest20_algo40/analysis/avg_rank/mean_norm_table_pooled_drop20pct_seed1.csv")
rows = list(csv.DictReader(p.open()))
best = None
for r in rows:
    v = r.get("variant", "")
    if "s1a_b_rw_" not in v or "_ckpt" not in v:
        continue
    # variant like ptllm_tpe_qwen_s1a_b_rw_hybrid_ckpt12__P5__H0
    mid = v.replace("ptllm_tpe_qwen_", "").replace("__P5__H0", "")
    # s1a_b_rw_hybrid_ckpt12
    try:
        arm = mid.split("s1a_b_rw_", 1)[1].rsplit("_ckpt", 1)[0]
        ckpt = int(mid.rsplit("_ckpt", 1)[1])
    except Exception:
        continue
    if ckpt not in (12, 14, 16):
        continue
    bal = float(r["mean_balanced_norm"])
    if best is None or bal > best[0]:
        best = (bal, arm, ckpt, mid, r.get("display_name"))
if best is None:
    raise SystemExit("no s1a_b_rw_* ckpt{12,14,16} rows in table")
print(f"{best[1]}\t{best[2]}\t{best[0]:.6f}\t{best[3]}")
PY
}

train_arm_phase2() {
  local arm="$1"
  local buf="$BUFFER_ROOT/B_rw_${arm}_next8"
  local p1_root="$WEIGHT_ROOT/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_${arm}"
  local init="$p1_root/ckpt16"
  local ckpt_root="$p1_root"  # write ckpt18..24 alongside
  local final="$p1_root/final_p2"
  [[ -f "$init/adapter_model.safetensors" ]] || { log "ERROR missing $init"; exit 1; }
  log "PHASE2 continue arm=$arm from ckpt16 -> $CKPTS_P2"
  "$PYTHON_TRAIN" examples/benchmark/train_p5_reinforce.py \
    --buffer-root "$buf" \
    --init-adapter "$init" \
    --output-dir "$final" \
    --dataset-order-file "$ORDER_NEXT8" \
    --strict-dataset-order \
    --advantage-mode per_dataset \
    --dataset-sequential \
    --checkpoint-at "$CKPTS_P2" \
    --checkpoint-root "$ckpt_root" \
    --checkpoint-label-offset 16 \
    --lr 1e-6 --kl-beta 0.05 --grad-accum 8 --seed 1 \
    2>&1 | tee -a "$PIPE"
}

write_manifest_phase2() {
  local arm="$1"
  local out="$2"
  python3 - <<PY
import json
from pathlib import Path
arm = "$arm"
ckpts = [int(x) for x in "$CKPTS_P2".split(",") if x.strip()]
root = Path("$WEIGHT_ROOT") / f"llamafactory_qwen35_9b_lora_p5_s1a_ep7p5_b_rw_{arm}"
rows = []
for c in ckpts:
    mid = f"s1a_b_rw_{arm}_ckpt{c}"
    adapter = root / f"ckpt{c}"
    if not (adapter / "adapter_model.safetensors").exists():
        raise SystemExit(f"missing adapter {adapter}")
    rows.append({"id": mid, "adapter": str(adapter), "resp": "json_only"})
Path("$out").write_text(json.dumps(rows, indent=2))
print(f"wrote {len(rows)} models -> $out")
PY
}

phase1() {
  log "=== PHASE 1 start arms=[$ARMS] (parallel across GPUs) ==="
  for arm in $ARMS; do
    reweight_arm "$arm" "$SRC_B_LIGHT" "$ORDER_L16" "B_rw_${arm}_light16"
  done
  local -a pids=()
  local -a arm_list=()
  local gpu=0
  local -a arm_arr=($ARMS)
  if (( ${#arm_arr[@]} > NUM_GPUS )); then
    log "ERROR: ${#arm_arr[@]} arms > NUM_GPUS=$NUM_GPUS"
    exit 1
  fi
  for arm in "${arm_arr[@]}"; do
    arm_list+=("$arm")
    train_arm_phase1 "$arm" "$gpu" &
    pids+=("$!")
    log "launched arm=$arm pid=${pids[-1]} gpu=$gpu"
    gpu=$((gpu + 1))
  done
  local fail=0
  local i=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      log "ERROR train failed arm=${arm_list[$i]} pid=$pid"
      fail=1
    else
      log "OK train arm=${arm_list[$i]} pid=$pid"
    fi
    i=$((i + 1))
  done
  if (( fail != 0 )); then
    log "ERROR phase1 train had failures; aborting eval"
    exit 1
  fi
  local mf="$LOGDIR/reward_sweep_phase1_manifest_${TS}.json"
  write_manifest_phase1 "$mf"
  eval_manifest "$mf" "phase1"
  log "=== PHASE 1 done ==="
}

phase2() {
  local arm="${BEST_ARM:-}"
  if [[ -z "$arm" ]]; then
    log "auto-picking BEST_ARM from seed1 table..."
    local pick
    pick="$(pick_best_arm)"
    arm="$(echo "$pick" | cut -f1)"
    log "picked: $pick"
    BEST_ARM="$arm"
  fi
  log "=== PHASE 2 start BEST_ARM=$arm ==="
  reweight_arm "$arm" "$SRC_B_MED" "$ORDER_NEXT8" "B_rw_${arm}_next8"
  train_arm_phase2 "$arm"
  local mf="$LOGDIR/reward_sweep_phase2_manifest_${TS}.json"
  write_manifest_phase2 "$arm" "$mf"
  eval_manifest "$mf" "phase2"
  log "=== PHASE 2 done BEST_ARM=$arm ==="
  log "Final pick among ckpt{12,14,16,18,20,22,24} for arm=$arm by mean_balanced_norm in seed1 table."
}

case "$PHASE" in
  1) phase1 ;;
  2) phase2 ;;
  all) phase1; phase2 ;;
  *) echo "PHASE must be 1|2|all"; exit 1 ;;
esac

log "ALL COMPLETE phase=$PHASE pipe=$PIPE"
