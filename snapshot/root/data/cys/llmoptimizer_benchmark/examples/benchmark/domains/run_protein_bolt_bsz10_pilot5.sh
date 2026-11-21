#!/usr/bin/env bash
# LOLBO bsz=10 ablation: 150 BO rounds × 10 = 1500 post-init oracle calls.
# e2e_freq=50 → same ~500-call stall window as BOLT bsz=50 + e2e_freq=10.
# 5 random peptides (seed 42), 4 GPUs (GPU0 runs 2 tasks in parallel).
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="/root/data/logs"
OUT_ROOT="/root/data/cys/blockdata_archive/domain_protein/bo80_bolt_bsz10_pilot5"
mkdir -p "$OUT_ROOT" "$LOGDIR"
export PYTHONPATH="$REPO"
export PROTEIN_BO80_BOLT_ROOT="$OUT_ROOT"

# seed 42 sample
TASKS=(pep_025 pep_114 pep_281 pep_654 pep_759)
# seed=42 | GPU0: pep_025 + pep_114 | GPU1: pep_281 | GPU2: pep_654 | GPU3: pep_759

COMMON=(
  examples/benchmark/domains/run_protein_bolt_lolbo.py
  --run-mode continue_bolt
  --max-oracle-budget 1500
  --n-trials 1500
  --bolt-init-points 1000
  --bsz 10
  --e2e-freq 50
  --oracle-mode bolt_native
  --no-skip-done
)

MASTER="$LOGDIR/bolt_bsz10_pilot5_launch.log"
cd "$REPO"
echo "[bsz10-pilot5] $(date -Is) tasks=${TASKS[*]} out=$OUT_ROOT" | tee "$MASTER"

launch() {
  local gpu="$1" task="$2" log="$3"
  echo "[bsz10-pilot5] gpu=$gpu task=$task -> $log" | tee -a "$MASTER"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "${COMMON[@]}" \
    --device-id 0 --task-ids "$task" \
    >>"$log" 2>&1 &
  echo "$!" >>"$LOGDIR/bolt_bsz10_pilot5.pids"
}

: >"$LOGDIR/bolt_bsz10_pilot5.pids"
launch 0 "${TASKS[0]}" "$LOGDIR/bolt_bsz10_pilot5_gpu0_${TASKS[0]}.log"
launch 0 "${TASKS[1]}" "$LOGDIR/bolt_bsz10_pilot5_gpu0_${TASKS[1]}.log"
launch 1 "${TASKS[2]}" "$LOGDIR/bolt_bsz10_pilot5_gpu1_${TASKS[2]}.log"
launch 2 "${TASKS[3]}" "$LOGDIR/bolt_bsz10_pilot5_gpu2_${TASKS[3]}.log"
launch 3 "${TASKS[4]}" "$LOGDIR/bolt_bsz10_pilot5_gpu3_${TASKS[4]}.log"

echo "[bsz10-pilot5] pids: $(tr '\n' ' ' <"$LOGDIR/bolt_bsz10_pilot5.pids")" | tee -a "$MASTER"
