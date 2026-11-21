#!/usr/bin/env bash
# Budget probe: do 100 rounds x {20,30} candidates beat the best init point on tasks
# where 100 rounds x 10 did not?
#
# Paired design: all 10 tasks had ZERO improvement over their init-pool best in
# stbo_bsz10_r100_fixed, so that run is the control and any win here is unambiguous.
# Everything except --bsz / --max-oracle-budget is held fixed (same init pool, same
# e2e-freq, same 1000 init points, budget-mode=acquisition => exactly 100 rounds).
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="/root/data/logs"
GEN="/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated"
TASKS="${TASKS:-pep_900,pep_901,pep_902,pep_905,pep_906,pep_925,pep_940,pep_912,pep_913,pep_917}"
ROUNDS="${ROUNDS:-100}"
E2E_FREQ="${E2E_FREQ:-33}"
ARMS="${ARMS:-20 30}"
INIT_POINTS="${INIT_POINTS:-1000}"
GPUS=(0 1 2 3)

export PYTHONPATH="$REPO"
export BOLT_MIN_FREE_GB="${BOLT_MIN_FREE_GB:-30}"
cd "$REPO"

IFS=',' read -r -a TASK_ARR <<<"$TASKS"
NT="${#TASK_ARR[@]}"
PIDFILE="$LOGDIR/stbo_budget_probe.pids"
: >"$PIDFILE"

launch_arm() {
  local bsz="$1"
  local budget=$((ROUNDS * bsz))
  # e2e_freq is part of the identity of a run, so keep it in the path (33 was the
  # original probe; 10 is BOLT's own setting).
  local suffix=""
  [[ "$E2E_FREQ" != "33" ]] && suffix="_e2e${E2E_FREQ}"
  [[ "$INIT_POINTS" != "1000" ]] && suffix="${suffix}_init${INIT_POINTS}"
  [[ -n "${TAG:-}" ]] && suffix="${suffix}_${TAG}"
  local out="/root/data/cys/blockdata_archive/domain_protein/stbo_r100_bsz${bsz}${suffix}_probe"
  mkdir -p "$out"
  echo "[probe] arm bsz=$bsz e2e_freq=$E2E_FREQ init=$INIT_POINTS rounds=$ROUNDS acq_budget=$budget out=$out"

  # deal the 10 tasks round-robin across the 4 GPUs
  local ngpu="${#GPUS[@]}"
  for ((g=0; g<ngpu; g++)); do
    local ids=""
    for ((t=g; t<NT; t+=ngpu)); do ids="${ids:+$ids,}${TASK_ARR[$t]}"; done
    [[ -z "$ids" ]] && continue
    local log="$LOGDIR/stbo_probe_bsz${bsz}${suffix}_gpu${GPUS[$g]}.log"
    echo "[probe]   gpu=${GPUS[$g]} tasks=$ids -> $log"
    CUDA_VISIBLE_DEVICES="${GPUS[$g]}" \
    PROTEIN_BO80_BOLT_ROOT="$out" \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    TQDM_DISABLE=1 \
    nohup "$PY" -u examples/benchmark/domains/run_protein_bolt_lolbo.py \
      --run-mode continue_bolt \
      --budget-mode acquisition \
      --max-oracle-budget "$budget" \
      --n-trials "$budget" \
      --bolt-init-points "$INIT_POINTS" \
      --bsz "$bsz" \
      --e2e-freq "$E2E_FREQ" \
      --oracle-mode bolt_native \
      --no-skip-done \
      --task-ids "$ids" \
      --init-data-path "$GEN/refs_0_1000_init.txt" \
      --init-scores-path "$GEN/refs_0_1000_scores.csv" \
      --init-offset-helper 0 \
      --device-id 0 \
      >>"$log" 2>&1 &
    echo "$!" >>"$PIDFILE"
  done
}

for bsz in $ARMS; do launch_arm "$bsz"; done

echo "[probe] pids: $(tr '\n' ' ' <"$PIDFILE")"
