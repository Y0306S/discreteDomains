#!/usr/bin/env bash
# Resume BOLT LOLBO: 100 tasks, strict 500 post-init oracle calls each, 1 job/GPU.
set -euo pipefail

REPO="${REPO:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="${LOGDIR:-/root/data/logs}"
mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"

MASTER="$LOGDIR/bo_protein_bolt_lolbo_100_parallel.log"
PIDFILE="$LOGDIR/bo_protein_bolt_lolbo_100.pids"
echo "[bolt-lolbo-100] $(date -Is) resume 4 shards (1/GPU) strict n_oracle_calls=500" | tee "$MASTER"
: >"$PIDFILE"

launch() {
  local gpu="$1" shard="$2" log="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$REPO/examples/benchmark/domains/run_protein_bolt_lolbo.py" \
    --n-tasks 100 --n-trials 500 --device-id 0 --skip-done \
    --oracle-mode bolt_native --bolt-init-points 1000 --bsz 50 --max-string-length 30 \
    --shard-index "$shard" --n-shards 4 \
    >>"$log" 2>&1 &
  echo "$!" >>"$PIDFILE"
}

launch 0 0 "$LOGDIR/bo_protein_bolt_lolbo_gpu0.log"
launch 1 1 "$LOGDIR/bo_protein_bolt_lolbo_gpu1.log"
launch 2 2 "$LOGDIR/bo_protein_bolt_lolbo_gpu2.log"
launch 3 3 "$LOGDIR/bo_protein_bolt_lolbo_gpu3.log"

echo "pids: $(tr '\n' ' ' <"$PIDFILE")" | tee -a "$MASTER"
wait $(tr '\n' ' ' <"$PIDFILE")
echo "[bolt-lolbo-100] $(date -Is) all shards finished" | tee -a "$MASTER"
