#!/usr/bin/env bash
# Probe trust-region TPE vs plain TPE on consistent-10 peptides (short budget for smoke).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
TASKS="pep_009,pep_231"
LOG="/root/data/logs/bo_protein_tr_tpe_probe.log"
mkdir -p "$(dirname "$LOG")"
echo "[tr-tpe-probe] $(date -Is) n_trials=80 tasks=$TASKS" | tee "$LOG"
python examples/benchmark/domains/run_bolt_bo_suite.py \
  --domain protein --optimizer tr_tpe --task-ids "$TASKS" \
  --n-trials 80 --seed 1 --device cuda:0 --skip-done \
  2>&1 | tee -a "$LOG"
echo "[tr-tpe-probe] done" | tee -a "$LOG"
