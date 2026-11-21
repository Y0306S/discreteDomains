#!/usr/bin/env bash
# Sync the peptide BO/SFT work from the blockdata repo (where it was developed) to the
# /root/data copy, so machines that only mount /root/data can run it.
#
# Deliberately TARGETED, not a blanket rsync: another agent is working in the /root/data
# copy (pt4opt_optuna_optimizer.py), and a full sync would clobber that.
set -euo pipefail

SRC="${SRC:-/root/data/cys/blockdata_archive/llmoptimizer_benchmark}"
DST="${DST:-/root/data/cys/llmoptimizer_benchmark}"

FILES=(
  # corrected BO core
  mindware/domains/protein/bolt_lolbo_runner.py
  mindware/domains/protein/bolt_lolbo_checkpoint.py
  examples/benchmark/domains/run_protein_bolt_lolbo.py
  examples/benchmark/domains/run_stbo_fixed_fleet.sh
  # tooling
  tools/build_bolt_init_pools.py
  tools/build_peptide_sft.py
  tools/stbo_status.py
  tools/bench_worker_density.sh
  tools/sync_peptide_to_data.sh
  # docs + report
  docs/BOLT_PEPTIDE_BO_FIX_PLAN.md
  docs/BOLT_PEPTIDE_SPLIT.md
  docs/PEPTIDE_SFT_HANDOFF.md
  analysis/bo_vs_init_report.html
)

DIRS=(
  configs/peptide_sft
  benchmark_data/rl_task_selection/peptide_sft_v2
)

echo "[sync] $SRC -> $DST"
for f in "${FILES[@]}"; do
  mkdir -p "$DST/$(dirname "$f")"
  cp -a "$SRC/$f" "$DST/$f"
  echo "  file $f"
done
for d in "${DIRS[@]}"; do
  # cp -a, not rsync: rsync is not installed on this box.
  rm -rf "$DST/$d"
  mkdir -p "$DST/$(dirname "$d")"
  cp -a "$SRC/$d" "$DST/$d"
  echo "  dir  $d/ ($(find "$DST/$d" -type f | wc -l) files)"
done

echo
echo "[sync] verifying the corrected runner landed"
if grep -q "init_offset_helper is a property of the \*init file\*" "$DST/mindware/domains/protein/bolt_lolbo_runner.py"; then
  echo "  OK: fixed runner present in $DST"
else
  echo "  FAIL: $DST still has the old runner" >&2; exit 1
fi
echo "[sync] done"
