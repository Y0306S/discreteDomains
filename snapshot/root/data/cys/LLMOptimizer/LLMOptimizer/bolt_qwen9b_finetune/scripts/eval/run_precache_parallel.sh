#!/usr/bin/env bash
# Run independent TFDS precache jobs in parallel (separate processes).
# Safe: each job targets different builder paths. Do NOT run two jobs that
# share the same amazon_us_reviews parent (e.g. Books bytes + subwords8k).
#
# Resume: --resume keeps .incomplete* dirs (partial Beam builds). Set RESUME=0 to wipe.
# This script never kills running jobs — it only starts missing ones.
set -euo pipefail

ROOT="/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune"
PY="/root/miniconda3/envs/taskset_eval/bin/python"
SCRIPT="$ROOT/scripts/eval/precache_datasets_v2.py"
LOGDIR="$ROOT/eval_runs/dl_test_15/parallel_precache"
mkdir -p "$LOGDIR"

BEAM_WORKERS="${BEAM_WORKERS:-32}"
TF_THREADS="${TF_THREADS:-16}"
RESUME="${RESUME:-1}"

RESUME_FLAG=()
if [[ "${RESUME}" != "0" ]]; then
  RESUME_FLAG=(--resume)
fi

run_job() {
  local name="$1"
  local pgrep_match="$2"
  shift 2
  if pgrep -f "precache_datasets_v2.py.*${pgrep_match}" >/dev/null 2>&1; then
    echo "skip $name (already running)"
    return 0
  fi
  echo "start $name -> $LOGDIR/${name}.log"
  nohup "$PY" "$SCRIPT" --skip-relocate "${RESUME_FLAG[@]}" \
    --beam-workers "$BEAM_WORKERS" \
    --tf-threads "$TF_THREADS" \
    "$@" >>"$LOGDIR/${name}.log" 2>&1 &
  echo "  pid=$!"
}

echo "Parallel precache (beam_workers=$BEAM_WORKERS, tf_threads=$TF_THREADS, resume=${RESUME})"
echo "Logs: $LOGDIR"

for lang in en ja ru zh; do
  run_job "wiki_${lang}" "20190301.${lang}_bytes" \
    --dataset "tokenized_wikipedia/20190301.${lang}_bytes"
done

for cat in Books_v1_02 Camera_v1_00 Home_v1_00 Video_v1_00; do
  run_job "amazon_${cat}" "${cat}_bytes" \
    --dataset "tokenized_amazon_reviews/${cat}_bytes" \
    --dataset "tokenized_amazon_reviews/${cat}_subwords8k"
done

if pgrep -f "precache_datasets_v2.py.*sun397_32x32" >/dev/null 2>&1; then
  echo "skip sun397 (already running)"
else
  echo "start sun397 -> $LOGDIR/sun397.log (tf_threads=64)"
  nohup "$PY" "$SCRIPT" --skip-relocate "${RESUME_FLAG[@]}" \
    --beam-workers "$BEAM_WORKERS" --tf-threads 64 \
    --dataset sun397_32x32 >>"$LOGDIR/sun397.log" 2>&1 &
  echo "  pid=$!"
fi

echo "Done. Tail: tail -f $LOGDIR/amazon_Books_v1_02.log"
