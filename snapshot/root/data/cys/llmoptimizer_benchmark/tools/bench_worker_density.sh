#!/usr/bin/env bash
# Measure BO throughput per GPU at 1, 2 and 3 workers, at the real run config.
# Each GPU gets the SAME 6-task workload, sharded across its workers, into its own
# output root. Wall-clock for all 6 tasks is the comparable number.
set -uo pipefail

REPO=/root/data/cys/blockdata_archive/llmoptimizer_benchmark
PY=/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python
G=/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated
BENCH=/root/data/logs/lolbo_bench
NTASKS="${NTASKS:-6}"
BUDGET="${BUDGET:-1000}"

rm -rf "$BENCH"; mkdir -p "$BENCH"
cd "$REPO"

# task ids: pep_100 .. pep_(100+NTASKS-1)
IDS=""
for ((i=0;i<NTASKS;i++)); do IDS="$IDS,pep_$(printf '%03d' $((100+i)))"; done
IDS="${IDS#,}"

launch() {  # gpu, nworkers
  local gpu="$1" W="$2"
  local root="$BENCH/w${W}/out"; mkdir -p "$root"
  local start; start=$(date +%s.%N)
  local pids=()
  for ((w=0;w<W;w++)); do
    # shard the same NTASKS across W workers via --n-shards/--shard-index over --task-ids
    local sub=""
    local k=0
    IFS=',' read -r -a arr <<<"$IDS"
    for ((j=0;j<${#arr[@]};j++)); do
      if (( j % W == w )); then sub="$sub,${arr[$j]}"; fi
    done
    sub="${sub#,}"
    [[ -z "$sub" ]] && continue
    PROTEIN_BO80_BOLT_ROOT="$root" BOLT_MIN_FREE_GB=1 PYTHONPATH="$REPO" \
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TQDM_DISABLE=1 \
    "$PY" -u examples/benchmark/domains/run_protein_bolt_lolbo.py \
      --run-mode continue_bolt --max-oracle-budget "$BUDGET" --n-trials "$BUDGET" \
      --bolt-init-points 1000 --bsz 10 --e2e-freq 33 --oracle-mode bolt_native \
      --no-skip-done --task-ids "$sub" --device-id 0 \
      --init-data-path "$G/refs_0_1000_init.txt" \
      --init-scores-path "$G/refs_0_1000_scores.csv" --init-offset-helper 0 \
      >"$BENCH/w${W}_worker${w}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  local end; end=$(date +%s.%N)
  local secs; secs=$(echo "$end - $start" | bc)
  local n; n=$(cat "$root/index.jsonl" 2>/dev/null | grep -c best_objective || echo 0)
  printf '%s %s %s\n' "$W" "$secs" "$n" >>"$BENCH/results.txt"
  echo "[bench] workers=$W gpu=$gpu wall=${secs}s completed=$n"
}

# run the three configs concurrently on separate GPUs
launch 1 1 &
launch 2 2 &
launch 3 3 &
wait
echo
echo "=== results (workers wall_s completed) ==="
sort -n "$BENCH/results.txt"
