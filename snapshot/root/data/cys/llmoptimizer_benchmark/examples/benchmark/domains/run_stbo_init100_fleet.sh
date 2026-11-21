#!/usr/bin/env bash
# STBO reference run, post-fix. This is the run the SFT datasets should be rebuilt from.
#
#   init pool     : 100 per-reference random mutants (>=0.75 similarity to that reference)
#   acquisition   : 100 rounds x 50 candidates  (budget-mode acquisition => 5000 acq calls)
#   E2E VAE       : fine-tune after 10 rounds without improvement (BOLT's own e2e_freq)
#   VAE           : authors'/APEXGo checkpoint (bolt_lolbo_runner defaults to it now)
#   latents       : encoded per task; the shared -train-zs.csv cache is bypassed
#
# All 1000 references are run so any train/test split (800/100, 900/100) can be carved
# out afterwards without re-running. The pre-registered test split is pep_900..pep_999.
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
PY="${PY:-/root/data/cys/blockdata_archive/miniconda3/envs/openbox_qwen_eval/bin/python}"
LOGDIR="/root/data/logs/stbo_init100_fleet"
GEN="/root/data/cys/BOLT/optimization/peptides/apex_oracle/init_data/generated"
OUT="${OUT:-/root/data/cys/blockdata_archive/domain_protein/stbo_init100_r100_bsz50_e2e10}"
N_TASKS="${N_TASKS:-1000}"
INIT_POINTS="${INIT_POINTS:-100}"
BSZ="${BSZ:-50}"
ROUNDS="${ROUNDS:-100}"
E2E_FREQ="${E2E_FREQ:-10}"
GPUS_CSV="${GPUS:-0,1,2,3}"
PER_GPU="${PER_GPU:-5}"
THREADS="${THREADS:-6}"
PIN="${PIN:-0}"   # 1 = taskset-pin each worker to THREADS cores
BUDGET=$((ROUNDS * BSZ))

IFS=',' read -r -a GPUS <<<"$GPUS_CSV"
NGPU="${#GPUS[@]}"
N_SHARDS=$((NGPU * PER_GPU))

mkdir -p "$OUT" "$LOGDIR"
export PYTHONPATH="$REPO"
export BOLT_MIN_FREE_GB="${BOLT_MIN_FREE_GB:-50}"
cd "$REPO"

PIDFILE="$LOGDIR/fleet.pids"
: >"$PIDFILE"
MANIFEST="$LOGDIR/config.json"
cat >"$MANIFEST" <<JSON
{
  "out_root": "$OUT",
  "n_tasks": $N_TASKS,
  "init_points": $INIT_POINTS,
  "bsz": $BSZ,
  "rounds": $ROUNDS,
  "acquisition_budget": $BUDGET,
  "e2e_freq": $E2E_FREQ,
  "budget_mode": "acquisition",
  "init_data": "$GEN/refs_0_1000_init.txt",
  "init_scores": "$GEN/refs_0_1000_scores.csv",
  "init_offset_helper": 0,
  "n_shards": $N_SHARDS,
  "gpus": "$GPUS_CSV",
  "per_gpu": $PER_GPU,
  "test_split": "pep_900..pep_999 (pre-registered)",
  "started": "$(date -Is)"
}
JSON

echo "[stbo-init100] $(date -Is)"
echo "[stbo-init100] tasks=$N_TASKS init=$INIT_POINTS bsz=$BSZ rounds=$ROUNDS e2e_freq=$E2E_FREQ"
echo "[stbo-init100] acq_budget=$BUDGET shards=$N_SHARDS (${NGPU} gpus x ${PER_GPU}) out=$OUT"

BASE_CPU=32
shard=0
for ((g=0; g<NGPU; g++)); do
  for ((w=0; w<PER_GPU; w++)); do
    cpu_start=$((BASE_CPU + shard * THREADS))
    cpu_end=$((cpu_start + THREADS - 1))
    log="$LOGDIR/shard${shard}_gpu${GPUS[$g]}.log"
    PINCMD=""
    [[ "$PIN" == "1" ]] && PINCMD="taskset -c ${cpu_start}-${cpu_end}"
    CUDA_VISIBLE_DEVICES="${GPUS[$g]}" \
    PROTEIN_BO80_BOLT_ROOT="$OUT" \
    OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
    OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" \
    TQDM_DISABLE=1 \
    ${PINCMD:+$PINCMD} \
    nohup "$PY" -u examples/benchmark/domains/run_protein_bolt_lolbo.py \
      --run-mode continue_bolt \
      --budget-mode acquisition \
      --max-oracle-budget "$BUDGET" \
      --n-trials "$BUDGET" \
      --bolt-init-points "$INIT_POINTS" \
      --bsz "$BSZ" \
      --e2e-freq "$E2E_FREQ" \
      --oracle-mode bolt_native \
      --skip-done \
      --n-tasks "$N_TASKS" \
      --n-shards "$N_SHARDS" \
      --shard-index "$shard" \
      --init-data-path "$GEN/refs_0_1000_init.txt" \
      --init-scores-path "$GEN/refs_0_1000_scores.csv" \
      --init-offset-helper 0 \
      --device-id 0 \
      >>"$log" 2>&1 &
    echo "$!" >>"$PIDFILE"
    shard=$((shard + 1))
  done
done

echo "[stbo-init100] launched $(wc -l <"$PIDFILE") workers"
echo "[stbo-init100] progress: wc -l $OUT/index.jsonl   (target $N_TASKS)"
echo "[stbo-init100] stop: kill \$(cat $PIDFILE)"
