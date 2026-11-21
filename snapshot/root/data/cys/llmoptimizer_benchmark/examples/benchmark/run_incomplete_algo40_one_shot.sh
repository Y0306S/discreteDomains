#!/usr/bin/env bash
# Scan every GPT P5+ / RC / RS arm and every Qwen s1b_attr_t2 P5+ TPE arm that is
# not algo-correct 40/40, then launch all missing (dataset, algorithm) slots.
#
# Resume now counts only the requested algorithm, so sibling algos sharing a
# dataset/variant folder are not skipped. Jobs that share (task, dataset,
# variant) are serialized so they cannot clobber each other.
#
#   bash examples/benchmark/run_incomplete_algo40_one_shot.sh
#   FAMILY=qwen bash ...          # Qwen P5+ only (no GPT API jobs)
#   FAMILY=gpt bash ...           # GPT P5+/RC/RS only
#   GPT_PARALLEL=6 NUM_GPUS=4 JOBS_PER_GPU=3 bash ...
#   DRY_RUN=1 bash ...
#
set -euo pipefail

REPO="/root/data/cys/blockdata_archive/llmoptimizer_benchmark"
cd "$REPO"

TASKS_JSON="${TASKS_JSON:-benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json}"
SUMMARY_CSV="${SUMMARY_CSV:-benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus/gpt_p5_p5plus_summary.csv}"
GPT_ROOT="${GPT_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40}"
QWEN_ROOT="${QWEN_ROOT:-benchmark_data/real_hpo_v2_mltest20_algo40_p5_s1b_attr_t2_p5plus}"
LOGDIR="${LOGDIR:-benchmark_data/real_hpo_v2_mltest20_algo40/logs}"
PYTHON_GPT="${PYTHON_GPT:-/root/miniconda3/envs/openbox_eval/bin/python}"
PYTHON_QWEN="${PYTHON_QWEN:-/root/miniconda3/envs/openbox_qwen_eval/bin/python}"
BASE_MODEL="${BASE_MODEL:-/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B}"
ADAPTER_PATH="${ADAPTER_PATH:-/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2}"
N_TRIALS="${N_TRIALS:-80}"
SEED="${SEED:-1}"
TIME_LIMIT="${TIME_LIMIT:-7200}"
GPT_PER_RUN="${GPT_PER_RUN:-300}"
QWEN_PER_RUN="${QWEN_PER_RUN:-1800}"
DATASET_RETRIES="${DATASET_RETRIES:-3}"
GPT_PARALLEL="${GPT_PARALLEL:-6}"
NUM_GPUS="${NUM_GPUS:-4}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"
PACK_MIN_FREE_MB="${PACK_MIN_FREE_MB:-20000}"
POLL_SEC="${POLL_SEC:-15}"
MAX_PASSES="${MAX_PASSES:-3}"
DRY_RUN="${DRY_RUN:-0}"
FAMILY="${FAMILY:-all}"  # all | gpt | qwen
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG="$LOGDIR/incomplete_algo40_one_shot_${TS}.log"
PIDFILE="$LOGDIR/incomplete_algo40_one_shot.pid"

mkdir -p "$LOGDIR" "$GPT_ROOT" "$QWEN_ROOT"
export PYTHONPATH="$REPO/examples/benchmark:${REPO}${PYTHONPATH:+:$PYTHONPATH}"
echo $$ >"$PIDFILE"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER_LOG"; }

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
is_alive() { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }

scan_manifest() {
  local out="$1"
  "$PYTHON_GPT" - "$out" "$GPT_ROOT" "$QWEN_ROOT" "$TASKS_JSON" "$SUMMARY_CSV" "$FAMILY" <<'PY'
import csv, sys
from pathlib import Path
from plot_mltest20_algo40_avg_rank import find_traj, load_tasks

manifest_path = Path(sys.argv[1])
gpt_root = Path(sys.argv[2])
qwen_root = Path(sys.argv[3])
tasks = load_tasks(Path(sys.argv[4]))
summary_csv = Path(sys.argv[5])
family_filter = sys.argv[6] if len(sys.argv) > 6 else "all"

def parse_variant(v: str):
    parts = v.split("__")
    opt = parts[0]
    prompt = parts[1] if len(parts) > 1 else ""
    hybrid = parts[2] if len(parts) > 2 else ""
    return opt, prompt, hybrid

def missing_jobs(family, root, disk):
    opt, prompt, hybrid = parse_variant(disk)
    rows = []
    n_ok = 0
    for t in tasks:
        if find_traj(root, t["task_type"], t["dataset"], disk, 80, algorithm=t["algorithm"]):
            n_ok += 1
            continue
        rows.append(f"{family}|{opt}|{prompt}|{hybrid}|{t['task_type']}|{t['dataset']}|{t['algorithm']}")
    return n_ok, rows

jobs = []
print("algo-correct coverage (40 task slots = dataset×algorithm):")
print(f"{'family':<6} {'folder':>7} {'algo':>7}  variant")

if family_filter in ("all", "gpt"):
    extra = [
        "ptllm_tpe__P5plusA5T50X50__H2",
        "ptllm_tpe__P5plusA5T50X50RC__H2",
        "ptllm_tpe__P5plusA5T50X50RS__H2",
    ]
    seen = set()
    for disk in extra:
        seen.add(disk)
        folder = sum(1 for t in tasks if find_traj(gpt_root, t["task_type"], t["dataset"], disk, 80, algorithm=None))
        n_ok, miss = missing_jobs("gpt", gpt_root, disk)
        print(f"{'gpt':<6} {folder:3d}/40 {n_ok:3d}/40  {disk}")
        jobs.extend(miss)

    for row in csv.DictReader(summary_csv.open(encoding="utf-8")):
        gv = row["variant"].strip()
        if gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
            continue
        if gv in seen:
            continue
        seen.add(gv)
        folder = sum(1 for t in tasks if find_traj(gpt_root, t["task_type"], t["dataset"], gv, 80, algorithm=None))
        n_ok, miss = missing_jobs("gpt", gpt_root, gv)
        if n_ok < 40 or miss:
            print(f"{'gpt':<6} {folder:3d}/40 {n_ok:3d}/40  {gv}")
        jobs.extend(miss)

if family_filter in ("all", "qwen"):
    for row in csv.DictReader(summary_csv.open(encoding="utf-8")):
        gv = row["variant"].strip()
        if gv.startswith("ptllm__") and not gv.startswith("ptllm_tpe"):
            continue
        disk = gv.replace("ptllm_tpe__", "ptllm_tpe_qwen__", 1).replace("ptllm__", "ptllm_qwen__", 1)
        folder = sum(1 for t in tasks if find_traj(qwen_root, t["task_type"], t["dataset"], disk, 80, algorithm=None))
        n_ok, miss = missing_jobs("qwen", qwen_root, disk)
        if n_ok < 40 or miss:
            print(f"{'qwen':<6} {folder:3d}/40 {n_ok:3d}/40  {disk}")
        jobs.extend(miss)

manifest_path.write_text("\n".join(jobs) + ("\n" if jobs else ""), encoding="utf-8")
n_gpt = sum(1 for j in jobs if j.startswith("gpt|"))
n_qwen = sum(1 for j in jobs if j.startswith("qwen|"))
print(f"repair_jobs={len(jobs)} gpt={n_gpt} qwen={n_qwen}")
PY
}

run_gpt_job() {
  local opt_base="$1" prompt="$2" hybrid="$3" tt="$4" ds="$5" algo="$6"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local logf="$LOGDIR/oneshot_gpt_${tag}_${ds}_${algo}_${TS}.log"
  log "  gpt START $tag $tt/$ds/$algo"
  local runner=(
    "$PYTHON_GPT" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$algo"
    --task-type "$tt"
    --dataset "$ds"
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval
    --llm-backends gpt4o-mini
    --output-dir "$GPT_ROOT"
    --time-limit "$TIME_LIMIT"
    --per-run-time-limit "$GPT_PER_RUN"
    --dataset-retries "$DATASET_RETRIES"
    --optimizer "$opt_base"
    --prompt-variant "$prompt"
    --resume
  )
  [[ -n "$hybrid" ]] && runner+=( --hybrid-variant "$hybrid" )
  [[ "$DRY_RUN" == "1" ]] && runner+=( --llm-mock )
  set +e
  "${runner[@]}" >>"$logf" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "  WARN gpt rc=$rc $tag $tt/$ds/$algo (see $logf)"
  else
    log "  gpt DONE $tag $tt/$ds/$algo"
  fi
}

run_qwen_job() {
  local opt_base="$1" prompt="$2" hybrid="$3" tt="$4" ds="$5" algo="$6" gpu="$7"
  local tag="${opt_base}__${prompt}"
  [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
  local logf="$LOGDIR/oneshot_qwen_${tag}_${ds}_${algo}_${TS}.log"
  log "  qwen gpu=$gpu START $tag $tt/$ds/$algo"
  local runner=(
    "$PYTHON_QWEN" examples/benchmark/run_toy_categorical_optimizers.py
    --algorithm "$algo"
    --task-type "$tt"
    --dataset "$ds"
    --n-trials "$N_TRIALS" --seed "$SEED" --skip-test-eval
    --llm-backends qwen-sft6 --qwen-backend hf
    --qwen-model-path "$BASE_MODEL"
    --qwen-adapter-path "$ADAPTER_PATH"
    --llm-max-tokens 768
    --llm-response-format rationale_json
    --output-dir "$QWEN_ROOT"
    --time-limit "$TIME_LIMIT"
    --per-run-time-limit "$QWEN_PER_RUN"
    --dataset-retries "$DATASET_RETRIES"
    --optimizer "$opt_base"
    --prompt-variant "$prompt"
    --resume
  )
  [[ -n "$hybrid" ]] && runner+=( --hybrid-variant "$hybrid" )
  [[ "$DRY_RUN" == "1" ]] && runner+=( --llm-mock )
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${runner[@]}" >>"$logf" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "  WARN qwen rc=$rc $tag $tt/$ds/$algo (see $logf)"
  else
    log "  qwen DONE $tag $tt/$ds/$algo"
  fi
}

run_gpt_pool() {
  local manifest="$1"
  mapfile -t JOBS < "$manifest"
  local total=${#JOBS[@]}
  local queue_idx=0
  declare -A GPT_PID=()
  declare -A GPT_KEY=()
  log "gpt API pool: $total jobs  parallel=$GPT_PARALLEL"

  while [[ $queue_idx -lt $total ]] || {
      local any=0 i
      for ((i=0; i<GPT_PARALLEL; i++)); do
        is_alive "${GPT_PID[$i]:-}" && any=1
      done
      [[ $any -eq 1 ]]
    }; do
    local i
    for ((i=0; i<GPT_PARALLEL; i++)); do
      if is_alive "${GPT_PID[$i]:-}"; then continue; fi
      GPT_PID[$i]=
      GPT_KEY[$i]=
      [[ $queue_idx -ge $total ]] && continue

      local found=-1 j family opt_base prompt hybrid tt ds algo tag key k busy
      for ((j=queue_idx; j<total; j++)); do
        IFS='|' read -r family opt_base prompt hybrid tt ds algo <<<"${JOBS[$j]}"
        tag="${opt_base}__${prompt}"
        [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
        key="gpt|$tt|$ds|$tag"
        busy=0
        for k in "${!GPT_KEY[@]}"; do
          if is_alive "${GPT_PID[$k]:-}" && [[ "${GPT_KEY[$k]:-}" == "$key" ]]; then
            busy=1
            break
          fi
        done
        [[ $busy -eq 0 ]] || continue
        found=$j
        break
      done
      [[ $found -ge 0 ]] || continue
      if [[ $found -ne $queue_idx ]]; then
        local tmp="${JOBS[$queue_idx]}"
        JOBS[$queue_idx]="${JOBS[$found]}"
        JOBS[$found]="$tmp"
      fi
      IFS='|' read -r family opt_base prompt hybrid tt ds algo <<<"${JOBS[$queue_idx]}"
      tag="${opt_base}__${prompt}"
      [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
      key="gpt|$tt|$ds|$tag"
      queue_idx=$((queue_idx + 1))
      log "dispatch gpt $queue_idx/$total $tag $ds/$algo"
      ( run_gpt_job "$opt_base" "$prompt" "$hybrid" "$tt" "$ds" "$algo" ) &
      GPT_PID[$i]=$!
      GPT_KEY[$i]="$key"
    done
    sleep "$POLL_SEC"
  done
  wait || true
}

run_qwen_pool() {
  local manifest="$1"
  mapfile -t JOBS < "$manifest"
  local total=${#JOBS[@]}
  local queue_idx=0
  declare -A QWEN_PID=()
  declare -A QWEN_KEY=()
  log "qwen GPU pool: $total jobs  gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU"

  while [[ $queue_idx -lt $total ]] || {
      local any=0 gpu slot
      for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
          is_alive "${QWEN_PID[${gpu}_${slot}]:-}" && any=1
        done
      done
      [[ $any -eq 1 ]]
    }; do
    local gpu slot
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
      for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
        if is_alive "${QWEN_PID[${gpu}_${slot}]:-}"; then continue; fi
        QWEN_PID[${gpu}_${slot}]=
        QWEN_KEY[${gpu}_${slot}]=
        [[ $queue_idx -ge $total ]] && continue

        local count=0 s
        for ((s=0; s<JOBS_PER_GPU; s++)); do
          is_alive "${QWEN_PID[${gpu}_${s}]:-}" && count=$((count + 1))
        done
        [[ $count -lt $JOBS_PER_GPU ]] || continue
        if [[ $count -ge 1 ]]; then
          local free
          free=$(gpu_free_mb "$gpu")
          [[ "${free:-0}" -ge $PACK_MIN_FREE_MB ]] || continue
        fi

        local found=-1 j family opt_base prompt hybrid tt ds algo tag key k busy
        for ((j=queue_idx; j<total; j++)); do
          IFS='|' read -r family opt_base prompt hybrid tt ds algo <<<"${JOBS[$j]}"
          tag="${opt_base}__${prompt}"
          [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
          key="qwen|$tt|$ds|$tag"
          busy=0
          for k in "${!QWEN_KEY[@]}"; do
            if is_alive "${QWEN_PID[$k]:-}" && [[ "${QWEN_KEY[$k]:-}" == "$key" ]]; then
              busy=1
              break
            fi
          done
          [[ $busy -eq 0 ]] || continue
          found=$j
          break
        done
        [[ $found -ge 0 ]] || continue
        if [[ $found -ne $queue_idx ]]; then
          local tmp="${JOBS[$queue_idx]}"
          JOBS[$queue_idx]="${JOBS[$found]}"
          JOBS[$found]="$tmp"
        fi
        IFS='|' read -r family opt_base prompt hybrid tt ds algo <<<"${JOBS[$queue_idx]}"
        tag="${opt_base}__${prompt}"
        [[ -n "$hybrid" ]] && tag="${tag}__${hybrid}"
        key="qwen|$tt|$ds|$tag"
        queue_idx=$((queue_idx + 1))
        log "dispatch qwen $queue_idx/$total gpu=$gpu $tag $ds/$algo"
        ( run_qwen_job "$opt_base" "$prompt" "$hybrid" "$tt" "$ds" "$algo" "$gpu" ) &
        QWEN_PID[${gpu}_${slot}]=$!
        QWEN_KEY[${gpu}_${slot}]="$key"
      done
    done
    sleep "$POLL_SEC"
  done
  wait || true
}

log "=== incomplete algo40 one-shot repair family=$FAMILY ==="
log "gpt_root=$GPT_ROOT qwen_root=$QWEN_ROOT"
log "gpt_parallel=$GPT_PARALLEL gpus=$NUM_GPUS jobs_per_gpu=$JOBS_PER_GPU max_passes=$MAX_PASSES"

for ((pass=1; pass<=MAX_PASSES; pass++)); do
  MANIFEST="$LOGDIR/incomplete_algo40_manifest_${TS}_pass${pass}.txt"
  log "pass $pass/$MAX_PASSES scanning → $MANIFEST"
  scan_manifest "$MANIFEST" | tee -a "$MASTER_LOG"
  if [[ ! -s "$MANIFEST" ]]; then
    log "all scanned arms are algo-correct 40/40"
    break
  fi
  GPT_MANIFEST="$LOGDIR/incomplete_algo40_gpt_${TS}_pass${pass}.txt"
  QWEN_MANIFEST="$LOGDIR/incomplete_algo40_qwen_${TS}_pass${pass}.txt"
  : >"$GPT_MANIFEST"
  : >"$QWEN_MANIFEST"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] || continue
    case "$line" in
      gpt\|*)
        [[ "$FAMILY" == "qwen" ]] || echo "$line" >>"$GPT_MANIFEST"
        ;;
      qwen\|*)
        [[ "$FAMILY" == "gpt" ]] || echo "$line" >>"$QWEN_MANIFEST"
        ;;
    esac
  done < "$MANIFEST"
  n_gpt=$(grep -cve '^$' "$GPT_MANIFEST" || true)
  n_qwen=$(grep -cve '^$' "$QWEN_MANIFEST" || true)
  log "pass $pass launching gpt=$n_gpt qwen=$n_qwen"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN: not launching jobs"
    break
  fi
  gpt_pid=""
  qwen_pid=""
  if [[ "$n_gpt" -gt 0 ]]; then
    run_gpt_pool "$GPT_MANIFEST" &
    gpt_pid=$!
  fi
  if [[ "$n_qwen" -gt 0 ]]; then
    run_qwen_pool "$QWEN_MANIFEST" &
    qwen_pid=$!
  fi
  [[ -n "$gpt_pid" ]] && wait "$gpt_pid" || true
  [[ -n "$qwen_pid" ]] && wait "$qwen_pid" || true
  log "pass $pass pools finished"
done

FINAL_MANIFEST="$LOGDIR/incomplete_algo40_manifest_${TS}_final.txt"
log "final scan → $FINAL_MANIFEST"
scan_manifest "$FINAL_MANIFEST" | tee -a "$MASTER_LOG"
if [[ -s "$FINAL_MANIFEST" ]]; then
  log "WARN still incomplete: $(grep -cve '^$' "$FINAL_MANIFEST") jobs in $FINAL_MANIFEST"
  exit 2
fi
log "=== family=$FAMILY algo-correct 40/40 complete ==="
if [[ "$FAMILY" != "gpt" ]]; then
  log "rebuilding seed1 leaderboard (algo-correct task slots, no --ignore-task-algorithm)"
  "$PYTHON_GPT" examples/benchmark/plot_mltest20_algo40_avg_rank.py \
    --seed1-only --rgs-pooled-trial-norm | tee -a "$MASTER_LOG"
fi
log "master_log=$MASTER_LOG"
rm -f "$PIDFILE"
