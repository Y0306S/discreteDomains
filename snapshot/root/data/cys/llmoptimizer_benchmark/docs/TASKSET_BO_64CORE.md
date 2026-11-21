# TaskSet CASH BO on a 64-core / 256 GB DDR4 box (all paths on `/root/data`)

Authoritative code: `/root/data/cys/llmoptimizer_benchmark`  
Do **not** use `/root/data/cys/blockdata_archive/llmoptimizer_benchmark` for this job. That tree is for SFT/GRPO and its 140 G `benchmark_data/` was never copied.

## Layout

| What | Path |
| --- | --- |
| BO + resume code | `/root/data/cys/llmoptimizer_benchmark` (`mindware/domains/bo_checkpoint.py`, `search.py`, `run_taskset_bo*.py`, `run_taskset_bo_queue.sh`) |
| Optuna / OpenBox (py3.11) | `/root/data/cys/conda_envs/hpo_benchmark` |
| TF 1.15 CPU `train_inner` | `/root/data/cys/conda_envs/taskset_eval` |
| Conda name aliases | `/root/miniconda3/envs/{hpo_benchmark,taskset_eval}` → the two dirs above |
| Eval wrapper | `/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/run_taskset_dl_eval_optimized.py` (tries data python first) |
| TaskSet checkout + spaces | `/root/data/cys/google-research`, `/root/data/cys/taskset_runs` |
| 80/20 split | `/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/dl_split_80_20_task_ids.json` |
| Corpus OUT (resume here) | `/root/data/logs/taskset_corpus_train_1000s_100t` |

TF smoke: `tf 1.15.0`, `tf.test.is_built_with_cuda() == False`. GPUs stay idle.

## 64-core / 256 GB DDR4 knobs

`train_inner` hardcodes **4** TF/OMP threads. Pinning fewer than 4 CPUs per worker caused a **~12×** slowdown. Filling the machine without oversubscribe:

```text
WORKERS=16
CPUS_PER_WORKER=4
16 × 4 = 64   (workers 0,2,4,… → CPUs 0–31; workers 1,3,5,… → 32–63)
```

| Knob | Keep | Why |
| --- | --- | --- |
| `CPUS_PER_WORKER` | **4** | Must match TF threads. Do not drop to 2. |
| `WORKERS` | **16** | Saturates 64 cores. 256 GB is not the limiter; RAM headroom is large (expect well under ~8 GB/worker). |
| Oversubscribe (`WORKERS>16`) | No, unless you measure | DDR4 + SMT contention on the same cores; TPE is cheap vs eval. |
| `WORKERS=12` | Only if this box also runs other jobs | Leaves 16 cores free. |
| NUMA | leave script defaults | Even workers on 0–31, odd on 32–63. |
| GPU | unused | CPU TF1 build. |

256 GB: leave ~16–32 GB for OS/page cache. 16 workers is memory-safe. If RSS climbs (watch `ps` on `train_inner`), drop to `WORKERS=12` before touching `CPUS_PER_WORKER`.

## Protocol (corpus, not paper)

| Setting | Value |
| --- | --- |
| inner steps | 1000 |
| eval_every | 1000 |
| replicas | 1 |
| trials / task | 100 |
| TPE startup | 1 |
| optimizer | nested Optuna TPE (`--optimizer tpe`) |
| metric | `min_normalized_valid_outer` |
| split | `SIDE=train` |

Paper 10k×5 is **not** this run. `--dummy` is a fake landscape — never mix with corpus logs.

## Launch / resume (same command)

Stop leftover workers first (PID list, not `pkill -f run_taskset_bo.py`):

```bash
if [ -f /root/data/logs/taskset_corpus_train_1000s_100t/worker.pids ]; then
  xargs -r kill < /root/data/logs/taskset_corpus_train_1000s_100t/worker.pids || true
fi
```

Then:

```bash
export REPO=/root/data/cys/llmoptimizer_benchmark
export PY=/root/data/cys/conda_envs/hpo_benchmark/bin/python
export PYTHONPATH=$REPO
export TASKSET_EVAL_PYTHON=/root/data/cys/conda_envs/taskset_eval/bin/python
export OUT=/root/data/logs/taskset_corpus_train_1000s_100t

cd $REPO
WORKERS=16 CPUS_PER_WORKER=4 STEPS=1000 N_TRIALS=100 SEED=1 OPTIMIZER=tpe SIDE=train \
  OUT=$OUT bash examples/benchmark/domains/run_taskset_bo_queue.sh
```

Dry run (queue only):

```bash
DRY_RUN=1 WORKERS=16 CPUS_PER_WORKER=4 bash examples/benchmark/domains/run_taskset_bo_queue.sh
```

Resume behaviour:

- `history.json` with `n_obs >= 100` → **SKIP**
- `n_obs = k < 100` → reload Optuna pickle if COMPLETE count matches, then continue from trial **k+1** (per-trial persist)
- no history (old in-flight kill) → start from trial 1; `flock` on `$OUT/_queue/claims/$task.lock` so a dead claim is reclaimable
- `pending.json` → that config is re-evaluated (crash mid-`train_inner`)

Copy `$OUT` onto the 64-core box if it is not already the shared `/root/data/logs/...` tree. Do not change `SEED` / `N_TRIALS` / `OPTIMIZER` / `OUT` when resuming.

## Sanity before a long run

```bash
cd /root/data/cys/llmoptimizer_benchmark
export PYTHONPATH=$PWD
export PY=/root/data/cys/conda_envs/hpo_benchmark/bin/python

$PY -c 'import optuna, openbox; print("optuna", optuna.__version__)'
/root/data/cys/conda_envs/taskset_eval/bin/python -c 'import tensorflow as tf; print(tf.__version__, tf.test.is_built_with_cuda())'
$PY -m unittest tests.domains.test_taskset_cash -v
$PY examples/benchmark/domains/run_taskset_bo.py --optimizer tpe --n-trials 4 --dummy
# one real TF eval (minutes):
$PY examples/benchmark/domains/run_taskset_bo.py --optimizer tpe --n-trials 1 --smoke --task-ids mlp_family_seed12
```

## Progress / stop

```bash
OUT=/root/data/logs/taskset_corpus_train_1000s_100t
tail -f $OUT/queue.log
# tasks with a checkpoint:
find $OUT -name history.json | wc -l
# stop cleanly:
xargs -r kill < $OUT/worker.pids
```

Do not use `run_taskset_bo_parallel.sh` on this box: it still assumes 128 logical CPUs (0–31,64–95 / 32–63,96–127). The queue script is the 64-core path.

## What was not copied

`benchmark_data/` (~125 G) and other SFT/GRPO trees stay on `/root/data/cys/blockdata_archive`. TaskSet CASH does not need them.
