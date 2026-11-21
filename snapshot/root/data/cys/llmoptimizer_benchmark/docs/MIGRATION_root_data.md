# Relocation to `/root/data/cys/llmoptimizer_benchmark` (2026-09-11)

## Why

A second machine can reach the shared Lustre mount `/root/data` but **not**
`/root/data/cys/blockdata_archive`. The DL / TaskSet CASH work had to become runnable from
`/root/data`, so its code and data now live here.

## What this tree is

This is the authoritative checkout for **DL / TaskSet** work. It is a copy of
the code from `/root/data/cys/blockdata_archive/llmoptimizer_benchmark`, plus only the data those
domains need.

Copied here (byte-for-byte verified):

| Path | Notes |
| --- | --- |
| `mindware/`, `examples/`, `tests/`, `scripts/`, `analysis/`, `docs/` | full code tree, `__pycache__` stripped |
| `README.md`, `requirements*.txt`, `.gitignore` | |
| `history_data/domain_dl` | 178 files, 2 591 930 B |
| `benchmark_data/domain_dl` | 12 files, 40 958 B |
| `benchmark_data/domain_taskset`, `history_data/domain_taskset` | created empty; TaskSet BO outputs land here |

## What deliberately stayed on `/root/data/cys/blockdata_archive`

`/root/data/cys/blockdata_archive/llmoptimizer_benchmark` was **not** deleted and is still the
live tree for the LLM SFT/GRPO pipelines. At the time of the copy it had ~15
running jobs (a GRPO eval pipeline plus `run_mltest20_algo40_eval.sh` workers)
writing into its `benchmark_data/`. Do not remove it.

Left there:

- `benchmark_data/` — 131 G, 675 664 files (all domains except `domain_dl`)
- `history_data/` — 16 G (mostly `domain_protein`; experience banks)
- `datasets_bundle/` (7 G), `wandb/`, `catboost_info/`

`/root/data` had only 87 G free against a 154 G source tree, so a full copy was
never possible regardless.

## Path handling

The DL / TaskSet entry points needed **no path edits**. They resolve everything
from `__file__` or from env-overridable roots in `mindware/domains/paths.py`:

- `REPO_ROOT` = `Path(__file__).resolve().parents[2]` — follows the tree.
- `TASKSET_ROOT` — env, default `/root/data/cys/taskset_runs` (51 G, already on `/root/data`).
- `ZAXIE_ROOT` — env, default `/root/data/zaxie`.

One additive change was made here: `HISTORY_ROOT` and `BENCH_ROOT` now accept
`DOMAIN_HISTORY_ROOT` / `DOMAIN_BENCH_ROOT`. Defaults are unchanged, so this
tree writes beside its own code. To read the bulk trees still on blockdata:

```bash
export DOMAIN_BENCH_ROOT=/root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data
export DOMAIN_HISTORY_ROOT=/root/data/cys/blockdata_archive/llmoptimizer_benchmark/history_data
```

This is the only file that differs from the blockdata copy. It was not applied
there because that tree has live jobs; the change is default-preserving and can
be copied over once they finish.

### Non-DL references left pointing at blockdata — intentional

137 `.sh` files and 3 `.py` files still contain the literal
`/root/data/cys/blockdata_archive/llmoptimizer_benchmark`. None of them are on the DL / TaskSet
path (verified by grep). They were left alone because:

1. The data, conda envs and model weights they need are on blockdata anyway
   (`/root/data/cys/blockdata_archive/model-weights`, `/root/data/cys/blockdata_archive/miniconda3`,
   `/root/data/cys/blockdata_archive/LLMOptimizer` — 395 / 60 / 67 references, not moving).
2. Pointing them here would run code from this tree while writing into
   blockdata output dirs that live jobs are already writing — a corruption risk.
3. Several occurrences sit inside single-quoted Python heredocs, where a blanket
   shell rewrite would produce a broken literal rather than an expansion.

The three `.py` files are `examples/benchmark/bo_history_utils.py`,
`plot_real_benchmark.py`, `prepare_rec4_warm_start_presets.py` — RL warm-start,
`real_hpo_cls` plotting and rec4 presets. If you need them from this tree,
override with the env vars above or pass their `--base-dir` / path arguments.

## Verified from this tree

```bash
cd /root/data/cys/llmoptimizer_benchmark
export PYTHONPATH=/root/data/cys/llmoptimizer_benchmark

/root/miniconda3/envs/hpo_benchmark/bin/python -m unittest tests.domains.test_taskset_cash -v
python examples/benchmark/domains/run_taskset_bo.py --optimizer tpe  --n-trials 16 --dummy
python examples/benchmark/domains/run_taskset_bo.py --optimizer smac --n-trials 16 --dummy
python examples/benchmark/domains/run_taskset_bo_suite.py --optimizer tpe --n-trials 20 --max-tasks 8 --dummy
```

All passed; 2 tests OK, outputs written to
`benchmark_data/domain_taskset/<task_id>/<tpe|smac>/seed1/`.
`get_adapter("taskset")` → `TaskSetAdapter`, `get_adapter("dl")` → `DLHPOAdapter`.

## Still not wired (unchanged by this move)

Dummy BO does not call `train_inner`. Real eval is CPU TF1 (env `taskset_eval`),
paper protocol 10k steps x 5 replicas, via
`/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/run_taskset_dl_eval_optimized.py`.
Plug `{optimizer_family, hyperparameters}` into that wrapper. All of its *data*
inputs are already under `/root/data/cys`, so the real-eval path is reachable
from here:

- `resolve_task_set_root()` (line 66) tries `/root/data/cys/google-research`
  **first**, before its two blockdata/`/root` fallbacks — already portable.
- `/root/data/cys/tensorflow_datasets` (line 454) — present.
- `/root/data/cys/taskset_runs/{best_optimizer_per_task.json,scripts/optimizer_search_spaces.json,gcs_cache}` — all present.

### Conda envs (now on `/root/data`)

Both BO and TF1 eval live under `/root/data/cys/conda_envs/{hpo_benchmark,taskset_eval}`.
`/root/miniconda3/envs/{hpo_benchmark,taskset_eval}` are symlinks to those dirs.
The eval wrapper prefers `/root/data/cys/conda_envs/taskset_eval/bin/python`.

64-core / 256 GB DDR4 runbook (resume, pinning, launch):
`docs/TASKSET_BO_64CORE.md`.
