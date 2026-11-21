# TaskSet CPU Campaign

Campaign: `/root/data/logs/taskset_major_families_cpu_20260929`.

The campaign uses the approved `taskset_major_families_split_80_20_v1.json`
split: 838 training tasks and 210 held-out tasks. It collects the 11 missing
training histories and evaluates Random Search, TPE and the corrected no-VAE
LOLBO adaptation on all 210 held-out tasks. The 827 complete training histories
are not scheduled. No offline model training is launched by this campaign.

## Evaluation Settings

- Seed 1; exactly 100 evaluated configurations per task, including failed fits.
- 1,000 inner training steps, validation every 1,000 steps, one replica.
- Minimize `min_normalized_valid_outer` in the existing conditional CASH space.
- TPE uses one random startup trial, matching the source BO corpus.
- LOLBO uses 10 initialization trials, batches of 10, and the existing ML
  surrogate defaults: 20 initial GP epochs, 2 subsequent epochs, learning rate
  0.01 and the default 2,000-5,000 candidate pool. Initialization counts toward
  the 100-evaluation limit.
- Source backfill: 11 tasks / 1,100 evaluations. Held-out evaluation: 630
  task-method runs / 63,000 evaluations. Total: 64,100 new evaluations.

Training histories are written into
`/root/data/logs/taskset_corpus_train_1000s_100t/<task>/tpe/seed1`.
Held-out histories are written into `<campaign>/test/<task>/<method>/seed1`.
Each evaluation checkpoints its history and optimizer state. Individual failures
remain in the history; they are not replaced by extra evaluations.

## Resource Allocation

This container sees 128 logical CPUs but has a quota of 40 CPU cores. Its memory
limit is 400 GiB, even though the host exposes about 1 TiB of RAM. A prelaunch
sample measured approximately 25 busy CPU cores in the container.

| Work | CPU affinity | GPUs |
| --- | --- | --- |
| Existing ML evaluation workers | 0-22 | 0, 1, 2, 3 |
| BOLT SFT allocation (resume later failed) | 24-27 and 28-31 | 0, 1 |
| Existing OptFormer training | Existing unrestricted affinity | 2 |
| Backfill worker 0 | 32-35 | None |
| Backfill worker 1 | 36-39 | None |
| Random Search worker | 32-35 | None |
| TPE worker | 36-39 | None |
| LOLBO worker | 32-39 | None |

The five new workers share eight logical CPUs in total. Each numerical library
uses four threads. Backfill workers use nice 10; baseline workers use nice 15.
Their child processes inherit CPU affinity and an empty `CUDA_VISIBLE_DEVICES`.
Existing jobs and affinities are not changed. Sharing CPUs bounds the new load
but can slow individual evaluations; GPU utilization and CPU demand still vary.

Workers defer new task dispatch when the container has less than 20 GiB of
memory headroom after allowing for reclaimable inactive file cache. This is a
dispatch check, not a reservation or a memory limit on existing jobs.

After launch, a five-second sample measured 28 busy cores, zero CPU-quota
throttled periods and about 189 GiB of container memory usage. The BOLT resume
had exited at 23:01:48 local time, before this TaskSet launch: Transformers
rejected RNG-state restoration because its `torch.load` path requires PyTorch
2.6 or newer. Its ranks were no longer on GPUs 0-1 in the subsequent check.
Those GPUs still held ML evaluation workers. GPU 2 held OptFormer and ML
evaluation workers; GPU 3 held ML evaluation workers. Existing environments and
jobs were not modified to address the BOLT error.

## Frozen Inputs And Isolation

`campaign.json` records the exact jobs, commands, settings and resource plan;
`split.json` is a copy of the approved split. `source_hashes.json` identifies the
code, scoring wrapper, task definitions and search-space snapshots.

Each task-method run gets a private TaskSet runtime with its own optimizer
registry. It uses the same training code and shared dataset cache, without
rescanning the large global registry or registering new configurations there.
Task order interleaves families, starting with the cheaper quadratic family for
held-out runs. All three methods use the same task order.

## Operations

The driver is `examples/benchmark/domains/run_taskset_cpu_campaign.py`.
Preparation validates split membership, budgets and baseline availability, then
freezes the inputs. Launch verifies the snapshot hashes and starts five detached
workers. Worker and job locks prevent duplicate dispatch within the campaign.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python \
  examples/benchmark/domains/run_taskset_cpu_campaign.py status \
  --campaign /root/data/logs/taskset_major_families_cpu_20260929
```

`launch.json` contains worker PIDs. `workers/*.json` records each worker's
current task, child PID and log. `job_logs/*.log` contains evaluator output;
`results/*.json` records completed run counts and failures. A process error or
incomplete 100-trial result stops that worker and records the error. Failed
workers require inspection before restarting with the `worker` action and the
same worker name; saved observations and optimizer state are resumed.

The split's original coverage audit remains a historical snapshot. Re-audit the
source corpus after backfill completion before FSBO/RGPE/OptFormer exports.
