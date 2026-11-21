# Peptide TPE runtime changes and resumption

Prepared on 2026-10-28 (Asia/Shanghai). Infrastructure changes 1-3 are implemented,
validated with real inference, and deployed to the existing test200 campaign.
The worker queue was resumed with 22 workers on LLMOptimizer2 and 36 on
LLMOptimizer4. `live_status.json` records the latest observed progress and memory.

The production check observed all 58 workers on distinct tasks, with readable
recovery databases and post-initialization trials in every database. Worker logs
contained no tracebacks or inference/storage errors; all eight services,
including the unchanged export service, reported zero errors. Container OOM
counters were zero on both machines. Memory use was about 81 GiB locally and
40 GiB remotely. Export progress was 700/800; source training remained queued.

## Changes

1. **Keep identical frozen weights loaded.** Seven inference services fingerprint
   validated weight banks and skip reloading weights already active on the GPU,
   including when the same frozen weights are used by another task. Different
   weights still trigger a strict reload. The export service on port 18874 is
   excluded from deployment and TPE routing.
2. **Reuse inference connections.** The TPE wrapper uses persistent HTTP connections
   through the existing loopback/SSH services. Transport retries retain the exact
   request body and request ID. Original task identities, RNG seeds and oracle
   nonces are preserved.
3. **Measure the bottlenecks.** Sampling, serialization, inference queueing,
   inference execution and checkpoint commits are timed. Profiling uses one
   shared SQLite file, flushed every 25 tells and on task exit. The recovery
   database still uses synchronous FULL transactions at the original boundaries.

The implementation wraps the pinned `tools/peptide_test200.py` adapter using
`tools/peptide_tpe_runtime.py` and `tools/peptide_tpe_rpc.py`. The launcher verifies
their hashes against a passed validation report stored on shared storage, so
both machines use the same validated runtime.

## Search space and budget

- Frozen author peptide VAE; 256 latent coordinates, each in `[-4, 4]`.
- The same peptide decoder and similarity/validity rules used for LOLBO.
- TPE remains batch one with its existing Optuna sampler.
- 100 measured initialization observations; at most 5,000 proposals and 5,000
  fresh post-initialization oracle evaluations. Cached repeats remain recorded.
- LOLBO's moving trust region is a different sampling policy and has not been
  added to this resumed campaign. These changes do not address TPE's low rate of
  similarity-feasible proposals by altering its proposal distribution.

No production recipe, optimizer state, historical score or pinned BO release
was rewritten. Before resumption, 29 tasks had 56,375 proposals and 51,419 fresh
evaluations saved; 171 tasks had not started. There were no completed TPE tasks.
The separate LOLBO test200 campaign had completed all 200 tasks.

## Allocation and training queue

| Machine | TPE workers | CPU allocation | Other reservations |
| --- | ---: | --- | --- |
| LLMOptimizer2 | 22 | 12-17, 19-20, 24-37 | BOLT, seven inference services, export CPU 18 and service CPU 39 |
| LLMOptimizer4 | 36 | 10-45 | CPUs 0-7 for RGPE; 8-9 for FSBO; 46-47 for the training supervisor |
| LLMOptimizer3 | 0 | None | User reservation retained |

The existing export supervisor continues to launch FSBO/RGPE source training
after all 800 source tasks have been exported. RGPE concurrency is scaled using
the measured peak memory of a complete source fit. TPE workers process their
assigned tasks and then take unlocked remaining tasks from the same queue.

This deployment does not complete the unfinished FSBO/RGPE evaluation adapters
or claim that their training is finished.

## Validation

`validation.json` passed the real component test:

- Seeded decoding matches the original service; persistent socket reuse works.
- Identical weights across task identities are loaded once; changed weights
  reload, and restoring the frozen weights restores identical encodings.
- An injected lost oracle response retries without a second oracle evaluation.
- Three additional TPE rounds from copied recovery databases produce identical
  proposals, peptides, scores, feasibility flags and accounting.
- A recorded, scored pending trial resumes with scoring prohibited.
- Completed archives verify, and the original campaign/database hashes remain
  unchanged by the diagnostic.

`pool_validation.json` additionally verifies all seven upgraded services using
seeded decoding and repeated requests. This pool check made zero oracle calls.
The component check made at most seven fresh diagnostic calls in isolated copies.
Two test-helper issues were corrected before launch: missing weight registration
and a local-only report path that the remote machine could not access.

Early production profiling showed sampling around 2.2 seconds per ask and
decoding RPCs around 0.23 seconds under the four-worker stage. These are initial
observations, not a controlled before/after speedup estimate. Startup, later
history lengths and the 58-worker pool can change timings.

At the subsequent 58-worker check, averages across currently profiled tasks were
0.68 seconds for sampling, 0.77 seconds for decoding RPCs, 0.68 seconds for oracle
RPCs and 0.015 seconds per recovery transaction commit. The pool included newly
started tasks with shorter histories; these averages must not be compared as a
controlled speedup against the resumed long-history tasks. Mean decoding queue
time was 0.46 seconds under the expanded pool.

## Evidence and operation

- `validation.json`: deterministic inference, saved-state recovery and archive checks.
- `pool_validation.json`: service-by-service equivalence checks.
- `deployment.json`: service PIDs, commands, allocator caps and deployment hashes.
- `live_status.json`: workers, active tasks, memory events, timing summaries and export status.
- `check_runtime.py`: isolated component test; it does not modify production trials.
- `status_runtime.py`: read-only status checks, with the remote check sent over SSH stdin.
- Shared campaign `workers/tpe_runtime.json`: runtime hashes and CPU layout.
- Shared campaign `workers/tpe_profile.sqlite`: one profiling file for the campaign.

Refresh the status with:

```bash
/usr/bin/python3 -S /root/reports/peptide_tpe_inference_20261028/status_runtime.py
```

Completed TPE tasks continue to be stored as one verified `.tar.gz` archive per
task. Active tasks retain their recovery files until successful archival.
