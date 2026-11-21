# Twelve Query inference replicas — November 17, 2026

The user authorized one additional LLM inference server per GPU and a subsequent
ten-minute production throughput comparison. The pool was expanded from eight
to twelve ORPO replicas, three on each A800 80-GiB GPU. The existing sixteen VAE
services, twenty-one optimizer workers, twenty-one private PostgreSQL databases,
CPU evaluation assignments, and 600-second recovery watchdog were retained.

## Placement

| Replica | GPU | Service CPU | Port |
| --- | ---: | ---: | ---: |
| llm8 | 0 | 30 | 21812 |
| llm9 | 1 | 31 | 21813 |
| llm10 | 2 | 33 | 21814 |
| llm11 | 3 | 36 | 21815 |

These CPUs share the existing bounded service allocation. SQL/evaluation CPU
sets and the monitor's reserved CPU 38 were preserved.

## Memory admission

The first unregistered candidates used a 26-GiB allocator limit. Full 32,768-token
requests exceeded that per-process limit. No production traffic was routed to
those candidates; their health evidence is retained in
`initial_max_context_failure.json`.

Every replica now retains the original 30-GiB allocator limit and full 32,768
input-token / 1,024 output-token envelope. Each physical GPU has one shared flock
file. Requests with at most 16,384 input tokens use shared admission (at most
three resident replicas); longer requests take exclusive admission. The OS
releases admission locks when a process exits. Transient CUDA cache blocks are
released before admission is relinquished. Model identity, prompts, RNG state,
sampling parameters, task selection, budgets, FSBO behavior, and saved results
were preserved. This is a scheduling change only.

All twenty-four ORPO/BOLT replica configs contain the same policy, and both
broker configs persist all twelve replicas for automatic recovery. ORPO uses
its shared queue; BOLT keeps its existing task affinity and prompt-cache policy.

## Rollout and verification

Workers were paused at durable checkpoints. Both hosts had zero live optimizer
workers and zero active SQL calls before the old inference processes were
replaced. No active SQL was terminated. See `quiescence.json` and
`before_resume_probes.json`.

* All twelve replicas passed exact model/adapter identity and GPU-placement checks.
* All twelve passed 32,768-token training-only probes submitted concurrently;
  per-GPU admission serialized these long requests. Each used 26.14453125 GiB
  peak PyTorch reserved memory.
* All twelve passed simultaneous 16,384-token training-only probes, with three
  active requests per GPU. Each used 21.39453125 GiB peak reserved memory.
* The capacity checks executed no SQL and did not add benchmark measurements.
* Thirty-three admission, health, transport-recovery, and campaign-recovery
  tests passed, plus an explicit twelve-replica health-pool check.
* The 312 launch pins were verified. Changes to previously pinned files were
  restricted to inference scheduling and resource configs. See
  `resource_only_changes.json`; all optimization/data/model pins were preserved.

`before/` contains the previous resource configuration and exact hash-verified
inference-server source. `rollout.py` records the checkpointed rollout procedure;
do not rerun it against an already-expanded pool.

## Bounded production observation

`observe.py` records 21 samples at 30-second intervals over 600 seconds after
all twenty-one workers and all twenty-one databases have been checked. Its real
process owns this bounded observation; `monitor_state.json` is only its state
record. It does not create or replace the existing recovery schedule.

The preceding healthy eight-replica window completed 196 evaluations in
600.4289 seconds (1,175.16 evaluations/hour). This comes from the established
recovery observations, not a projected linear multiplier.

The final comparison is written to `production_rate_10min.json`, with real
evaluation counts, generation counts, queue/admission waits, prompt sizes, and
each replica's completed requests. Sequential production windows can differ in
SQL and prompt difficulty, so the result is a short-window observation rather
than a controlled causal estimate.

The initial ten-minute report completed. The user then extended the same window
to thirty minutes, and subsequently ended the extended observation after about
16.5 minutes to request a completion estimate. The bounded observer was stopped;
the campaign and existing recovery watchdog remained active. The measured
partial rate and ETA are in `partial_rate_and_eta.json`. There is no completed
thirty-minute result. See `monitor_state.json` and `final_health.json`.
