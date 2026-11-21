# Query FSBO readiness and bounded capacity pilot

Final state: **20 prepared slots, 16 running FSBO readiness workers on
LLMOptimizer4, and four parked slots on LLMOptimizer2**. All ten query VAE
services remain running on LLMOptimizer2. The user selected preparation and
the pilot only: no full optimization campaign, acquisition function, or
Thompson sampling wrapper is running. RGPE was not launched.

## Active workers

The active FSBO services are w00, w05, w06 and w07–w19. Each has an independent
PostgreSQL database and uses three CPU cores. The single-thread FSBO model
shares the first of its database's cores and waits during SQL execution.
The 16 database allocations use physical cores 0–47 without sibling overlap.
LLMOptimizer4 has a 64-core CPU quota and 256 GiB memory limit.

Fourteen database copies use `/dev/shm`. The two moved databases w05/w06 use
private files under this run's `databases/` directory. Those files were fully
read into the remote host's cache before measurement; `cache_warmup.json`
records the operation. These two workers depend on maintaining a warm cache
and enough remote memory. The container denied increasing `/dev/shm` from
128 to 192 GiB; its size was not changed.

All active services are reachable from both hosts on localhost, using ports
19600, 19605, 19606 and 19607–19619. Access requires the private token in
`service_token`; the token is not included in reports. The dedicated SSH
forward is recorded in `cpu_service_tunnel_process.json`.

`plan.json` records all slot configurations and the actual admission decision.
`active_allocation` is authoritative: 0 local and 16 remote. Local slots
w01–w04 have `admitted: false`. Their services and databases were stopped and
their private RAM database copies removed. Host process records are in
`host_LLMOptimizer2.json` and `host_LLMOptimizer4.json`.

These are task-bound readiness services for six training queries. Their only
authenticated operation is the bounded `/pilot` endpoint, which can fit on
known Bao observations and time known query plans. They do not dispatch
held-out query optimization jobs or generate acquisition proposals.

## FSBO implementation and warmup

The workers load the new query-specific source weights from:

`/root/data/logs/query_transfer_train800_20261106_v1/fsbo/weights`

Checkpoint SHA256:
`a7eddba4ba1de5b52904a6dcdf5589578643b6ae39774f475886c18882900e22`.

The shared upstream FSBO deep-kernel model and the peptide implementation's
`fsbo_target_refit.fit_target` are reused with query input dimension 64.
Each fit starts from the query meta-checkpoint, jointly adapts the feature
network and GP, uses target min/max scaling of negative runtime, and restores
the best measured likelihood. Adam uses learning rate 0.001, maximum 10,000
updates, patience 16 and minimum improvement 0.0001.

Warmup uses the selected query's 49 authoritative Bao imports. Censored
observations and repeated canonical plans are excluded from fitting. For the
six pilot queries, that leaves 1–3 distinct uncensored plans per query.
Features use the author query VAE posterior mean, canonical plan encoding,
and the query training corpus's saved coordinate normalization. All source
weights, training features, Bao data, and implementation paths are pinned in
`plan.json`. Only train-split queries were used in this capacity probe.

## Pilot result

`active_16_concurrency_pilot.json` is the final admission result:

- All 16 active workers loaded the query checkpoint, completed Bao warmup and
  produced finite posterior predictions.
- All 16 executed their known SQL plans concurrently: 48 SQL executions,
  including one discarded warmup and two measured repetitions per worker.
- Median measured runtime was 1.058 times each worker's isolated baseline;
  the largest ratio was 1.196. There were no active-worker errors or CPU
  throttling during this check.
- Three-core affinity and the shared optimizer core were verified.
- Remote memory headroom exceeded the 32 GiB admission reserve.
- Local memory headroom after parking was about 70 GiB.

The earlier 13-worker remote fit-and-query test also passed, including
simultaneous FSBO target fitting before SQL execution. Its median query
runtime ratio was 0.993, maximum 1.041, with no throttling. See
`pilot_LLMOptimizer4.json`. Additional moved-worker warmup and isolated
baselines are in `pilot_extra_w00.json`, `pilot_extra_w05.json` and
`pilot_extra_w06.json`.

## Why four local slots are parked

The initial local disk-backed database timed out on a known plan that
completed within a few seconds on RAM-backed workers. It was moved to the
remote host and passed after cache warmup. The disk-copy timeout is retained
in `failed_disk_local_pilot.json`.

Local background memory then increased to approximately 386.85/400 GiB.
The revised local pilot's query helper also exceeded its 90-second process
deadline. Continuing local admission would violate the 30 GiB reserve.
Its report is retained in `failed_memory_local_pilot.json`. All four remaining
local services were parked and their private databases removed to release
memory. Earlier failed evidence was preserved rather than counted as passed.

The current recommendation is **16 active query CPU workers plus 10 VAE
services**. Reconsider the four prepared local slots when the other local
workloads have released memory, then repeat admission and concurrency checks.
The proposed 24-worker expansion was not attempted.

This is a small capacity test using six known plans. It does not establish
full FSBO campaign performance, target-fit memory at large observation
counts, or sustained VAE throughput. A full campaign still needs its target
task queue and an agreed acquisition rule; Thompson sampling remains absent.

## Evidence and operation

- `active_pool.json`: active FSBO readiness endpoints, parked slots, VAE pool.
- `final_validation.json`: final process, affinity and checkpoint checks.
- `active_16_concurrency_pilot.json`: final measured result.
- `plan.json`: pinned source assets and slot configurations.
- `host_*.json`: service and PostgreSQL PIDs with their allocations.
- `workers/wXX/service.log`: model service startup logs.
- `workers/wXX/target_warmup.pt`: fitted pilot target state, separate from
  the source query meta-weights.

The active private databases and readiness services remain up. A full
campaign remains stopped as requested.
