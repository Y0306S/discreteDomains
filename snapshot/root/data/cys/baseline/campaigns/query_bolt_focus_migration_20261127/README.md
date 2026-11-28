# BOLT-focused query migration — 27 November 2026

The migration is implemented and running. All durable experiment data and new
operational code are under this shared campaign directory. LLMOptimizer2 has not
been shut down.

## Current deployment

- LLMOptimizer3: 16 frozen-query-VAE services, one on every A16 GPU; no live
  SemanticOPT LLM replicas or BOLT SQL/LOLBO workers.
- LLMOptimizer4: 15 independent BOLT SQL/LOLBO workers and databases, with the
  controller and VAE forwarding running as detached host-local supervisors.
- LLMOptimizer5 (added 28 November, SSH gateway port 38783): another 15 independent
  SQL/LOLBO workers and private databases. Verified quota 64 CPU equivalents,
  memory 256 GiB. Its host-local supervisor and direct VAE tunnel are independent
  of Optimizer2. Total capacity is now 30 concurrent BOLT workers.
- SemanticOPT: all 20 workers paused; 73/200 tasks complete and 101,438 saved
  evaluations. All 20 LLM replicas and the broker were stopped after draining.
- Both CPU-only servers are now identified and included.

The 15-worker placement assigns 60 CPU affinities within the 64-CPU quota and
leaves four CPU equivalents of reserve. It was admitted in stages 12 → 14 → 15.
Optimizer5 independently assigns 60 affinities plus the same reserve and was
admitted in stages 12 → 15 after database and memory checks.
Initial samples showed no throttling or OOM events. This is an admitted capacity,
not a measured optimal throughput. Keep 16 VAEs unless sustained VAE queueing
and a controlled 24-service comparison justify expansion; 24 is the planned
decoder CPU admission ceiling on Optimizer3's 28-CPU quota.

`placement.json` records each database and CPU assignment. Two existing databases
remain on shared storage; the others use their existing RAM data directories.
Temporary tablespaces retain their existing shared-Lustre locations. Work memory,
SQL parallelism, the 50-second measurement limit and spill limits were not changed.
The historical temporary-storage constraint remains; six previously paused BOLT
tasks stay excluded until ordinary queued tasks finish.

## Preserved experiment state

The original scientific recipe, runner, pool, initialization observations, model
checkpoint and counted budgets are unchanged. BOLT still targets 800 evaluations
per query including its saved 30-plan warm start. The operational task wrapper
replaces all eight old VAE URLs with sixteen independent forwarded URLs on
Optimizer4 ports 24300–24315. It retains the runner's seed and ordering policy.

Consistent SQLite backups are in `snapshots/initial` and `snapshots/drained`.
`saved_state_audit.json` verifies all retained SemanticOPT record prefixes and
completed BOLT measurement records across 281 journals. No new ambiguous SQL
measurements were introduced. SemanticOPT's pre-existing deferred CEB_11A122
in-flight marker is preserved, unchanged, and must be audited before retry.
The stopped broker's requests were backed up to `snapshots/broker_requests.sqlite`.
Selected previously local convergence reports are in `retained_reports`.

## Independence and supervision

Optimizer4 connects directly to Optimizer3 through a restricted forwarding-only
SSH key stored on Optimizer4. Neither the key nor the tunnel depends on
Optimizer2. All 37 old campaign forwarding pairs through Optimizer2 were cancelled;
new BOLT measurements committed afterward. The original eight VAEs were adopted
by verified PID/start ticks; their detached sessions are independent of SSH.

The VAE and forwarding supervisors restart only their owned failed services.
A failed SQL task is paused independently and its healthy worker continues to
another task. A crashed worker process is isolated and is **not automatically
replayed**, because its child SQL call may be unresolved. Other workers continue.
Inspect its journal and database, resolve any unfinished SQL, then archive that
slot's `processes/sqlNN.json` before deliberately relaunching it.

To restart a supervisor, run locally on its designated host:

```
# Optimizer3
/root/data/cys/envs/bolt_clean_venv/bin/python /root/data/cys/baseline/campaigns/query_bolt_focus_migration_20261127/control_host.py launch LLMOptimizer3 --role vae
# Optimizer4
/root/data/cys/envs/bolt_clean_venv/bin/python /root/data/cys/baseline/campaigns/query_bolt_focus_migration_20261127/control_host.py launch LLMOptimizer4 --role forward
/root/data/cys/envs/bolt_clean_venv/bin/python /root/data/cys/baseline/campaigns/query_bolt_focus_migration_20261127/control_host.py launch LLMOptimizer4 --role bolt
```

PID/start-tick checks and host-local locks prevent duplicate supervisors. Status
is in `bolt.status.json`, `vae.status.json`, `forward.status.json`, and the canonical
BOLT campaign `status.json`. `verified_migration.json` records the live integrated
verification. A canonical `degraded` state currently includes the six historical
task pauses; consult `failures` for new worker failures.

## Later pausing or resuming

To pause BOLT, create its global `PAUSE` file and let every worker reach its full
saved-round boundary. Verify all workers are paused and no database has active
measurement SQL before reallocating resources. `STOP_bolt`, `STOP_vae`, and
`STOP_forward` stop the corresponding supervisor loop only; they intentionally
leave live children running. After draining, terminate owned children using their
PID/start-tick records before stopping the VAE/forwarding supervisors. Never
terminate an active SQL worker merely to accelerate a handoff.

For later SemanticOPT resumption, keep its saved proposal batches, RNG state,
broker requests and 800-evaluation amendment. First give it exclusive databases
and sufficient CPU/GPU capacity; BOLT currently owns the Optimizer4 databases.
If Optimizer2's container is destroyed, restore its static PostgreSQL clones from
the shared database source before using the original placement: `/dev/shm` is
ephemeral. Restore the LLM services, broker and networking on admitted hosts and
verify ownership before removing SemanticOPT's `PAUSE`. Do not rerun initialization
or restore drained backups over newer observations. Do not launch the legacy
BOLT controller: the new controller on Optimizer4 is authoritative.

Six topology/process tests passed, all fifteen databases passed ownership and
idle admission checks, all sixteen model hashes matched, and the integrated
encode/decode probe succeeded without fallback. Production continuation and
per-worker saved evaluation deltas are recorded in `verified_migration.json`.

## Second CPU host expansion — 28 November

Optimizer5 uses `placement_cpu5.json`, global worker slots 15–29, and private
PostgreSQL ports 20500–20514. Thirteen database data directories are in its new
RAM allocation; two are in dedicated shared-storage paths. All copies come from
the retired, cleanly stopped worker22 source, with each copied file checked by
SHA256. All tablespace symlinks were redirected to private per-worker scratch
paths. No source database or running Optimizer4 database was modified.

The expansion uses the original task order, warm observations, frozen model,
scientific recipe and task leases. `cpu5_runtime.py` is an independent host-local
supervisor; the canonical controller on Optimizer4 aggregates its heartbeat and
worker identities without treating remote PIDs as local processes. Losing that
heartbeat reports a failure and does not replay SQL. All 30 global task leases
must remain unique. Slot-process failure and task quarantine retain the same
independent behavior as the initial deployment.

On Optimizer5, restart its supervisor with:

```
nohup /root/data/cys/envs/bolt_clean_venv/bin/python -u /root/data/cys/baseline/campaigns/query_bolt_focus_migration_20261127/cpu5_runtime.py supervise > /root/data/cys/baseline/campaigns/query_bolt_focus_migration_20261127/logs/cpu5_supervisor.log 2>&1 < /dev/null &
```

The host-local lock prevents duplicate supervision. Existing processes are
adopted by PID/start ticks. Use `STOP_cpu5` only to stop supervision; it leaves
children running. Use the campaign's global PAUSE and drain before reallocating
its SQL resources. Its forwarding-only private SSH key stays on Optimizer5.

Expansion evidence is in `cpu5/databases_ready.json`, per-slot `copy_manifest.json`
and `ready.json`, admission records, and `cpu5/verified.json`. Eight topology and
process tests now cover both hosts. `gpu_usage_latest.json` and `samples/` contain
bounded live resource measurements; these are not a sustained throughput study.

## Authorized random feasible fallback — 28 November

The user authorized random feasible plan fallback only for `CEB_2B326` and
`CEB_2C202`, stalled at 730 and 670 saved evaluations respectively. Their original
proposal attempts exhausted 200 latent batches, with 9/10 and 2/10 accepted plans.
Those partial proposals were not persisted and no SQL measurement started in
either failed batch. The unfinished batches are therefore replaced with ten fresh
random feasible plans each. On subsequent rounds, the original latent proposal
strategy runs first; only an underfilled batch after the same 200-retry limit is
topped up randomly, retaining all accepted latent proposals.

`random_feasible_fallback.py` is installed by the operational task wrapper and is
scoped by `random_feasible_fallback_20261128.json`. It uses the existing native
random join-plan generator, strict codec validation, a no-cross-join check, and
canonical uniqueness against all prior observations and the current batch.
Random sampling is deterministic, bounded at 10,000 draws per batch, and
checkpointed. Accepted random plans are encoded with the frozen VAE. Decoder
substitution fallback remains rejected. Structural feasibility does not guarantee
successful execution: the ordinary SQL timeout, censoring and evaluation counting
are retained, as is the total 800-evaluation budget including the original warm
start.

This is an explicit method amendment for these two tasks. Random proposals and
measured observations are labeled `random_feasible_fallback`, with policy hashes
and per-round provenance, so comparisons can distinguish BOLT with random
fallback from unamended BOLT. The original runner and recipe files stay frozen;
all previous measurements and warm observations are preserved. The other six
paused tasks remain deferred.

`random_fallback_20261128/` holds consistent pre-change SQLite backups, original
pause records, measurement digests, staging checks and requeue records. The six
fallback tests passed, including native validation, uniqueness, bounded failure,
deterministic generation, retention of latent proposals and task scoping. Staging
verified twenty real random plans through the VAE with zero SQL calls and no
changes to existing measurement records.
