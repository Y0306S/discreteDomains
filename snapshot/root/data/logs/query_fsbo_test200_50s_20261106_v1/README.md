# Query FSBO with 50-second query-plan cap

Launched 2026-11-06 on LLMOptimizer4 using 16 independent database workers
and the existing pool of 10 frozen query VAE inference services.

Uses the query-trained FSBO checkpoint, the 200 held-out query tasks, 49 imported
Bao observations per task, and 1,000 charged acquisition proposals per task
in batches of 10. Acquisition uses the peptide batch Thompson implementation.
VAE updates and stall-triggered VAE recentering are disabled. Ordinary TuRBO
trust-region updates remain enabled. Each new SQL plan execution is capped at
50 seconds. Historical imported Bao observations retain their original measurements.

`recipe.json` pins the implementation and model assets. `launch.json` records
commands, host, PIDs, process identities, and preflight health. `launch_validation.json`
is a read-only progress snapshot; per-task authoritative evidence is stored under
`runs/`. Worker output is under `logs/`. Run `validate_launch.py` on LLMOptimizer4
to refresh the launch snapshot. The snapshot is not a recurring monitor.

Related runtime-limit operation: `/root/reports/runtime_limits_20261106/`.
The current ML FSBO campaign policy is 600 seconds and remains paused.
TaskSet campaign settings are 600 seconds; the eight active ToSFiT workers
were safely restarted between evaluations and verified to advance afterward.
RGPE was not launched.

## Validated storage recovery, 2026-11-06

Seven workers encountered PostgreSQL SQLSTATE 53100 while query spill files and
parallel shared memory competed with 14 database copies on the 128 GiB tmpfs.
These infrastructure errors were retained as failed database attempts; they did
not become measured objectives or GP targets.

A private disk-backed temporary tablespace was tested on w07. The exact failed
CEB_11A44 proposal was replayed after verifying no database session remained and
that PostgreSQL had returned a definite ENOSPC error without a runtime result.
It reached the ordinary 50-second censored timeout and the pending batch finished.
All 26 pre-existing committed proposal records passed a checksum comparison.
See `storage_pilot_validation.json` for the gate used before fleet rollout.

The same database default was then applied to all 16 databases, taking effect on
new connections. Work memory, parallelism, query hints, the 50-second timeout,
FSBO checkpoint, acquisition, frozen VAE and recenter settings were retained.
The original recipe and all pinned implementation/model assets are unchanged.
Changing spill storage can affect measured runtime: `storage_configuration.json`
records the timing and scope of this operational change, and historical results
retain their original measurements.

All seven stopped workers were resumed. Their seven ENOSPC proposals were each
replayed exactly once: four completed and three returned normal censored timeouts.
The 186 committed proposal results preceding those failures were all unchanged.
Each replay remains the same single charged proposal; both database attempts,
including the infrastructure failure, remain visible in its audit trail.

`launch.json` is the original launch record. `current_workers.json` gives current
PIDs, and `storage_resume.json` records the seven replacements. The latest storage
recovery checks are in `recovery_validation.json`, including settings, process
identities, result preservation, replay outcomes and filesystem headroom. Refresh
them with `validate_storage_recovery.py` on LLMOptimizer4 using the query Python
environment. These scripts are one-shot operations, not recurring monitors.
