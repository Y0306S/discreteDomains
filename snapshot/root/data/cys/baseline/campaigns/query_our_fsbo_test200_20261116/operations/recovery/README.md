# Ten-minute monitoring and automatic recovery

The existing query-training-hourly-20261112 job now checks every 600 seconds. Its local monitor records all four historical/current campaigns; automatic mutation is restricted to the active Query BOLT → ORPO + FSBO campaign.

The recovery scheduler is an independent persistent Python process on LLMOptimizer4, pinned to CPU 61. It invokes recovery.py on LLMOptimizer2 using a dedicated SSH key restricted to that exact recovery command. The SSH host identity is pinned. The scheduler runs immediately, then on a monotonic 600-second cadence, skips missed slots, uses exclusive locks, and records each result in history.jsonl, latest.json and state.json. It sends no chat notifications.

The local read-only monitor also runs every 600 seconds. Recovery restores that monitor if its process disappears.

## Automatic actions

- Verify controller/worker process start times, both hosts, all private databases, active inference/VAE services and the remote inference transport.
- Preserve unknown errors and changed pinned code as a hold. Honor a healthy manual PAUSE. STOP or disabled config prevents restart.
- For known infrastructure failures, request a durable pause, let SQL finish under the existing timeout, and wait for optimizer exits.
- If generation remains blocked after five minutes, signal only verified campaign optimizers after confirming every database on that host is idle and PAUSE has been present for at least 290 seconds. Do not signal workers with active or unverified SQL.
- Restore existing PostgreSQL data directories, CPU allocations, VAE services, inference stage and forwarding. Verify original immutable dependencies.
- Back up any interrupted measurement journal. Authorize remeasurement of the exact saved read-only plan only after every old optimizer is gone and every database is ready with zero active SQL. Never invent a score or replay a completed measurement.
- Archive old errors, validate readiness, resume one controller, and verify 21 launch acknowledgements.
- Stop automatic recovery attempts after three failures without measurement progress; keep checking and recording the hold.

The current run's recipe, evaluation budget, timeout, model checkpoints, measured scores and optimization policy are unchanged.

## Evidence and controls

- state.json: actual remote scheduler process identity, cadence and next due time.
- latest.json / history.jsonl: each scheduled check and recovery outcome.
- local_latest.json / local_history.jsonl: each local recovery invocation.
- incidents/: original errors, quiescence evidence, journal backups and continuation records.
- verification.json: first scheduled check, restricted SSH round trip and live remote PID validation.
- config.json: enabled and interval_seconds. Creating STOP disables recovery and stops the remote scheduler at its next short sleep boundary; it does not stop the experiment or local read-only monitor.
- To re-enable after STOP, remove it and explicitly launch watchdog.py on LLMOptimizer4, first checking its flock/PID to avoid duplicates.

## Validation and limits

Sixteen recovery tests passed, covering healthy no-op behavior, known/unknown failures, manual pause, loading grace, changed code, journal preservation, active SQL exclusion, unrecorded workers, and generation-only drain guards. A controlled local-monitor stop was automatically repaired by the first remote scheduled check while all 21 experiment workers continued.

This is programmed recovery of known infrastructure failures, not unattended arbitrary source-code debugging. Unknown failures remain paused with evidence. A persistent process cannot run while its machine is stopped: loss of both environments, loss of the SSH endpoint/authorization, or a stopped remote scheduler requires an external restart. No platform-level restart API or system scheduler is available in these containers.

2026-11-17 recovery repair:
- A deep health timeout alone no longer drains a running controller with all 21 workers. This is recorded as degraded_health; bounded worker transport retries and durable errors still trigger recovery.
- Recovery records ownership and the inode of PAUSE. It can resume its own interrupted drain while preserving unowned manual pauses.
- The drain age uses the first observed continuous pause. SIGTERM handlers touching the same file do not reset the five-minute drain interval between hosts.
- All 21 recovery regression tests pass. Campaign code, recipe, policy, model pins, and scores were preserved. No uncertain measurement needed replay during this repair.
- Evidence: progress_repair_20261117.json and progress_repair_verified_20261117.json.
