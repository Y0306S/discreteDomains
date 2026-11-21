# Final QueryOpt TPE task recovery

Task: `CEB_9A74`, worker `LLMOptimizer3.w05`, test200 split, seed 0.

The October 21 failure occurred when the supervisor queried a newly created SQLite
file before its `artifacts` table existed. Its exception handler terminated the
runner. The saved checkpoint retains 49 Bao observations, zero completed
proposals, and one pending proposal that maps to a cached Bao plan.

## Changes

- Initialize the results journal before spawning each evaluator process.
- Require the existing schema for any nonempty results database.
- Retain the last observed progress and record a warning for temporary SQLite
  lock or schema-creation errors; retry at the next monitor poll.
- Preserve fatal handling for unexpected database errors.
- Update only the rolling supervisor checksum. The campaign plan, TPE runner,
  author VAE, initializations, budget, timeout, and existing outcomes are unchanged.

Six startup/recovery regression tests and twelve existing storage/dispatcher tests
passed. PostgreSQL port 5445 passed a real table read and had no other clients.
CPU 24 is reserved for the optimizer; PostgreSQL uses CPUs 25-27. GPU 1 has room
for the query VAE alongside the existing ToSFiT actor.

Original code, manifest, checkpoint, and empty journal are preserved in `before/`.
`preflight.json` records hashes, task state and database checks; `launch.json`
records the new dispatcher identity. The dispatcher runs a fresh pause/resume
smoke test before admitting the production task. It keeps the existing automatic
handoff to Random Search after TPE finishes.

The other already-running dispatchers retain their loaded code until their next
restart. No other task process was stopped or restarted in this recovery.

## Verified production restart

At October 22, 17:05 China time, the smoke test had passed all 20 proposals,
including a checkpoint reload after proposal 10. Production `CEB_9A74` was running
with six saved proposals, all using cached scores at this early snapshot.
`verification.json` confirms the original 49 Bao rows and pending proposal were
preserved, the original pending proposal did not execute SQL again, both worker
processes were alive, and CPU affinity was correct. No monitoring error was present.
