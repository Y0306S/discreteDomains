# QueryOpt On Machines 2 And 4

The user reserved LLMOptimizer3 for their own work on 2026-10-26. All remaining
test200 Random Search work belongs on LLMOptimizer2 and LLMOptimizer4.

| Machine | Workers | Worker CPU reservations |
| --- | ---: | ---: |
| LLMOptimizer2 | 9 | 36 physical cores |
| LLMOptimizer4 | 11 | 44 physical cores |
| LLMOptimizer3 | 0 | Reserved for the user |

Each worker retains one optimizer core and three private PostgreSQL cores.
Existing worker allocations are disjoint. The nine partial machine-3 tasks are
reassigned to the nine machine-2 GPU workers and given priority over their other
unfinished tasks. Those workers then drain their remaining claims and the
shared queue. Machine4 retains its CPU VAE execution.

The original 78 completed tasks and 94,342 committed proposals are retained.
All 29 unfinished task journals were checked, and their committed result hashes
were recorded in tasks/. Ten pending database calls were reviewed against the
dead workers, free task locks, zero clients on the original databases, and
absence of a committed result. They are explicitly allowed to replay the same
pending plan. Original attempts remain in the journal; extra physical database
calls are recorded separately from the 1,000 unique-proposal budget.

Machine3's nine private PostgreSQL instances were shut down cleanly. Their
reproducible tmpfs IMDb copies were removed after checking the clean restore
archive. Database logs and shutdown records are in retired/. BO results and
optimizer checkpoints are on shared storage and remain intact.

The original rolling plan no longer contains machine-3 workers. The previous
machine-4 handoff controller is disabled. The October23 expansion's source pins
are superseded by this deployment and prevent using that old launcher with the
changed plan. Use manage.py launch on machines 2/4; it refuses machine 3.
Do not start new benchmark or training jobs on machine 3 unless the user releases
that reservation.

On reassignment, only the checkpoint metadata's PostgreSQL port is updated.
The original checkpoint is retained in checkpoints/, and the migration is
recorded beside both the checkpoint and the task. The optimizer, RNG state,
initializations, pending plan, numerical runner, VAE, timeout and budget remain
the same. Six recent smoke records match the current sources and allocations.
Fourteen older records precede the October22 monitoring fix and are refreshed
with the same 20-proposal pause/reload gate. SQL reads and a real disk-spilling
sort are verified again on every destination database before use.

The campaign remains seed0, author VAE, 49 Bao initial observations followed by
1,000 unique recovered plans, and a 100-second SQL timeout. Private SQL spill
tablespaces use shared storage. The existing storage check requires 256 GiB and
300,000 free entries at startup; this is not a per-query disk quota guarantee.

manage.py actions: inventory, quiesce, release-machine3, prepare, launch, worker.
Inventory, original claims, source hashes, reassignment records, database health,
checkpoint backups, and per-worker state are retained here.

## Handover Verification

At 2026-10-26 11:32 China time, all 20 workers had passed the smoke/reload gate
and entered production: nine on LLMOptimizer2 and eleven on LLMOptimizer4.
Live process audits found no overlapping physical worker cores, worker errors,
or CPU throttling. All nine transferred tasks have checkpoint migration records.
All 16,342 previously committed proposals in the 29 partial tasks retained their
original hashes; 48 further proposals had been committed at this snapshot.
LLMOptimizer3 has no benchmark workers, private PostgreSQL instances, or GPU
compute processes and remains reserved for the user.
