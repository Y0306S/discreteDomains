# OptFormer to QueryOpt Random Search Handover

Authorized by the user on October 14, 2026. This deployment borrows released
OptFormer CPU capacity on LLMOptimizer4 for the existing QueryOpt test200 random
search queue. Existing TPE workers continue independently.

## Execution

- `handoff.py` checks every 30 seconds after its live smoke gate passes.
- Each random worker reserves four distinct physical cores: one for the frozen
  VAE and sampler, three for its private PostgreSQL instance.
- At most eight workers can be admitted. ToSFiT keeps CPUs 0-11, and controller
  processes use CPU 39. SMT siblings of reserved cores are excluded.
- A completed OptFormer task retains its cores until its process and any pinned
  descendants exit, including its final fit. Interrupted unfinished tasks retain
  their reservations for recovery.
- The controller removes borrowed cores from OptFormer's protocol and restarts
  only its scheduler, adopting the existing evaluators. Optimizer code, model
  checkpoints, task budgets, scores and evaluation settings are preserved.
- Admission uses the existing cross-campaign RAM registry, with 32 GiB headroom.
  Each database uses about 9 GiB of tmpfs; runtime reserves another 4 GiB.

## Benchmark Policy

The original runner and author VAE remain unchanged. The deployment selects its
existing `--device cpu` option because LLMOptimizer4 has no GPU. Device, database
endpoint and package versions are recorded in each task's normal provenance.
Worker allocation records identify the machine and physical CPU reservation.

The workers share the original campaign's locked claims and output directories:

`/root/data/cys/logs/query_test200_baselines_20261012`

This retains the same test200 split, 49 Bao initial observations, 1,000 unique
additional recovered executable plans, seed 0, sequential sampling, 100-second
SQL timeout, author decoder recovery, and per-task SQLite evidence storage.
Duplicate recovered plans are rejected before evaluation. The existing TPE
workers can subsequently join the same random-search queue without double
claiming tasks.

## Validation and Operation

Six focused tests cover final-fit retention, interrupted-task reservations,
pending evaluations, released cores, SMT siblings, and insufficient CPU space.
The live gate performs 20 real SQL evaluations, pausing after ten and resuming
the saved checkpoint for the remaining ten. Every new worker must pass the
existing fleet's own smoke/reload verification before production.

Run commands on LLMOptimizer4 with
`/root/data/cys/envs/bolt_clean_venv/bin/python`:

- `handoff.py stage`: prepare and run the first smoke check only.
- `handoff.py enable`: validate that completed check and enable the handover.
- `handoff.py start`: recover the handover controller without duplicating it.
- `handoff.py status`: inspect worker process identities and current tasks.

`status.json` contains the controller heartbeat, active workers, waiting reason
and remaining OptFormer tasks. `controller_process.json` identifies the detached
controller. Per-worker logs and failure evidence remain in this directory.

`provision_postgres.py` created UID/GID 10000 and granted that UID search-only
access through `/root`; other users did not receive directory access. A verified
clean PostgreSQL backup supplies separate databases under
`/dev/shm/query_random_optformer_20261014`.
