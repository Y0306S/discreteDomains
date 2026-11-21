# Query FSBO local expansion

The user authorized launch on 2026-11-07. The ten-proposal bounded production
batch passed artifact, history, checkpoint, timeout, and recipe validation with
zero unresolved database calls (see batch_validation.json). All four production
workers were launched at 11:39 China time; production_processes.json records
their identities. Final live verification is recorded in production_verification.json.
The former hold marker is archived as LAUNCH_HOLD.resumed_20261107.

Adds four query workers w01-w04 on LLMOptimizer2 (CPUs 76-87) to the existing
16-worker campaign on LLMOptimizer4. The separate TaskSet campaign remains disabled.

The frozen recipe, source weights, Bao initialization, held-out split, 50-second
plan limit, Thompson wrapper, and no-VAE-update/no-recentering policy are unchanged.
run_worker.py calls the verified original runner with operational endpoint records
from plan.json. It uses the same task and database locks and writes into the same
campaign so completed results and checkpoints are retained.

Admission preserves 24 CPU cores and 200 GiB for future ML ORPO plus 4 cores and
32 GiB headroom. These are admission budgets, not exclusive kernel reservations.
The four databases use private tmpfs copies and disk-backed temporary tablespaces.
No additional VAE GPU services or ORPO training are launched.

Launch stages: provision, real Bao-fit/query concurrency pilot, one bounded
production batch, batch artifact validation, then launch.py for all four workers.
The original recipe is checksum-checked at every worker start.

Live verification passed: all four workers hold distinct task locks and have
committed new database results; all 16 original workers are alive. The campaign
now has 20 CPU workers sharing 10 healthy Query VAE services. Each local worker
uses three CPU slots (12 total). Peptide TPE remains deferred.

At 14:47 China time, w02 was recovered from the interrupted temporary-file
operation with all 286 prior results preserved. The failed proposal was replayed
with both attempts retained and the original 50-second cap. Remote w17, which
had stopped after encountering that failed task in the queue, was also restarted.
production_verification.json confirms all 20 workers alive. No additional
databases or CPU allocations were created beyond the existing 20-worker plan.
