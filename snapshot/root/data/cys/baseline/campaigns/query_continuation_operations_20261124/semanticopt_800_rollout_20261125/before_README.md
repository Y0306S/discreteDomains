# Query campaign recovery, 24 November 2026

The active supervisors are `fleet.py controller semanticopt` and `fleet.py controller bolt`, launched on LLMOptimizer2. Their process identities are in each campaign's `control/isolated_controller.process.json`. These replace the original `control/run.py` controllers; do not launch both implementations.

The SemanticOPT campaign uses 20 SQL slots and 20 LLM replicas, with 400 proposal evaluations per query and 800 for the first ten pilot queries. BOLT uses eight SQL slots and eight VAE replicas, continuing the 30 saved evaluations to 800 total per query. BOLT fits the saved implementation's GP and uses BoTorch acquisition on CPU, with the frozen VAE served on GPUs. Experiment recipes, optimizer implementation, model checkpoints, and measurement journals retain their original identity.

## Failure isolation

- Every BOLT query runs in a fresh Python interpreter. This prevents a second call to `torch.set_num_interop_threads(1)` in the same interpreter from failing at the next query.
- Each task and SQL slot retains the original exclusive file lock, preventing duplicate workers from measuring the same task.
- A child task failure creates `task_pauses/<task>.json`; that task is skipped until its journal is inspected. The SQL slot can take another query if its database is healthy.
- A slot failure or pause leaves other slots and both controllers running. The campaign status reports degradation without creating a campaign-wide PAUSE file.
- An explicit campaign `PAUSE` remains the operator's graceful stop mechanism. A `slot_pauses/workerNN.PAUSE` file holds only that slot between queries. Removing a slot pause allows that live slot to continue. Task pauses require journal inspection before removal, particularly for unfinished SQL measurements.
- A controller can adopt live workers from `control/workers.process.json` using PID and process start ticks. Controller reconnection during this recovery did not interrupt the already running query children.

## Recovery evidence

`semanticopt_before_resume.json` and `bolt_before_resume.json` record the pre-recovery measurement prefixes. No unresolved SQL measurements or occupied task/slot leases were found before launch. All 28 database directories and settings were checked in `preflight_LLMOptimizer*.json`. `resume_preparation.json` records validation of 162 SemanticOPT file pins and six BOLT pinned files; the runner also checks the frozen pool and journal recipe identity.

The previously edited SemanticOPT status helper was restored to its exact pinned content before launch. All new operational code is outside the pinned source files.

Initial remote SSH launches hit LLMOptimizer4's saturated authentication queue. Established SSH multiplex connections were reused to restore the remaining workers. Missing reverse forwards for broker port 23090 and VAE ports 23300–23307 were restored via that existing master connection. These are required for remote workers; verify them from the remote host, not only locally. `restored_remote4_forwards.json` records the command and successful return code.

Tasks that failed solely because these forwards were absent were inspected before release. No unfinished SQL measurements were found; their errors and pause markers were archived under `transport_failures/`. `transport_tasks_released.json` records the affected tasks. This operation did not delete or replay measured evaluations.

`verified_resume.json` confirms that all prior SemanticOPT record prefixes and completed BOLT measurement journal values still match their pre-recovery hashes, and both campaigns have fresh evaluations, all 20/8 slots alive, and no task pauses or slot failures at verification time.

Run `python3 -m unittest discover -s <this-directory> -p test_isolation.py -v` for the failure-isolation tests. They exercise a failed task followed by healthy queries, separate child process identities, and continued healthy-worker execution after another slot exits.

Do not rerun `resume.py` on an active campaign. It is the one-time recovery launch record. To inspect current results, read the campaigns' `status.json`, worker status files, per-query progress, and task pause markers. `verify_resume.py` is read-only with respect to experiment state and refreshes the verification evidence in this directory.

## CEB_10A32 decoder recovery

At 680 evaluations, proposal round 66 exhausted 200 latent batches (2,000 sampled vectors) with nine unique feasible plans and 1,991 rejected outputs. The incomplete batch never reached SQL execution. Its saved optimizer state contains 65 completed batches and no `measurement/66/*` entries.

`bolt_proposal_recovery_policy.json` explicitly amends only CEB_10A32 from round 66 onward. It raises the limit to 2,000 latent batches, retains the original BoTorch sampling and trust-region schedule, and continues rejecting duplicate and fallback outputs. `bolt_proposal_recovery.py` checkpoints the fitted surrogate, accepted/rejected proposals, and next retry index after every completed decoder batch. Resuming uses the same fitted surrogate and continues retry seeds; a fully assembled proposal batch is also reusable before the main journal commits it. Surrogate construction is seeded for this amended task as well as training. This changes its proposal sampling trajectory and is recorded in each affected batch's fit metadata; it is not represented as an identical rerun of the original proposal attempt.

Recovery evidence and all prior measurement hashes are stored under `decode_recovery_10A32/`. Slot 4 on LLMOptimizer3 was yielded at a verified boundary with no pending SQL measurements; its previous CEB_11A122 task remains resumable and eligible for subsequent scheduling. The fleet uses eight BOLT SQL slots throughout this recovery. The original recipe, model checkpoint, original runner file, and all previously measured SQL results remain intact.

## Deferred issues and pilot priority

At the user's request, CEB_10A32 (BOLT decoder novelty shortage) and CEB_11A122 (SemanticOPT temporary-file I/O failure) are deferred until all other tasks in their respective campaigns finish. Each campaign's `deferred_tasks.json` records this order; the task pause markers keep them out of ordinary scheduling. BOLT's active decoder task was yielded at a verified boundary with no unfinished SQL measurements, retaining its 700-evaluation state and partial proposal checkpoint. The deferred SemanticOPT task retains its journal for later I/O and interrupted-measurement inspection. These tasks require that inspection before release; no automatic measurement replay is authorized by the deferral.

Supervisors report `deferred_tasks` and `remaining_non_deferred_tasks`; once ordinary work finishes they report `awaiting_deferred_review`. They continue to preserve the original total of 200 tasks and do not count deferred tasks as complete.

SemanticOPT scheduling already places the ten 800-evaluation pilot queries before the other 190 queries (400 evaluations each). Leases allow one SQL worker per query. Once all eligible pilots are leased, other workers take ordinary queries so available SQL workers remain productive. Pilot priority is scheduling priority, not a barrier preventing all other queries from running concurrently.
