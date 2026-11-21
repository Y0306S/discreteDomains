# Retrained Query FSBO: interim comparison

Snapshot: 18 November 2026, approximately 14:41 China time. This is an interim report of the running `query_our_fsbo_vae80_test200_20261117` campaign. Detailed timestamps and source evidence are in `summary.json`.

The new hybrid finds better query plans than standalone frozen FSBO and SemanticOPT on the available matched queries. It has **not yet demonstrated a clear additional benefit over our previous BOLT → ORPO+FSBO hybrid using the frozen author VAE**.

## Progress and training

At the report snapshot, the controller counted 21,575/160,000 saved measurements (13.5%): 6,000 warm-start measurements and 15,575 optimization measurements. All 200 BOLT30 warm-start pools were complete; 29 optimization tasks had started and five had completed. Snapshot journal reads occurred over a short interval and contain 15,540 measurements in completed ten-proposal rounds; controller progress also includes current partial batches and later commits.

The run uses ten proposals per round and 80 total rounds per task, including three warm-start rounds. RL is disabled in this campaign. Monitoring remains active every minute and known-failure recovery remains on its independent five-minute watchdog. The latest process identities, states and timestamps are in `/root/.codex/loops/query-training-hourly-20261112/state.json` and `latest.json`. The continuing-monitoring note is `/root/MONITORING.md`.

Training is complete:

- The VAE was fine-tuned from the author checkpoint on 64 training tasks, with 16 separate training-split tasks for validation and no overlap with the 200 test tasks. Early stopping completed at epoch 16 and selected epoch 8. Validation loss fell from 0.06614 to 0.03560; exact reconstructed-sequence accuracy increased from 75.35% to 92.90%.
- FSBO was trained on the resulting representation for 100,000 steps, with `resumed_from: 0`. Validation selected step 3,600. This FSBO training did not append updates to the previous FSBO model. The checkpoint was reloaded and verified.
- Optimization was restarted for the new representation; earlier optimization history was not reused. The exact existing BOLT30 measurements were reused.

Improved reconstruction validation establishes a representation-training improvement; it does not itself establish better held-out query optimization.

## Same 21 queries at 200 new proposals

This table compares geometric means of the best uncensored measured query runtimes. Each row compares the retrained hybrid with the named method on exactly the same 21 queries. Lower query runtime is better; these percentages do not describe inference throughput or total campaign wall time.

| Comparator | Retrained hybrid runtime relative to comparator | Runtime reduction | Queries better / worse / within 5% |
|---|---:|---:|---:|
| Previous BOLT → ORPO+FSBO hybrid, frozen author VAE and previous FSBO weights | 0.9936× | 0.6% | 4 / 2 / 15 |
| Standalone frozen FSBO | 0.8045× | 19.6% | 13 / 3 / 5 |
| SemanticOPT | 0.8555× | 14.4% | 12 / 4 / 5 |

For the previous-hybrid comparison, both methods use identical BOLT30 initial plans and saved runtimes. A descriptive task bootstrap gives a new/old runtime ratio interval of 0.9701–1.0160, spanning equal performance. The 0.6% difference is too small to establish that retraining helps.

The baseline protocols have different initialization: the hybrids use BOLT30, whereas standalone FSBO and SemanticOPT use the same Bao49. Consequently, 200 new proposals means 230 total measurements for a hybrid versus 249 for those baselines. A sensitivity comparison at exactly 230 total measurements per method (200 hybrid proposals versus 181 baseline proposals) gives 20.8% lower runtime than frozen FSBO and 14.7% lower than SemanticOPT on these same 21 queries. Initializer, host and storage differences still prevent a causal attribution to FSBO retraining.

There are 24 new-run tasks with at least 200 optimization proposals. On all 24, comparisons with frozen FSBO and SemanticOPT are respectively 21.6% and 17.2% lower runtime. Those estimates use a larger task cohort and should not be substituted into the common 21-query table as if all rows used the same queries.

## At the furthest shared stage of each hybrid task

The previous hybrid has 21 tasks with 28–40 completed optimization rounds each. Comparing each query at the furthest round completed by **both** runs yields a new/old runtime ratio of **1.0026×**, approximately **0.3% higher runtime** for the retrained run. Two queries are more than 5% better, three more than 5% worse, and 16 within 5%. The descriptive task-bootstrap ratio interval is 0.9794–1.0261.

These common prefixes were all measured before the new RAM migration. This avoids crediting post-migration RAM timings for an improvement at these matched endpoints, although historical host, load and stochastic proposal differences remain. The fixed-200 comparison is likewise a historical comparison rather than a controlled replay.

## What the FSBO arm contributes

On equal-round prefixes of those 21 tasks:

| Diagnostic | Retrained | Previous |
|---|---:|---:|
| FSBO requested-slot fill rate | 93.2% | 94.3% |
| Returned FSBO decode acceptance | 44.8% | 48.0% |
| Incumbent improvements from FSBO proposals | 3 | 3 |
| Incumbent improvements from LLM proposals | 100 | 110 |

The retrained model has not resolved the decoding/rejection bottleneck. The new run's best plan at these matched endpoints comes from the LLM on all 21 queries. FSBO sometimes improves the incumbent earlier, but later LLM proposals supersede it.

Across all 29 started new-run tasks and their completed rounds, five incumbent improvements came from FSBO and 129 from the LLM. Current best-plan sources are LLM on 25 tasks, FSBO on one, and the BOLT initializer on three. These counts include tasks at different budgets and describe behavior rather than a matched performance comparison.

## Limits and evidence

The current task subset is small and follows campaign scheduling order; it is not a random representative sample of all 200 test queries. Five tasks have finished. No final full-test ranking is justified yet.

RAM/cache placement changed at 20,604 total saved measurements, and changing storage can change measured runtimes. No post-migration speed difference should be attributed solely to model training. Source timings are historical, with different seeds, dates and hosts; task-bootstrap intervals do not account for these confounders or training-seed variation.

Only uncensored successful observations can improve an incumbent. SemanticOPT has 200 actual measured proposals per task; no flat extension or invented observations were used. Its historical run includes constrained random fallback proposals, so the comparison describes its implemented pipeline.

The report rereads and hashes raw frozen-FSBO histories for the included tasks, rereads SemanticOPT's measured SQLite histories and verifies both against the existing curve export. Hybrid journals are opened read-only. Models, scores, campaign settings and recovery policy are not changed by this analysis.

Artifacts:

- `comparison.png` and `comparison.pdf`: unsmoothed convergence on the common 21-query cohort. Runtime is relative to each query's Bao49 incumbent, using one common denominator across methods.
- `matched_200_proposals.csv`: per-query results at 200 new proposals, including the equal-total-budget sensitivity.
- `hybrid_equal_available_rounds.csv` and `hybrid_before_ram_change.csv`: comparisons at each task's shared stage.
- `summary.json`: training records, numerical results, diagnostics, limitations and source evidence.
- `compare.py`: reproducible read-only analysis. Its output directory is the only analysis destination; it does not change campaign runs.
