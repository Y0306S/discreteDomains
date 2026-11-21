# QueryOpt Test200 Baselines

Three methods run the same 200 held-out CEB queries, seed 0. Each task imports
49 official Bao observations and has 1,000 additional proposal slots. LOLBO uses
batches of 10. TPE asks for one proposal and receives its result before asking
again. Random Search draws one candidate at a time until it has evaluated 1,000
distinct recovered query plans, excluding all Bao starts and its earlier plans.
SQL executions within each worker are sequential. LOLBO/TPE cache hits consume
a proposal slot and reuse their runtime. Random Search rejects such repeats
without consuming an evaluation slot, so it can require more database calls and
decoder draws than the other methods.

## Methods

- Random Search: Optuna RandomSampler samples 64 continuous latent coordinates
  uniformly. Bounds include [-4, 4] and all initial encoded observations. A SHA256
  identity of the recovered executable SQL, including join hints, detects
  duplicate plans even when their original token strings differ. Rejections
  have durable SQLite evidence and are retained in resumable sampler state.
- TPE: Optuna TPESampler, with its default independent per-coordinate density
  models, learns from observed runtimes in the same latent bounds. Each result
  updates TPE before its next proposal. It is not SQL-plan
  tree search or multivariate TPE.
- LOLBO variant: the released latent GP, TuRBO trust region and Thompson
  sampling implementation, using the frozen author VAE. The GP is updated;
  the VAE is not retrained. Record the variant as frozen_vae_latent_turbo.

All three decode through the same verified author VAE and SQL codec, prohibit
cross joins, and retain the established decoder retry/fallback behavior.
A cross-join decode is resampled at the same latent point up to 51 times;
exhaustion uses the historical Bao plan with the largest recorded runtime.
For Random Search that fallback is already a Bao start, so it is rejected as a
duplicate and a new vector is drawn. No duplicate is substituted to fill its
budget. After 10,000 consecutive duplicate draws for a slot, the task reports
an incomplete failure with its checkpoint retained. It does not claim 1,000
unique plans when the decoder has not supplied them.
Cache hits and decoder fallbacks are recorded explicitly. LOLBO retains its
existing duplicate-plan GP filter, with progress/trust-region counters updated
even for an entirely cached batch.

Random/TPE SQL calls have a 100-second timeout. LOLBO uses the released adaptive
timeout policy. Censored runtimes are passed numerically to Optuna; its TPE does
not model censoring. LOLBO uses the censored GP likelihood. These are existing
baseline settings and are not the separate ML campaign's 2,400-second policy.

## Allocation And Storage

The user's final scheduling instruction is strict full-fleet stages:
LOLBO on all 14 workers first, then TPE on all 14, then Random Search on all 14.
Each stage must complete all 200 queries and drain its workers before the next
stage starts. LLMOptimizer2 supplies six workers; LLMOptimizer3 supplies eight.
Each worker owns one optimizer core, three PostgreSQL cores, and a distinct
database endpoint. Both hosts reserve at most 40 CPU cores including existing
ML inference/evaluation/controller allocations. Existing BORA and its queued
methods retain their slots. GPUs decode plans and, for LOLBO, fit the GP.

Each method has a shared cross-host queue of 200 tasks with exclusive claims.
The pipeline controller is code/query_test200_pipeline_v2.py; its current phase
is recorded in pipeline.json. It requires fresh smoke/resume verification on
every worker for each method before that method enters production.
The active LOLBO workers retain the original runner and source hashes. Queued
TPE and Random Search use code/run_query_ceb_baselines_v2.py and the per-method
settings recorded in plan.json. The controller can adopt the running LOLBO
stage without relaunching its workers.
Results are in outputs/<task>/<method>/seed0/. Each task keeps one SQLite
proposal journal plus readable history, status, and the latest checkpoint.
The completed train800 campaign and its results are preserved. Its native
assets are shared through a symlink and checked against recorded hashes.

Smoke results live separately under smoke/. Production requires successful
pause/resume checks on every worker, with matching code and launch settings.
