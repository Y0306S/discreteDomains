Query-domain FSBO and RGPE training
=================================

This run creates new query-domain model parameters. It contains no Thompson
sampling, acquisition function, target optimization loop, or PostgreSQL calls.
The peptide weights and ongoing experiments are preserved.

Completed corpus: 32,790 observations from 788 eligible training queries.
Twelve train800 queries had fewer than two distinct uncensored canonical plans
and are listed in `corpus/manifest.json`. FSBO uses 631 training queries and
157 validation queries; RGPE fits all 788 sources. The test200 split is held out.

Source: `/root/data/cys/logs/query_train800_author_fixed_20261004`.
Use the authoritative train800 split only. The test200 holdout is excluded.
Require complete 1,049-observation histories with the pinned author VAE,
official Bao initialization, frozen VAE, and no unresolved database calls.
Exclude censored observations; retain the first uncensored measurement for
each distinct encoded plan. Tasks with fewer than two such plans are recorded
as excluded. The corpus manifest records counts and checksums.

Representation: re-encode plans using the frozen author query VAE's posterior
mean (64 dimensions). First use the exact oracle's alias codec to canonicalize
each plan; verify a decode/encode/decode round trip preserves the executed join
tree and operators, then deduplicate canonical plans. Pin each SQL checksum.
Do not use the proposal latent vector, because repaired
or fallback plans may differ from the proposal. Derive coordinate min/max
bounds from the training sources. Keep the per-task observed runtime in seconds.

FSBO: call the exact shared training loop used by peptide_transfer.py:
`examples/benchmark/domains/train_taskset_transfer.py::fsbo`.
Fresh parameters, hidden layers [128, 128], RBF kernel, no ARD, Adam 0.001,
batch size 50, per-task min/max response scaling, 100,000 updates, validation
every 100 updates. Deterministically partition source tasks into 80% training
and 20% validation using seed 123. Select the lowest validation predictive NLL
and verify it after reload. Negate runtime for the FSBO maximization convention.

RGPE: fit the same upstream SMAC source GP used for peptides: Matern-5/2 ARD,
normalized runtime, original optimizer and restarts. Reuse
`tools/peptide_rgpe_exact.py` for exact bounded-memory likelihood gradients.
Use all eligible train800 sources with all retained observations (no cap).
Verify finite posterior moments and identical predictions after pickle reload.
Store model parameters and provenance in `rgpe/models.sqlite`.
Ensemble weights depend on future target observations and are not fit here.

Resources: export on LLMOptimizer2 GPU 1, two CPU threads, 6% per-process GPU
memory cap. FSBO and 24 single-core RGPE workers run at nice 10 on
LLMOptimizer4; existing jobs are not stopped or reconfigured.

Status: `corpus/status.json`, `fsbo/status.json`, `rgpe/status.json`,
`rgpe/dispatch.json`, and `training_pipeline.json`.
The supervisor writes `final_validation.json` only after both full training
jobs succeed and the selected FSBO checkpoint and complete RGPE bank pass
their final checksum/provenance audit.
Best FSBO artifact: `fsbo/weights`; resumable state: `fsbo/training_state.pt`.
Individual RGPE failures are recorded in task logs and failure JSON files.
Re-running the same commands resumes only with identical corpus/settings.
