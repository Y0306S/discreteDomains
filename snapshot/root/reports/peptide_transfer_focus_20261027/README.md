# Peptide Transfer Preparation

## TPE Issue And Pause

The user requested pausing test200 TPE and prioritizing frozen-VAE representations,
FSBO/RGPE training and batch-posterior preparation. Existing histories and optimizer
states are preserved. The method pause marker prevents the standard launcher,
CPU handoff and inference migrations from restarting TPE. Explicit instruction is
required to resume it.

Both hosts report zero remaining TPE worker/task processes and no forced kills.
All 29 active recovery databases passed SQLite quick checks; 56,375 acquired
observations/proposals remain recorded and no pending trial remains after draining.
The CPU handoff supervisors are stopped while transfer work is prioritized; their
updated code also respects the pause marker. Existing LOLBO and BOLT work continues.

At 23:15 CST, 29 TPE tasks were active and none had completed. Median progress was
1,825 of 5,000 proposals. Four sampled tasks used roughly 7-9 seconds per recent
proposal; none of their latest 200 proposals met the similarity constraint. This
sample is not a full-campaign feasibility estimate. Shared serialized inference,
batch-one calls, 256 independent parameter suggestions and durable per-round writes
contribute to latency; their separate costs have not been profiled. TPE remains a
frozen-VAE, batch-one method; changing its locality would create a separate variant.

## Frozen Representation

The complete source export reads only audited train800 archives. All distinct
task/sequence observations are encoded with the frozen author VAE posterior mean.
Original measured scores remain unchanged, and there are zero new oracle calls.
The original task-specific, updated-VAE coordinates are not pooled. Test200 is
excluded from offline source fitting. Target observations will use the same frozen
encoder and train800-only coordinate normalization during evaluation.

`frozen_encoding_check.json` verifies a 256-sequence exported batch against fresh
encodings on the original shared service and the now-idle GPU1 replica: both match
exactly, repeated encoding is identical, and seeded decoded sequences agree.
Encoding took 1.31 seconds versus 0.023 seconds in this single infrastructure probe;
this includes request contention and is not an end-to-end export speedup estimate.
The resumable queue uses `PEPTIDE_EXPORT_SERVER` to select the verified dedicated
replica while preserving the exporter, source hashes and original corpus identity.
The export resumed beyond all 246 previously committed tasks. Initial new-task
times on the dedicated replica were approximately 5-8 seconds per task; this is a
short throughput sample, not a promise for the remainder. The original single-task
transaction boundary safely discards any unfinished task and retains committed data.

## RGPE Merged Posterior

For independent component GP posteriors and weights held fixed during acquisition,
the weighted sum is a GP with mean `sum(w_t * mean_t)` and full covariance
`sum(w_t**2 * covariance_t)`. Batch Thompson sampling needs this full covariance,
including correlations between candidate points; merging marginal variances alone
does not suffice. Correlated components would additionally need cross-covariances.

The new merger streams each active source contribution into one accumulated
covariance and draws only once from the combined posterior. It preserves the
upstream RGPE ensemble's target output scaling and records bounded numerical
jitter. It is restricted to the upstream `average` variance mode. It avoids stacking
all source covariance matrices or factorizing each source posterior for sampling,
but still computes each contributing GP prediction. At 5,000 candidates, one
float64 covariance is about 191 MiB, with additional memory for source predictions,
accumulation, Cholesky factorization and fitted GP models. This is not an OOM
guarantee or a hundredfold speedup.

This is an isolated posterior component, not yet a complete peptide RGPE
evaluation adapter. Full source-bank batch runtime and constraint-aware selection
remain to be validated after training. FSBO likewise still needs its peptide
evaluation adapter; source training alone does not complete either baseline.

`rgpe_merge_check.json` records a passing component check using three small upstream
GP fits on a train800 task, with the full 256-coordinate input. Merged means and full
covariance match upstream RGPE, source posteriors remain unchanged, 20,000 sampled
draws reproduce covariance within 3.90% relative error, and mean sampling error is
at most 2.17 standard errors. Fifty correlated draws repeat exactly for a fixed seed;
invalid indefinite covariance is rejected. This check did not train production models,
use test200 training data or call the oracle. It did not benchmark 800 full source GPs
against 5,000 candidates.

## Training Status

The full export and training supervisor has resumed with the verified dedicated
GPU1 service (`http://127.0.0.1:18874`). After all 800 task exports finish, the existing
queue runs the 100-update FSBO update/reload check, then 100,000 full FSBO updates
and complete RGPE source fits. The source exporter and corrected per-task-minmax
FSBO training loop have not been modified. The supervised export routing change
does not invalidate the corpus or alter scores.
