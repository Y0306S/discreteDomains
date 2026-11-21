# Peptide FSBO/RGPE code audit and worker reassignment

Audit date: 2026-10-28, Asia/Shanghai. Ten remote TPE workers were moved to exact
RGPE source fitting. The new queue has ten compact fits plus the existing legacy
fit; TPE retains 22 local and 26 remote workers. No peptide evaluations were
rerun or added by the audit.

## Findings

### P1: The source-GP gradient implementation prevented practical concurrency

The upstream [ARD Matern derivative](/root/data/cys/baseline/envs/bl_rgpe/lib/python3.7/site-packages/smac/epm/gp_kernels.py:473)
materializes an `N x N x D` tensor. With 5,100 observations and 256 coordinates,
one float64 tensor alone is about 49.6 GiB. Several simultaneous arrays explain
the first fit's measured 150.6 GiB peak and 8.7-hour duration. Ten legacy fits
would exceed the machine's 256 GiB memory limit.

**Fixed for new fits.** [peptide_rgpe_exact.py](/root/data/cys/llmoptimizer_benchmark/tools/peptide_rgpe_exact.py:28)
contracts the same exact marginal-likelihood derivatives through dense matrix
products without retaining an `N x N x D` array. For each coordinate, the
squared-difference contraction uses row/column sums and `B @ X`; it does not
approximate the covariance or discard observations.

The unchanged upstream model retains Matern-5/2 ARD, amplitude/noise priors,
hyperparameter bounds, response normalization, L-BFGS-B and ten random restarts.
The temporary objective override is removed before serialization, leaving
ordinary SMAC model pickles in the existing bank.

Six real-feature derivative comparisons produced identical likelihood values
and maximum gradient differences below `1e-14`. A small complete optimization
with all ten restarts matched upstream posterior predictions. The full-size
5,100-by-256 diagnostic peaked at 1.82 GiB and made no oracle calls.

The 26-second full-size diagnostic measures initial factorization plus one
likelihood/gradient evaluation, **not an entire source-GP training run**. New
full fits remain running; their eventual runtime and final serialization peak
must be measured before claiming a full-training speedup.

### P1: Full covariance was clipped and the upstream ranking sampler used the wrong factor orientation

[SMAC's prediction path](/root/data/cys/baseline/envs/bl_rgpe/lib/python3.7/site-packages/smac/epm/gaussian_process.py:287)
clips every covariance entry to a positive floor. Legitimate negative
off-diagonal posterior covariances must be preserved for correlated batch draws.
The previous merger called this path, so its earlier upstream-comparison test
could reproduce the upstream issue without detecting it.

[The upstream ranking sampler](/root/data/cys/baseline/repos/transfer-hpo-framework/rgpe/utils.py:107)
multiplies row-wise standard-normal draws by the lower Cholesky factor `L`.
Their covariance is then `L.T @ L`, rather than the intended `L @ L.T`.
Its jitter retry also keeps using a fixed jitter instead of the increased value.

**Fixed in the peptide helper.** [full_posterior](/root/data/cys/llmoptimizer_benchmark/tools/peptide_rgpe_posterior.py:8)
reads the underlying GP's signed covariance. The corrected
[ranking sampler](/root/data/cys/llmoptimizer_benchmark/tools/peptide_rgpe_posterior.py:69)
uses `L.T`, bounded numerical jitter and an explicit failure for invalid
covariances. Its independent test reproduced the clipping issue, preserved
negative correlations and matched covariance with 0.115% relative sampling
error. Batch-50 shape, deterministic seeds and negative-MIC score direction
also passed.

The vendor repository was left unchanged. The eventual peptide RGPE adapter
must call this corrected sampler instead of the upstream ranking sampler.
These helpers are not currently deployed in a peptide RGPE evaluation campaign.

### P1: Cached source predictions need an explicit scale contract

Source banks fit raw positive MIC with internal standardization; their ordinary
SMAC predictions are transformed back into each source task's MIC units.
The RGPE ensemble subsequently applies the current target's scale and offset.
Combining those raw source predictions directly would apply incompatible output
scales.

**The helper now supports the correct contract.** For these raw-label cached
banks, call `merge_posterior(..., source_output='standardized')`. It combines
unit-standardized source predictions, weights covariance by squared weights,
and applies the target's scale exactly once. An independent test verifies that
contract. The default `ensemble` mode remains suitable for the upstream
ensemble's already standardized training labels.

The cache-to-target adapter must explicitly choose the appropriate mode. Source
training itself has the correct MIC sign and normalization.

### P1: Peptide FSBO/RGPE BO evaluation adapters are still incomplete

The [FSBO wrapper](/root/data/cys/llmoptimizer_benchmark/tools/peptide_transfer.py:224)
and source-GP fitter produce training artifacts. They do not provide a complete
held-out peptide optimization campaign. The remaining work includes target-task
adaptation, RGPE ranking weights/LOOCV, cached-model loading, frozen-coordinate
inverse mapping, candidate decoding, similarity constraints and the agreed
100-initialization plus 5,000-proposal accounting.

There is no launched peptide FSBO/RGPE evaluation campaign. A full source-bank
performance test and an end-to-end batch-50 dry run remain required. The new
source-fitting workers do not make this gap complete.

### P2: FSBO is the original algorithm with an adapted recipe, not identical upstream defaults

The [current shared trainer](/root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains/train_taskset_transfer.py:96)
uses the original FSBO neural-feature/GP modules and task-wise marginal
likelihood training. It uses the previously selected recipe: `[128,128]`, RBF,
no ARD, Adam at 0.001, 50 observations per update and 100,000 updates. Fixed
support/query validation selects the minimum predictive NLL checkpoint.

The upstream defaults instead use four 32-unit layers, Matern, a 0.0001 learning
rate and cosine scheduling. Its `meta_train` epoch visits every training task;
100,000 upstream epochs therefore do not mean 100,000 gradient updates.
Reporting our run as a literal reproduction of those defaults would be wrong.

**No new correctness bug was found in the completed FSBO training artifact.**
Its current recipe is intentional and was not changed or retrained during this
audit. Paper/default fidelity remains a reporting distinction.

## FSBO artifact checks

`fsbo_validation.json` confirms:

- All 100,000 requested updates completed; selected checkpoint is step 94,900.
- 640 source tasks train the model; 160 train800 tasks select the checkpoint.
- Every training task was sampled; 2,547,880 distinct training observations
  were encountered across minibatches.
- The 200 held-out tasks are excluded from offline sources.
- Checkpoint hashes match, trainable tensors/Adam state are finite and the
  selected checkpoint's reload check passed.
- Infinite constraint-bound buffers are valid GP metadata, not nonfinite weights.

FSBO learns shared neural features and shared GP parameters; it does not fit
800 independent source GPs. Its backend is PyTorch and GPyTorch. RGPE uses the
upstream SMAC/scikit-learn GP backend; BoTorch supplies QMC sampling utilities.

## Data and score direction

The immutable source corpus contains all 800 training tasks and 4,053,113
distinct measured sequence/outcome observations, with 2,989-5,100 per task.
Features use the frozen author VAE's 256-coordinate posterior means and
train800-derived coordinate normalization. The audit made zero oracle calls.

FSBO uses historical `score = -MIC` and maximizes. RGPE source GPs fit
`MIC = -score` and minimize; the batch sampler negates draws when exposing
negative-MIC scores. Current-task observations remain allowed at evaluation.

## Worker reassignment

- Ten remote TPE workers on CPUs 36-45 stopped gracefully; all ten recovery
  databases passed SQLite integrity checks. No forced termination was used.
- TPE now runs 48 workers: 22 on LLMOptimizer2 and 26 on LLMOptimizer4.
- Ten new exact RGPE fits use remote CPUs 36-45. The existing fit on CPU 0
  continues and was adopted by the replacement queue.
- The queue excludes completed and active source tasks, checks source
  provenance, serializes bank commits and reserves memory for active fits'
  future growth. It keeps at least 16 GiB unallocated headroom.
- The first compact measurement gives a minimum 4 GiB reservation per worker;
  observed peaks can raise that reservation and reduce later concurrency.
- BOLT, model checkpoints, source histories and LLMOptimizer3 were unchanged.

At the initial production check, all eleven fits were active near 100% of one
CPU core each. The ten compact fits had measured peaks of approximately
1.6-1.8 GiB, with no OOM events. These are running-fit measurements, not final
training peaks. The compact queue replaces the old supervisor without killing
its separately launched source fit.

## Evidence

- `exact_validation.json`: upstream objective/gradient and full-size memory checks.
- `posterior_validation.json`: signed covariance, QMC factor orientation and scaling.
- `fsbo_validation.json`: completed training artifact and source split checks.
- `bank_validation.json`: saved source GP versus its complete normalized training data.
- Shared `workers/tpe_to_rgpe_reassignment.json`: stopped workers and recovery checks.
- Shared `workers/rgpe_runtime.json`: validation hashes, worker slots and memory policy.
- Shared transfer `remote_queue.json`: current source-fitting progress and reservations.
- Shared transfer `rgpe_cpu36.log` through `rgpe_cpu45.log`: exact-likelihood progress.

Source fitting is resumable at completed-model boundaries. Interrupting an active
optimizer would restart that source fit; it does not have a mid-L-BFGS checkpoint.
The existing long fit was therefore preserved.
