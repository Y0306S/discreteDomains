# Peptide Train800 Audit And Test200 Initialization Revalidation

## Result

All 800 completed V5 training archives pass the integrity and model-reload audit. All 20,000 held-out starting candidates pass current peptide rules and fresh outcome comparison. Original histories, references and recipes were preserved.

The train800 oracle spot-check has one tolerance failure, diagnosed as reproducible APEX batch-size numerics. Its original `passed:false` flag and fixed tolerance remain in the evidence; it has not been relabeled as a passing numerical comparison.

## Test200 Starting Peptides

- Exactly 100 distinct, measured starting sequences per task; 200 tasks total.
- Every sequence uses the 20-amino-acid alphabet, has length at most 30, contains at most one cysteine and has at least 75% Levenshtein similarity to its own reference.
- Freshly scored 20,199 unique sequences: the 20,000 starting candidates plus 199 valid reference candidates. No score differences exceed relative/absolute tolerance 0.0001.
- Maximum relative difference is 0.00000211565; maximum absolute difference is 0.000396733 MIC units. A comparison passes when either tolerance accommodates the difference.
- Derived pool: `test200_revalidated_initialization.json`; SHA256 `3f1bc840f8d70f53a553e85cc6d17c70da2f2241c6d29d24455a44dbe72320ef`. It retains original outcomes and revalidation provenance.
- Reference `pep_235`, `CYRAYICRKKFLSLRKASIKLQSL`, contains two cysteines. It remains valid task context, but is not itself a permitted evaluated candidate. All its 100 starting mutants pass. The task has a compatible neighborhood, so this does not block evaluation.

## Train800 Checks

The audit checks archive inventory/member hashes, pinned code/assets/environment, task/reference identity, initialization alignment, all sequence rules and similarity labels, negative-MIC score signs, duplicate/cache labels, proposal/fresh/recenter accounting, feasible final incumbents, GP/VAE array alignment and weights. It reloads each saved objective and constraint GP on CPU and checks finite posterior means and nonnegative finite variances at 16 recorded latent points. Every task's VAE update changed 183 tensors.

GPyTorch intentionally stores infinite constraint upper bounds. These are allowed constraint metadata, not nonfinite learned weights. The audit treats them accordingly.

| Measurement | Result |
| --- | ---: |
| Completed and integrity-passing tasks | 800 / 800 |
| Tasks improving their initial feasible best | 654 / 800 (81.75%) |
| Median per-task relative MIC improvement | 34.7656% |
| Median initial best MIC | 43.0759 |
| Median final best MIC | 22.7956 |
| Fresh post-initialization calls | 3,973,113 |
| Recenter fresh calls | 1,650,277 (41.5% of fresh calls) |
| Acquisition proposals | 2,833,731 |
| Cached repeat proposals, including recentering | 1,691,498 |
| New feasible distinct sequences | 499,793 |
| Distinct task/sequence measurements, including initialization | 4,053,113 |
| Tasks ending at fresh-call budget / proposal budget | 758 / 42 |

Similarity-infeasible probes are retained for constrained BO learning. About 12.6% of fresh observations are feasible, so exploration and recentering are costly. This is an efficiency limitation, not evidence that final recommendations violate rules: all final incumbents are feasible, and every task has feasible new observations. These are surrogate APEX MIC predictions, not experimental antimicrobial measurements.

## Oracle Numerical Investigation

Spot-checked initialization, best initialization, final incumbent and a fresh acquisition for every task: 3,045 recorded outcomes, 2,954 distinct sequences. This is a stratified spot-check, not a fresh rescore of all four million measurements.

One outcome exceeded the fixed relative tolerance of 0.0001:

`pep_278`, initialization index 88, original source row 278088, sequence `LEWPWKRLALGSIIARYL`.

| Raw uncached APEX batch | MIC |
| --- | ---: |
| Original stored value | 36.84697342 |
| One sequence | 36.85066986 |
| Four sequences | 36.84251404 |
| 16 repeated sequences | 36.84689713 |
| 64 sequences | 36.84698105 |
| 100 original pool sequences | 36.84696579 |

Each raw case was recomputed three times, bypassing the objective cache. Each batch composition reproduced its own value exactly. The four-sequence case exactly reproduces the spot-check difference: 0.00445938 MIC units, 0.0121024% relative. Larger batches closely reproduce the original outcome. Evidence supports numerical batch-size variation in the eight-model FP32 APEX ensemble; it does not indicate a wrong peptide, changed weights or corrupted archive. The diagnosis does not attribute the variation to a specific CUDA kernel.

Existing measurements remain unchanged. Baselines use the same pinned APEX predictor and report their actual call and batch protocol. Differences this small should not be interpreted as meaningful improvements between methods. No tolerance was relaxed to erase the failure.

## Evidence

- `train800_audit.json`: complete archive/model integrity checks and per-task results.
- `train800_spotcheck_report.json`: unchanged numerical comparison report, including the one failure.
- `apex_batch_diagnosis.json` and `diagnose_apex.py`: uncached numerical reproduction.
- `test200_pool_rules.json`, `test200_rescore_report.json`: complete starting-pool checks.
- `audit.py`, `rescore.py`: reproducible audit and fresh-outcome tools.
