# Peptide FSBO/RGPE wrapper verification

## Results

Both real end-to-end component checks passed. No full production evaluation
campaign was launched.

| Method | Component result | Production readiness |
| --- | --- | --- |
| FSBO | Passed: 50 + 1 proposals, 51 fresh scores, two target fits | Prepared and preflight passed |
| RGPE | Passed: 50 + 1 proposals, 50 fresh scores, one cached repeat | Waiting for the complete 800-source bank and its performance check |

FSBO's additional production-setting probe fitted all 151 observed peptides
with the original 10,000-update maximum and patience criterion. It stopped
after 9,130 updates in 83.5 seconds; a subsequent batch-50 proposal from 5,000
candidates also passed. This establishes the production adaptation path, not
the runtime of a complete 5,000-evaluation campaign.

RGPE's bank had one completed source model when the diagnostic was pinned.
Its ranking selected weights `[0, 1]` for the source and target. The independent
math check also explicitly combined the real source and target with nonzero
diagnostic weights and verified full covariance and target scaling. That check
did not change production ranking weights. An 800-source run remains untested.
At the latest preparation check, 7 of 800 source GPs had completed training.

Fresh encoding comparisons differed by at most `3.88e-7`; repeated sequences
retained exactly the same stored coordinates. Completed run data are contained
in one verified archive per run, with job locks stored separately.

The prepared campaign is
`/root/data/logs/peptide_transfer_test200_20261028_v1`.
Its [readiness manifest](/root/data/logs/peptide_transfer_test200_20261028_v1/readiness.json)
and [FSBO production preflight](/root/data/logs/peptide_transfer_test200_20261028_v1/fsbo/preflight_report.json)
record the launch gates. RGPE preparation writes an awaiting-training plan
instead of permitting a partial source bank in production.

The peptide evaluation wrappers use the trained FSBO checkpoint or the RGPE
source bank, frozen author-VAE posterior means, and BOLT/LOLBO's native Sobol
candidate generation and constrained batch Thompson sampler. This is the
approved batch-50 adaptation of original FSBO/RGPE, not BOLT's ten-history
deep-kernel baseline or original sequential FSBO EI.

Production keeps 100 revalidated initial peptides plus at most 5,000 proposals
and 5,000 fresh APEX sequence evaluations. Cached repeats consume proposals but
not fresh oracle calls. Similarity-infeasible proposals remain recorded and are
excluded from the feasible incumbent, matching the existing peptide protocol.
Alphabet, length and cysteine violations fail before scoring.

Offline sources are train800 only. FSBO trains on 640 source tasks and selects
its checkpoint on the remaining 160 train800 tasks; RGPE production requires
all 800 source GPs. Only the current target's observations update its models.
The VAE stays frozen, and each repeated peptide reuses its first stored encoding.

## Verification Scope

The real component checks use held-out `pep_235`, all 100 revalidated starting
peptides, 51 proposals in two rounds (50 + 1), 64 candidates initially, 25 FSBO
adaptation steps, and 32 RGPE ranking samples. FSBO uses the completed
100,000-update checkpoint. RGPE uses the completed real source models available
when the diagnostic recipe was pinned; this is not an 800-source production test.

Each test checks native batch 50, target refitting from 100 to 150 observations,
real VAE decoding and APEX scoring, model reloads, scored-pending recovery with
new scoring forbidden, identical results in a copied recovery database, budget
accounting and verified archive cleanup. An additional proposal-only check uses
5,000 candidates and batch 50 without new oracle evaluations.

GPU floating-point operations may produce small changes when batch/padding
shapes differ. Fresh encoding comparisons use `rtol=1e-5, atol=1e-5`; stored
encodings for repeats must match exactly. The report records the observed error.

RGPE target checkpoints save the learned kernel, normalization and training
data, rebuilding matrix factorizations on reload without hyperparameter
optimization. Source model payloads remain unchanged in their SQLite bank.

## Artifacts

- [FSBO execution log](fsbo_execution.log)
- [RGPE execution log](rgpe_execution.log)
- [FSBO check results](fsbo_report.json)
- [RGPE check results](rgpe_report.json)
- [Production-setting execution log](production_preflight_execution.log)
- [Combined component report](report.json)
- Real diagnostic runs and recovery copies are archived under
  `/root/data/logs/peptide_transfer_wrapper_check_20261028_v3`.
- [Independent real-bank covariance and LOOCV check](/root/reports/peptide_transfer_wrapper_check_20261028_v1/rgpe_math_report.json)

The earlier V1/V2 diagnostics identified a pause-boundary issue and an overly
strict batch-independent bitwise encoding assertion. Their evidence remains
separate; no production campaign, source GP or offline checkpoint was changed.

## Production Gates

Preparation does not launch a campaign. FSBO requires its verified completed
checkpoint and recipe-bound wrapper checks. RGPE additionally requires all 800
source GPs and a complete-bank performance check. The full 5,000-evaluation
target horizon is not covered by these bounded component tests.

Large exact target GPs and exact joint leave-one-out ranking can be expensive
late in a run. The merger avoids separate sampling factorizations and stacked
source covariance matrices; predictions from nonzero-weight source GPs and
source ranking remain necessary. It is not a guarantee against all OOM failures.
