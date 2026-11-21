# Peptide FSBO Training And Target-Refit Audit

## Training Checkpoint

The peptide checkpoint passed 20 independent checks. No checkpoint corruption,
untrained feature extractor, or held-out test-task leakage was found.

| Check | Result |
| --- | --- |
| Source tasks | 800: 640 meta-training, 160 validation |
| Meta-training optimizer updates | 100,000 |
| Selected checkpoint | Step 94,900 |
| Recomputed validation predictive NLL | -0.4642411083 |
| Untrained initialization validation NLL | 0.8195271034 |
| Validation tasks better than initialization | 160/160 |
| Training observations sampled | 2,547,880/3,242,361 |
| Trained GP and feature extractor | Both verified; all optimizer parameter states reached 100,000 updates |

All 800 corpus payload checksums, training implementation hashes, optimizer state,
source/validation separation, and train800/test200 separation were checked.
Validation uses the exact saved per-task minmax recipe and fixed disjoint
support/query observations. It is an offline predictive validation result, not
evidence that the constrained BO search necessarily finds better peptides.

Checkpoint: `/root/data/logs/peptide_transfer_train800_20261027_v2/fsbo/weights`.
SHA256: `74e8a6be92b579f79fc69b4bf2b35e2c7cbaf7728e294247bc353c871a305a94`.

## Target-Refit Findings And Fix

The old online stopping rule required 17 nearly equal consecutive losses. Above
800 observations GPyTorch's random likelihood probes could defeat this rule.
Before this fix, 1,061/1,127 recorded fits hit 10,000 updates, and the latest-fit
median was 4,565 seconds. This cap was per target adaptation before a batch of
50 proposals; it was not the BO evaluation budget or offline training budget.

The new policy:

- Jointly adapt the GP and feature extractor, with a 1,000-update ceiling.
- Stop after 16 likelihood checks without an improvement of at least 0.0001
  from the last meaningful best-loss anchor. Small improvements accumulate.
- Reuse the same seeded native likelihood randomness on each optimizer-loop
  evaluation, restoring the caller's RNG afterward. Above 800 points the
  likelihood remains a scalable numerical estimate, not dense Cholesky.
- Save parameters before the optimizer update and restore the exact best
  measured parameters. Verify the loss with gradients enabled, matching the
  training RBF numerical path.
- Record optimizer updates, optimizer-loop likelihood evaluations, best step,
  stop reason, settings, code hash and amendment hash. The restoration check
  performs one additional likelihood evaluation.

Dense Cholesky was investigated but rejected after its backward computation
measured about 0.93 seconds per update at 1,400 observations. The retained
seeded native computation measured about 0.24 seconds per update there.

## Verification

On real saved `pep_111` observations, two independent fits produced identical
weights and posterior samples. Both stopped after 443 updates and restored
step 427, with fitted NLL improving from -0.40656 to -0.78992.
They took 104.68 and 103.79 seconds. The same task's previous 1,350-observation
fit took 4,802 seconds and 10,000 updates; the new test used 1,400 observations.
This is an observed comparison between consecutive histories, not a controlled
whole-campaign speed estimate.

The 5,100-observation scale test passed with about 0.97 GiB peak RSS, below the
6 GiB worker reservation. It ran three updates and verified best-weight
restoration; it did not establish convergence at that size.

The end-to-end test passed real inference, two BO batches, 51 proposals,
saved-batch recovery without rescoring, reproducible resume, frozen encoding,
task isolation, archiving/accounting, and a 5,000-candidate/50-sample acquisition.
All 19 focused queue and amendment tests passed. Further tests verified a flat
loss stops after 16 updates and the final permitted update is measured.

## Recovery And Provenance

All 48 existing worker states were backed up at committed boundaries with no
pending evaluation batch. Their original recipe hashes and measured histories
are preserved. Before relaunch, each saved state blob was checked against its
maintenance checksum. The new policy affects subsequent target fits only.

Post-restart checks passed on both machines: 22 workers on LLMOptimizer2 and
26 on LLMOptimizer4, with no halted slots, replaced/dead worker identities,
incorrect affinities, or shared cores between independent workers. Five remote
tasks had already committed new fits with the correct amendment hash at the
check; local workers were still completing their first resumed fits. See
`local_live.json` and `remote_live.json` in the artifact directory.

Runtime amendment:
`/root/data/logs/peptide_transfer_test200_20261028_v1/fsbo/target_fit_amendment.json`.

Full artifacts, SQLite backups, and hash-verified original/amended source:
`/root/data/logs/peptide_fsbo_stopping_audit_20261029/`.

## ML Findings: Recorded Only

The preserved ML worker also has the adjacent-loss stopping weakness. It
additionally calls `optimizer.step()` before saving the weights associated
with the pre-update loss, so the selected weights and recorded best loss can
refer to different steps. Its default adaptation ceiling is 100, and normal
100-evaluation ML histories do not reach the 800-point stochastic threshold.

Evidence: `/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_fsbo_minmax_20261006/code/examples/benchmark/fsbo_worker.py`,
lines 170-198 and 219. No ML code or historical result files were changed,
as requested. This audit does not independently certify every ML checkpoint.

## Remaining Search Concern

The pre-fix peptide search audit found only 25 feasible fresh proposals among
51,624 fresh evaluations (0.0484%), and six tasks had improved. Its stored
scores, cache accounting, constraints and incumbents passed integrity checks.
This runtime fix does not resolve that low feasible-proposal rate or establish
an improvement in final BO quality.
