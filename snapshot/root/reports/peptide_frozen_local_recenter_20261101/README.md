# Frozen peptide BO wrapper implementation

Implemented in `/root/data/cys/llmoptimizer_benchmark/tools/peptide_local_bo.py`
and `peptide_local_policy.py`; usage and method details are documented in
`/root/data/cys/llmoptimizer_benchmark/docs/peptide_local_bo.md`.

Prepared recipes (no optimization workers launched):
`/root/data/logs/peptide_frozen_local_recenter_20261101_v1/{fsbo,rgpe,tpe,gp,hebo,smac}`.

FSBO, RGPE and TPE share the actual train800 LOLBO benchmark settings:
100 initialization measurements, batch 50, 5,000 acquisition proposals,
5,000 shared fresh acquisition/recenter measurements, similarity 0.75, seed 1,
native local region and sparse perturbations, and recenter after 10
non-improving acquisition rounds using two passes of the recent batch plus
unique feasible top-k. VAE updates are disabled; the original author weights
stay frozen. Method-specific objective models/refitting are preserved.

The common wrapper associates target constraint feedback with actual decoder
input vectors, retains canonical frozen features for FSBO/RGPE transfer,
includes the latest batch when restarting a region, and durably recovers
pending proposals/scores, models, sampler/RNG and finite recenter progress.
TPE records the sparse distributions before asking, asks all 50 vectors before
feedback, and tells every scored vector with its measured constraint.

Validation artifacts:

- `bolt_settings_check.json`: static/AST inspection of the actual train800
  configuration against native BOLT defaults and author launch settings. It
  explicitly reports the original common benchmark budget/seed/initialization
  overrides. **LOLBO was not executed or tested.**
- `component_tests.log`: 17 behavioral/backend/recovery checks, including real
  target GP, HEBO/MACE and SMAC RF/EI components, TPE batch-50 handling, frozen
  VAE assertions, budget/phase accounting, archives, and fresh-process replay
  without rescoring. FSBO/RGPE bridge checks use fixtures; no new full source-bank
  optimization campaign was performed.
- `frozen_inference_check.json`: real train800 `pep_000` initialization encoding,
  50-vector local TPE decoding and 8 posterior-reconstruction examples through
  the existing inference service. VAE tensors were unchanged and gradients
  disabled. Zero new oracle calls. This is a functionality check, not a
  statistically meaningful pass-rate comparison.
- `recipe_verification.json`: dependency hashes and package checks for all six
  prepared recipes, train800 setting parity, frozen VAE, task scope, initialization
  pool and confirmation that no task runs were started.

Previous pinned recipes, completed histories, VAE weights and transfer source
banks were not modified. Train-scope transfer recipes flag source-task overlap;
they are training diagnostics. Use a separate `--scope test` preparation for
held-out test200 evaluation with these same optimizer settings.

No improvement in fresh feasible rate or feasible MIC is claimed until a paired
optimization experiment is run. No VAE training, locality tuning or LOLBO rerun
was included.
