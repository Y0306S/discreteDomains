# Peptide TPE warm-start audit

Checked 2026-11-06. Read-only inspection; no experiments or production changes.

The local BOLT paper (`baseline/baseline_papers/01_bolt.pdf`), page 6, explicitly specifies 1,000 randomly mutated initialization sequences per peptide, with at least 75% similarity to the seed. Page 7 confirms identical initialization-set sizes for held-out comparisons. This is sequence similarity, not predictive accuracy. Our experiment is a smaller-budget adaptation, not an exact reproduction of that protocol.

The completed test200 recipe specifies 100 initialization observations per task, 5,000 proposal/fresh-call limits, similarity 0.75, and global latent bounds [-4,4]. The initialization audit verified 20,000 distinct-within-task measured starting peptides and revalidated their scores. All passed the peptide and similarity rules.

`/root/data/cys/llmoptimizer_benchmark/tools/peptide_test200.py`, lines 214–236, encodes these observations and explicitly inserts each into the Optuna study with its latent parameters, score, and similarity constraint before the first `ask`. Thus the initialization was passed to the optimizer, not merely included in the plotted incumbent. The current adapter SHA256 equals the production recipe's pinned hash: `05bc94d16648e4af2b6ec954659efad56c08aea9ec5505d7efd7d4f4c409b5b4`.

The saved-sampler audit records Optuna's 10-trial startup threshold and independent (`multivariate=False`) TPE. The 100 completed initialization trials already exceed that threshold. No separate additional random startup is required for TPE to begin fitting. Constraints are attached to initialization and evaluated proposals; the saved callback is intentionally omitted during serialization and reattached on resume (lines 124–127 and 239–240).

The earlier full-history audit found only 171 similarity-feasible outcomes among 888,995 fresh TPE evaluations (0.0192%), and improvement on 3/200 tasks. That directly supports a feasibility failure. Independent sampling of 256 global latent coordinates does not ensure nearby decoded peptides, even when trained on feasible observations. This does not establish that every TPE implementation fails or quantify the isolated effect of 100 versus 1,000 initial points.

A 50-observation warm start is possible for a new reduced-budget experiment, but is smaller than the 100 already supplied and is not an established fix. A useful training-task pilot would hold initialization and budgets fixed while comparing global TPE with explicitly local latent proposals; separately compare 50/100/1,000 starting observations if initialization size is the question. Changing both at once confounds their effects. A new 50-start comparison must use the same 50 points across methods and cannot be substituted into the old 100-start plot as an equivalent run.

Evidence: `/root/reports/peptide_tpe_diagnosis_20261101/README.md`, `sqlite_sample_audit.json`, `/root/reports/peptide_train800_test200_audit_20261027/README.md`, production `tpe/recipe.json`, and the pinned adapter.
