# Peptide TPE diagnosis — 2026-11-01

The completed TPE configuration has a severe feasibility failure. The flat convergence curve reflects the archived outcomes: it usually retains the common starting best because almost none of its new candidates pass the required similarity constraint. This is evidence about this specific global, independent latent-space TPE configuration.

## Full production-history audit

All 600 completed histories matched their SHA256 values from the earlier full archive audit. The TPE flags were independently recomputed using normalized Levenshtein distance for every one of its 1,020,000 initialization/proposal rows. The independent bit-parallel distance implementation passed 300 comparisons with a separate dynamic-programming implementation. No inference, APEX calls, training or production changes were made.

| Method | Fresh post-initialization evaluations | Feasible fresh evaluations | Feasible rate | Tasks improved |
|---|---:|---:|---:|---:|
| TPE | 888,995 | 171 | 0.0192% | 3/200 |
| FSBO | 887,925 | 7,025 | 0.7912% | 196/200 |
| LOLBO | 988,652 | 132,479 | 13.4000% | 167/200 |

Rates above count fresh logical oracle evaluations, including LOLBO recentering. Counts of all proposals include cached repeats and are a different denominator.

TPE generated 1,000,000 proposals. Only 307 passed similarity, including cached repeats; 303 were in proposals 1–1,000 per task and only four were in proposals 1,001–5,000. Fifty-four tasks generated no feasible post-initialization proposal, including cached candidates. Eighty-eight tasks had no freshly evaluated feasible proposal. Only three proposals beat their respective initial best, on three tasks.

## What the implementation does

The frozen recipe uses plain Optuna 4.9.0 TPE with one proposal per round, 256 independent continuous latent coordinates bounded by [-4,4], and the frozen author VAE. It has no local trust region. This matches the plain-TPE choice recorded in the existing baseline plan. In contrast, FSBO and LOLBO use local latent-space proposal mechanisms; LOLBO also adapts its VAE.

The high-dimensional independent proposal mechanism does not preserve the joint latent structure needed to decode close variants of a particular reference. Observed candidates overwhelmingly fail the 75% similarity requirement. Correct constraint handling can rank previous outcomes without guaranteeing that the next decoded candidate is feasible. The evidence supports a search-policy/representation mismatch as the main explanation; it does not independently establish that changing a particular sampler setting will solve it.

The frozen adapter maximizes negative MIC, initializes all 100 measured task-specific samples and supplies similarity violations to Optuna. Its current file hash matches the production recipe. Runtime transport wrappers leave sampling and scoring policy unchanged, and their prior checks verified deterministic inference/resume equivalence.

## Saved Optuna trial inspection

On pep_925, all 5,100 saved trial records have 256 in-bounds parameters, history-matching scores and consistent constraint signs. The saved sampler has multivariate=False. Reconstructing Optuna's actual default-gamma good set at completion selects 25 feasible initialization trials. None of its 5,000 proposed peptides was feasible.

Its initial mean latent L2 norm was 5.24, compared with 11.00 for its first 1,000 proposals and 10.05 for its last 1,000. This is supporting evidence of exploration away from the initialization distribution, rather than a proof about every task's decoder behavior.

A concrete example: reference KIQAVYRGVRVRRRIRRMH; proposal LIKILIKIKLKLKLKLKL had MIC 3.98795 but similarity 0.05263 (5.3%), below the required 75%. Rejecting this apparently good unconstrained score was correct.

The all-task audit independently checks recorded sequences and feasibility. The detailed SQLite/sampler reconstruction is a one-task spot check. No new decoder or oracle measurements were made.

## Recommended follow-up

Preserve this run as global independent TPE, and label its interpretation accordingly. Prepare a separate local/trust-region TPE pilot centered on feasible initialization latents, with the same initial pools, oracle, similarity rule and budgets. Compare multivariate sampling as a separately declared variant if needed. First verify meaningful feasibility and improvement on a small, predeclared training-only pilot before allocating another full test200 campaign. A pilot can diagnose the change without presenting test-driven tuning as an unbiased held-out comparison. This audit does not launch a new experiment.

Evidence: audit.json, per_task.csv, sqlite_sample_audit.json. Reproduction: python3 audit.py and python3 audit_sampler.py. The former depends on the prior convergence report's task_summary.csv for the expected history hashes.
