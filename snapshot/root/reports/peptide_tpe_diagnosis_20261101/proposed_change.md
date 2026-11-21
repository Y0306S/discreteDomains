# Proposed peptide change: local multivariate TR-TPE

Status: proposal only. No production code, completed archives or experiments changed.

## Change

Create a separately named `TR-TPE` baseline using the existing peptide test200 adapter's initialization, decoder, oracle, constraints, recording and recovery logic. Replace its proposal policy with local multivariate TPE. Keep the completed global independent TPE results available under an explicit label.

1. **Local region:** center a bounded latent search region on the best measured similarity-feasible candidate. Initialize from the current task's measured pool, with the frozen author VAE. Specify the initial region size using training-only diagnostics, then freeze it before held-out evaluation. Retain the original absolute [-4,4] limits and all 256 coordinates.
2. **Joint proposals:** use Optuna multivariate TPE (`multivariate=True`) to sample a joint latent vector. This shares mixture components across coordinates; it is not a full-covariance model and does not guarantee decoder feasibility.
3. **Adaptation:** contract the region after sustained infeasibility or lack of feasible improvement; expand after sustained feasible improvement. On a restart, choose another distinct measured feasible candidate. Never promote an infeasible point to center. Initial radius, counters, restart policy and caps must be frozen in a versioned recipe after the training-only pilot.
4. **Stable coordinates:** retain every observation's absolute latent vector. If the region or normalized coordinate map changes, rebuild the local Optuna study from correctly mapped, in-region measured observations. Keep out-of-region measurements in the durable archive. Never reuse old normalized coordinates under a different center or radius. No scores are repeated during rebuilding.
5. **Visible sampling behavior:** record the region, center, absolute proposed vector, sampler mode and any restart per ask. Require actual joint local sampling; a global independent fallback must stop the pilot with a clear diagnostic rather than silently continue.

The existing legacy `TrustRegionTPESampler` is useful reference code, but needs an adapter: it expects `z_*` names rather than `z000`, updates a minimizing incumbent, and identifies acceptable trials via a large-value cutoff rather than this adapter's explicit similarity constraints. Its relative-search-space behavior must also be checked against the installed Optuna API. It is not approved as an unchanged drop-in replacement.

## Preserve the evaluation protocol

- Same task-specific 100 measured starting peptides and scores; paired arms share their initialization encodings and declared random seeds.
- Same frozen author VAE, APEX bacteria-0 objective, maximization of negative MIC, and similarity >= 0.75. TPE performs no GP/VAE training.
- Batch one; at most 5,000 acquisition proposals and 5,000 fresh post-initialization logical oracle evaluations per full run.
- Same treatment of cached duplicates and infeasible candidates. Do not add uncounted reject/resample attempts, similarity repair or skipped scoring as part of this comparison.
- Same durable scored-pending recovery, sampler/RNG state, accounting and checksummed archives. Include local-region state and restart history in resume checks.

## Validation before the pilot

Use isolated copies to verify objective/constraint signs, measured feasible-center selection, actual local bounds, joint sampler invocation, region contraction/expansion, restart and coordinate rebuilding. An Optuna API fallback must be detected explicitly. Test exact fresh-process resume across a region change and a scored pending event, with rescoring prohibited.

Use existing inference validation evidence plus small training-only decode checks to establish that the selected region is connected to decoder-feasible neighborhoods. A local box alone cannot guarantee valid peptide decodes. Log these diagnostic decodes separately; they are not production BO observations or free production resampling.

## Training-only paired pilot

- Preselect **8 of train800** using metadata (reference length and initialization statistics), without using held-out task results. Cover short, medium and long references. Persist the task list and selection rule before collecting new outcomes.
- Compare current global independent TPE with the proposed local multivariate TR-TPE on the same tasks, with **seeds 1 and 2** and **500 proposals per task/seed/arm**.
- This is **8,000 proposals per arm, 16,000 total**, with at most 16,000 fresh post-initialization oracle evaluations. Component/decode diagnostics have a separate explicit budget recorded before execution.
- Start with **4 single-core workers**, preferably two each on LLMOptimizer2 and LLMOptimizer4 after checking current reservations. Reuse an existing compatible frozen-VAE inference service if it has capacity; reserve no new training GPU.
- Compare fresh feasible rate, distinct new feasible candidates, improvement from the common starting best, duplicate rate, runtime and convergence. Report paired task/seed outcomes, not only a pooled score.

Proposed practical screening targets: at least **1% fresh feasible evaluations**, a fresh feasible candidate in at least **12/16 task/seed runs**, and improvement over initialization in at least **4/16 runs**. Also require better paired feasibility than the global arm. These are pilot gates, not guaranteed outcomes or a claim of statistical superiority. Failure triggers further training-only diagnosis rather than a full held-out launch.

The two-arm pilot evaluates the combined change. It does not isolate the separate contributions of locality and multivariate sampling; add a declared ablation only if that distinction becomes necessary.

## Full evaluation after the pilot

Freeze the selected implementation and all settings. Start a new test200 campaign with the original per-task budgets; retain the existing global TPE result as a separate method. Plot TR-TPE alongside global TPE, FSBO and LOLBO, including feasible-rate and cost summaries. The test200 outcomes already prompted this revision, so describe the revised experiment transparently as a follow-up on the same held-out set; a fresh untouched set would be needed for a completely independent generalization claim.
