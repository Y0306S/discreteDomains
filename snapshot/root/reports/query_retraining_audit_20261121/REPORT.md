# Query ORPO + FSBO retraining audit — 21 November 2026

The combined hybrid is performing well against historical standalone baselines, but **VAE/FSBO retraining has not demonstrated an optimization benefit over the previous hybrid**. New decoder tests do not support a catastrophic VAE collapse. The main unresolved concerns are broad latent-space acquisition, surrogate transfer quality, and the simultaneous reduction of the FSBO source corpus.

## Saved optimization results

At 200 new measured proposals, the 123 retrained-run queries that have reached this budget have geometric mean best-runtime ratios of **0.6927 versus frozen standalone FSBO** and **0.6875 versus SemanticOPT**: 30.7% and 31.3% lower measured runtime. These are partial, scheduling-dependent cohorts. The hybrids use BOLT30 starts, while those baselines use Bao49; storage, host, date and load differ. These comparisons do not establish a causal retraining benefit.

Only 21 queries have histories in both the original and retrained hybrids. On exactly matched completed-round prefixes, with identical BOLT30 observations, the retrained/original ratio is **1.00257** (0.26% higher runtime), task-bootstrap95% interval **0.97936–1.02608**. There are2 queries >5% better,3 >5% worse,16 within5%. At a fixed200 new proposals the ratio is0.99360, with an interval spanning equal performance. The original run stopped early; newer progress cannot extend this paired comparison.

Across136 retrained-run queries in the snapshot, FSBO supplied65 incumbent improvements versus635 from the LLM. Final/current best sources were FSBO24, LLM97 and BOLT15. FSBO filled82.44% of requested slots; acceptance among returned decodes was31.10%. These all-run counts mix query templates and budgets and cannot be directly compared with the older21-query rates as evidence of regression. The rates are after internal decoder retries. Source and censoring accounting is in `summary.json` and `comparison.json`.

## What changed

The previous FSBO used788 usable source tasks (631training/157validation),32,790 measured source plans. The retrained FSBO uses80tasks (64training/16validation),2,663 plans, of which1,734 are gradient-training plans. This was the configured80-task retraining recipe. It simultaneously changed the VAE, source-corpus coverage, embeddings, normalization and FSBO weights. Comparing the two campaigns cannot isolate the VAE.

The selected VAE is epoch8, validation loss0.035600 versus0.066137 initially. Teacher-forced exact-string accuracy increased from75.35% to92.90%. The selected FSBO weights are from step3,600, although training ran100,000updates. Its much worse final-step validation loss was not accidentally deployed: selection and verified reload use the best checkpoint. Existing GPU encoder/corpus equivalence checks pass for both the new FSBO representation and the separate original retrieval representation. No evidence of a coordinate mismatch was found. Both models remain frozen during query evaluation.

## New independent-seed decoder test

The diagnostic selects the first held-out query from each of15templates and its first4 saved BOLT plans, without inspecting runtimes. Each setting has240 stochastic draws/model, clustered within those15queries. It makes no SQL calls, changes no production configuration, and does not select a checkpoint. Both models use the same query/plan cohort, with each model's own source-coordinate spans for perturbations.

| Setting | Original valid non-crossjoin | Retrained valid non-crossjoin | Original novel vs BOLT30 | Retrained novel vs BOLT30 |
| --- | ---: | ---: | ---: | ---: |
| Encoded mean |219/240|239/240|10/240|2/240|
| Gaussian noise,5% source-coordinate span |138/240|201/240|52/240|58/240|
| Gaussian noise,20% span |22/240|32/240|20/240|28/240|
| Unranked native initial candidate geometry |23/240|28/240|20/240|25/240|

Reconstruction and small-neighborhood validity improve. Reconstruction at the mean mostly reproduces known plans. Both decoders struggle in the broad candidate geometry used by acquisition. This geometry test is before GP ranking, so its validity rate is not the final accepted-proposal rate or a measurement of query speed. The wrapper's initial raw-latent offsets reach±3.2 on about20/64coordinates; source minmax normalization scores candidates but does not constrain them to the data manifold. The retrained model's narrower source spans make this same absolute box relatively wider. The evidence supports a search-scale investigation, not an unvalidated production change.

## Controlled predictive comparison

A separate original-VAE FSBO is trained from scratch with exactly the retrained variant's80tasks,64/16split, measured source rows and labels,100,000-update budget, seed123, architecture, optimizer and checkpoint-selection procedure. It recomputes normalization on those same80tasks, matching the current policy. Source-payload hashes and exact runtime-label alignment were checked; retrained bounds match production. It runs on CPU cores freed by the completed peptide test on LLMOptimizer3 and is never promoted.

A paired transfer probe then uses the saved BOLT pools for15held-out queries spanning15templates. The first up to10 distinct uncensored plans provide support; remaining distinct uncensored plans provide evaluation. All model variants rank the same saved candidates with the same measured outcomes. Each fits the native target model with the production10,000-update maximum and patience16. This tests predictive ranking and selected-candidate regret without new SQL timing variability. It is not a BO convergence experiment and must not be used to tune checkpoints on these test tasks. All 45 adapted tests and 45 checkpoint-only posterior tests completed (15 queries × 3 models). The test queries are disjoint from the 80-task and 788-task source corpora. They span the same template families; this is not a template-held-out experiment. The original-VAE matched80 control selected step3,300, validation NLL0.19985, versus0.26093 for the retrained representation; these source validation numbers alone do not establish query optimization performance.

| Model | Mean ranking correlation before fitting | After native target fitting | Geometric mean selected/best candidate runtime before | After |
| --- | ---: | ---: | ---: | ---: |
| Original VAE, matched80 sources |0.133|0.145|2.350|1.676|
| Retrained VAE, same80 sources |0.171|0.151|1.684|1.725|
| Original VAE, previous788 sources |0.228|0.211|1.572|1.646|

Lower selected/best runtime is better;1.0 means selecting the fastest plan in the saved evaluation pool. Ranking correlation measures alignment with negative runtime, so higher is better. “Before fitting” still conditions the checkpoint GP posterior on the same support observations, with no gradient updates. “After” uses native target adaptation, up to10,000updates and patience16. This test uses posterior-mean selection from a saved pool; live acquisition uses posterior samples and decoding.

After native adaptation, retrained80/original80 selected runtime is **1.02919** (+2.92%), task-bootstrap95% interval **0.91688–1.16455**, with4 wins >5%,4 losses >5%,7 within5%. Against the previous788-source model it is **1.04808**, interval0.78378–1.38353. These are inconclusive differences on a small, single-seed diagnostic.

Before adaptation, retrained80 beats original80 in this saved-pool test (ratio0.71662, interval0.55898–0.92924); that advantage disappears after fitting. Adaptation changes retrained80 selected runtime by a geometric factor1.02433, with4 improvements >5% and9 regressions >5%; the aggregate effect is small and does not prove that adaptation is harmful generally. Original80 benefits substantially from adaptation in this diagnostic, while original788 changes little in aggregate. The result makes target-fitting generalization worth testing, rather than supporting a categorical claim that VAE retraining failed.

Predictive uncertainty also merits investigation: after fitting, predictive NLL is dominated by CEB_9B37 for all three models (approximately130,798 original80;108,754 retrained80;21,384 original788). This suggests poor uncertainty calibration on that saved candidate pool. Source validation loss and target training loss therefore cannot substitute for held-out ranking/uncertainty checks.

## Assessment and next experiment

There is no demonstrated query optimization gain from the current combined retraining, and no evidence of a catastrophic VAE collapse. Reconstruction improved; useful latent exploration and target prediction remain weak. The production comparison also confounds the VAE change with reducing788source tasks to80.

A further controlled BO experiment should predefine the same initializer, query cohort, proposal budget, runtime environment, and acquisition policy, and vary the representation with source coverage held fixed. Acquisition-range and target-adaptation ablations should use a separate development cohort, with frozen choices assessed on untouched test queries. The diagnostic has not changed, stopped, rolled back or promoted production models. It issued zero SQL calls. The existing query campaign remains active.

## Artifacts

`controlled_transfer.png/pdf/svg`, `controlled_results.json`, `diagnostics.png/pdf/svg`, `summary.json`, `comparison.json`, per-query CSVs, `decoder_probe.json`, `findings.json`, and reproducible analysis scripts are in this directory. The isolated source-control training and predictive-probe implementation/checkpoints are under `/root/data/cys/tmp/query_vae_control_20261121`; they do not replace campaign assets. Prior audit sources remain intact.
