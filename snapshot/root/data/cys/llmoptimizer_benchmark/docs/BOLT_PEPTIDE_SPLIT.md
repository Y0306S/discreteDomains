# Pre-registered train/test split — BOLT peptide multi-task BO

**Fixed 2026-09-12, before any results from the corrected pipeline existed.**
Do not change this split after seeing results. Recorded so the STBO reference and the
later BOLT arm are comparable to Zeng et al., ICLR 2026.

## Split

| Role | References | Count | Use |
|---|---|---|---|
| **train** | `pep_000` … `pep_899` | 900 | STBO runs; their top-K solutions become the LLM fine-tuning corpus |
| **test** | `pep_900` … `pep_999` | 100 | held out; STBO baseline curve and BOLT evaluation |
| *reporting subset* | `pep_900` … `pep_919` | 20 | matches the 20-task subset in the paper's Table 11 |

This is the paper's split: *"We created a library of L = 1000 extinct peptides and held out
the last 100 as validation."*

## Why the boundary is not arbitrary

The paper states the validation peptides are "at least 25% different from any other peptide
in the library". Verified 2026-09-12 by computing, for every reference, its maximum
similarity `1 - lev(s_i, s_j)/len(s_i)` to any other library member:

| | max similarity to any other reference | refs with a >0.75-similar neighbour |
|---|---|---|
| train `000-899` | **0.960** | 174 / 900 |
| test `900-999` | **0.750** (never exceeds) | **0 / 100** |
| test refs with a >0.75-similar *training* ref | — | **0 / 100** |

So the held-out set is genuinely isolated: no test seed is more than 75% similar to
anything else, while 174 training seeds have near-duplicates (up to 0.960, e.g. `ref[116]`
vs `ref[166]` differ only by a leading residue). That isolation is what makes the transfer
result meaningful — a test seed 96% similar to a training seed would make LLM transfer
trivial. Re-picking the test set from `000-899`, or selecting a subset of `900-999` after
seeing scores, destroys this property.

## Metric

Predicted MIC vs *A. baumannii* ATCC 19606, **lower is better**, summed (and mean) over
test tasks. In our exported rows the maximize objective is stored, so

```
best_MIC = -best_feasible_objective
```

**Always report `best_feasible_objective`, not `best_objective`.** The latter ignores the
similarity constraint; reporting it is what let a fully infeasible sweep look healthy
(mean MIC 3.42 vs the published STBO 31.30 — 9.2x "better" than the baseline).
Additionally require `n_init_feasible == 1000` and
`best_feasible_similarity >= 0.75` on every row.

## Published reference points (paper Table 11, 20 validation tasks, summed MIC)

| k | STBO/MTBO | BOLT-600 |
|---|---|---|
| 1 | 5727.74 | 564.45 |
| 100 | 1551.49 | 100.58 |
| 200 | 1111.72 | 96.78 |
| 500 | 792.55 | 94.24 |
| 1000 | **625.95** | 92.10 |

`k` counts init candidates evaluated, so `STBO @ k=1000` is the best of the 1000 random
mutants. Our regenerated pools give **759.9 ± 67.3** over seeds 0-4 (refs 900-919), ~2σ
from the published value — acceptable given the paper does not state which 20 of the 100
validation peptides it used.

## Deviations from the paper (deliberate, state them when reporting)

| | paper | ours |
|---|---|---|
| acquisition batch size | 50 | **10** |
| post-init oracle budget | 20,000 | **1,000** (100 rounds × 10) |
| E2E VAE update frequency | every 10 steps | **every 33** |

Init pool size (1000) and the 0.75 similarity constraint match the paper.
