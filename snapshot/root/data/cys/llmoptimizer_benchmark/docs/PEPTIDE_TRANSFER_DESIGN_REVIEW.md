# Design review: task similarity, good/bad regions, and explore/exploit for the peptide domain

**Purpose.** We have a working HPO/LLM stack for (a) retrieving similar tasks, (b) asserting
good/bad regions in a prompt, and (c) trading off explore vs exploit. This document asks
whether those three components port to the antimicrobial-peptide domain, and proposes what
to build. Every claim below is labelled **[measured]**, **[not supported]**, or
**[untested]**.

**Status.** Requesting review before implementation. Nothing in §4 has been built.

Prior art this builds on: `docs/task_similarity_research.md` (retrieval cascade),
`docs/dynamic_gates_research.md` (readiness gates, prefer/avoid regions, X50 explore/exploit).
Evidence base: a completed 1000-task single-task-BO reference run
(`stbo_init100_r100_bsz50_e2e10`; 100 init points, 100 rounds × 50 candidates, E2E VAE
finetune after 10 stalled rounds; 0 constraint violations, 0 invariant breaches).

---

## 1. The structural fact that governs everything

In HPO, all tasks share a configuration space. That is what makes "config X is good"
transferable, what makes Spearman-on-shared-configs a similarity measure, and what makes a
prefer/avoid interval meaningful across tasks.

**Peptides do not share a search space.** Each task is a ball of radius
`0.25 × len(seed)` in edit distance around *its own* seed sequence, and the benchmark
library was deliberately built so that held-out seeds are ≥25% different from every other
peptide in it.

**[measured]** Across 300 random task pairs, of task A's 100 feasible sequences, the number
also feasible for task B was **zero — in every pair**. The feasible sets are disjoint, not
merely weakly overlapping.

Direct consequences:

| HPO component | Portability |
|---|---|
| `space_hash` / exact-space filter | undefined — no shared space |
| Spearman / TST-R on shared configs (§2, "the definition of HPO similarity") | **undefined** |
| Copying a prefer/avoid region from a source task | **invalid** — a good sequence for seed A is infeasible for seed B by construction |
| Role-Jaccard on absolute coordinates | undefined |

Anything transferable must therefore be expressed in **seed-relative** coordinates. The rest
of this document is about which seed-relative coordinate to use.

---

## 2. Candidate coordinate 1: physicochemical descriptor deltas — **rejected**

The natural first choice, and the one the domain literature suggests: represent a candidate
by how it differs from its seed in aggregate physicochemistry — Δnet charge, Δhydrophobicity
(GRAVY), Δhydrophobic moment (amphipathicity), Δaromaticity, Δlength, Δcysteine count. This
space *is* shared across all 1000 tasks, which is what we need.

We tested it on all completed tasks.

**[measured] There is a real, systematic direction.**

| descriptor | % of tasks where it changed | sign consistency (of those) | mean change |
|---|---|---|---|
| **net charge** | 91% | **96%** | **+1.97** |
| cysteine count | 4% | 88% | −1.20 |
| length | 36% | 78% | +0.66 |
| GRAVY | 100% | 66% | −0.25 |
| hydrophobic moment | 100% | 64% | +0.05 |
| aromaticity | 65% | 65% | +0.03 |

BO reliably adds ~+2 net charge. Everything else is weak.

**[not supported] But the deltas do not explain outcome.**

| model | cross-validated R² on log₁₀(MIC) |
|---|---|
| **delta only** | **0.014** |
| seed only | 0.311 |
| seed + delta | 0.373 |

Aggregate physicochemical change explains **1.4%** of the variance in achieved MIC. The
*seed you started from* explains 31%. A prefer/avoid map over Δcharge bins would be binning
a coordinate that is essentially uncorrelated with the objective — our existing contrast
gate would correctly refuse to emit anything.

**[not supported] The retrieval premise looks real but is mostly geometry.** Seed distance
predicts delta distance (Spearman ρ = **+0.363**, p≈0; nearest-decile seed pairs have 94%
smaller delta distance than farthest-decile). That suggests retrieval should work. However,
**37-69% of the winning delta is predictable from the seed alone** (R² = 0.51 for Δcharge,
0.61 for ΔGRAVY, 0.69 for ΔCys; mean 0.37 across descriptors). A seed that is already highly
cationic has little room to add charge; a seed with no cysteines cannot remove any. So the
correlation reflects *similar constraints*, not a transferable recipe — retrieving a
neighbour's delta returns something derivable from your own seed.

**Conclusion.** Descriptor-delta space is not a viable coordinate for regions or retrieval.
Note also that the one genuinely transferable fact — "increase net charge, favour
hydrophobicity and amphipathicity" — is already a single sentence in the BOLT system prompt.
Descriptor-level transfer appears close to saturated by that sentence, which is a plausible
explanation for why the published method needs no explicit similarity machinery.

---

## 3. Candidate coordinate 2: edit-shell × position × amino acid — **recommended**

Because the space is discrete and countable, we can do something impossible in continuous
HPO: compute coverage **exactly**, as an absolute number rather than a proxy.

**[measured] Coverage of the 1-edit shell** (the `L × 19` single-substitution neighbourhood),
from a stratified 40-task sample:

| reference length | median 1-edit coverage | median 2-edit coverage |
|---|---|---|
| 10-12 | 62.9% | ~2.6% |
| 13-14 | 51.1% | ~2.6% |
| 15-17 | 26.7% | ~1.2% |
| 18-20 | 8.0% | ~0.5% |
| 21-25 | 11.7% | ~0.2% |

Caveat: only ~30-40% of distinct evaluated sequences are the same length as the reference,
so these figures cover the substitution sub-space only; true coverage of the full feasible
set (which includes indels) is lower.

**[measured] Task difficulty spans eight orders of magnitude.** The substitution-only
feasible ball ranges from 16,435 sequences (L=10) to 8.47×10¹² (L=25), median 241 million.
The ratio of a 5,000-call budget to ball size runs from ~25% at L=11 to ~6×10⁻¹⁰ at L=25.
A flat budget is therefore badly mismatched across tasks, and **task size is probably the
strongest single task descriptor available** — it is exactly computable from the seed length,
before any oracle call.

### Proposal

Use **edit distance × position × amino acid** as the region coordinate:

- **Good/bad regions:** a per-task `L × 19` table of (position, substituted amino acid) →
  observed effect on MIC. This yields directly assertable statements — *"at position 7,
  prefer R/K; avoid P"* — rather than an interval over an aggregate. With ~5,000 acquisition
  calls over ≤25 positions there is real data per cell.
- **Unexplored regions:** the same table, read as tried/untried. Exact, absolute, and
  cheap. The 1-edit shell is only 190-475 cells per task.
- **Cross-task aggregation:** align by position *role* (cationic / hydrophobic / aromatic /
  Cys) rather than absolute index, since lengths differ. This is the existing role-Jaccard
  idea applied one level down.

**Our existing readiness machinery ports unchanged.** The gates in
`_marginal_prefer_avoid_numeric` (≥4 observations, ≥2 per bin, ≥2 scored bins,
overlap→`neither`), the contrast gate (top-vs-bottom gap > 0.05·span), the stagnation gate,
and TuRBO trust length for adaptive width are all statements about *statistical readiness*,
not about hyperparameters. They apply directly to this table. We should also carry over the
fix already identified in `dynamic_gates_research.md`: rank-based `prefer_frac/avoid_frac`
always emits something, so gate on contrast so the system can say "no signal yet".

**[untested]** This coordinate has not been validated. Recommended first test in §5.

---

## 4. Explore/exploit and the top-5 / bottom-5 / recent-10 prompt block

The HPO pattern (best-5, worst-5, most-recent-10, plus a coverage map for exploration)
translates directly in form: sequences with their MIC, similarity-to-seed, and descriptor
deltas. Two issues must be settled first.

### 4a. Leakage — a correctness constraint, not a preference

The current SFT data builder enforces that **no real sequence and no achieved MIC appears in
the prompt**. That guarantee exists because the previous pipeline leaked the answers in
900/900 rows: the target sequences were written into the user message and the model was
trained to copy its own input.

Adding top-5 / bottom-5 / recent-10 reintroduces sequences and MICs into the prompt. That is
legitimate **only if the data is constructed causally** — context must be history strictly
*before* the target. Under the current `elite10` construction (targets = the finished run's
top-10 sequences), a top-5 block would contain the targets and recreate the exact bug.

This requires a **trajectory-imitation builder**: for task *t* and round *r*, context =
history up to *r*, target = what was found next. Different dataset, and it must be validated
with the same per-row assertions.

### 4b. It changes the method class

The published method is **initialization-only transfer**: the LLM sees only the seed, emits
an initialization set once, and a standard single-task BO loop runs unchanged on top. There
is no history in the prompt and no in-loop LLM call.

Putting trajectory history in the prompt makes the model an **in-loop proposer**, which is
the Optformer / LLAMBO design — a different method with different baselines and different
compute characteristics. Both are legitimate research directions. **They should not be run
as one arm**, and a result from the second should not be reported as a result about the
first.

**Recommendation:** treat these as two separate workstreams, and be explicit about which one
any given number belongs to.

---

## 5. Proposed next steps, cheapest first

| # | Step | Cost | Decides |
|---|---|---|---|
| 1 | **Exhaustive 1-edit enumeration, all 1000 tasks** | **75 s of oracle** | Establishes a baseline BO must beat. 321,328 sequences total. If 5,000 acquisition calls do not beat 75 seconds of enumeration, the latent-space machinery is not earning its keep. |
| 2 | Build the per-position substitution table from the existing 1000 runs; apply the current readiness/contrast gates | ~1 h, offline | Whether §3's coordinate carries signal where descriptors did not. Measure: does a held-out position/AA cell's predicted effect correlate with its realised effect? |
| 3 | Re-run coverage over **all 290 truncated tasks**, not the 20-task sample | ~1 h | Whether early stopping is genuine neighbourhood exhaustion or a decoder failure. See §6. |
| 4 | Task-difficulty model: predict achievable MIC from seed descriptors + ball size | minutes | Budget allocation and task selection. Seed-only R² is already 0.311. |
| 5 | Trajectory-imitation dataset + causal-context validation | days | Only if §4 is approved as a separate workstream. |

Steps 1-4 are offline, use no API budget, and follow the "offline replay first" rule already
adopted for the HPO retrieval work.

---

## 6. Open questions for review

1. **Is the in-loop proposer in scope at all?** §4 shows it is a different method from the
   published one. If we only want to reproduce and extend initialization-only transfer, the
   top-5/bottom-5/recent-10 block is out of scope and step 5 should be dropped.

2. **Should the acquisition budget scale with task size?** A flat 5,000 calls means ~25% of
   the entire feasible space for a 10-mer and ~6×10⁻¹⁰ for a 25-mer. A size-aware budget
   would be more defensible but breaks comparability with the published protocol.

3. **How should early-stopped tasks be reported?** 290/1000 stopped before the budget. An
   earlier hypothesis — that stopping on short references is genuine exhaustion while on long
   references it is a decoder defect — was supported by a 40-task sample but **contradicted
   at full scale**: truncation is *more* common on short references (40.6% at L=10-14) than
   long (16.5% at L=18-25), and long references achieve the best MICs. This is unresolved and
   is why step 3 exists. **We should not report a mechanism for early stopping until it is
   settled.**

4. **Is descriptor space worth one more attempt with a nonlinear model?** R²=0.014 was from a
   linear fit. A nonlinear model might do better, but the effect would have to be large to
   change the conclusion, and the seed-determinism result (37-69%) is independent of model
   class.

---

## 7. Summary for the reader in a hurry

- Peptide tasks have **disjoint search spaces** [measured], so the HPO definition of task
  similarity does not port and regions cannot be copied between tasks.
- **Physicochemical descriptor deltas fail** as a shared coordinate: systematic (+2 net
  charge, 96% consistent) but explain only 1.4% of outcome variance, and are largely
  determined by the seed anyway [measured]. Recommend abandoning this route.
- **Edit-shell × position × amino acid is the recommended coordinate** [untested]. It makes
  coverage exactly computable — something not possible in HPO — and our existing readiness
  and contrast gates apply to it unchanged.
- The **top-5/bottom-5/recent-10 block requires causal dataset construction** to avoid
  recreating a label-leakage bug, and it changes the method from initialization-only transfer
  to an in-loop proposer. Needs an explicit scope decision.
- **The cheapest useful action is step 1**: exhaustive 1-edit enumeration across all 1000
  tasks costs 75 seconds and gives us a baseline we currently do not have.
