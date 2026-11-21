# Dynamic gates: LLM call frequency, region insertion, and explore/exploit

Research notes + test plan for replacing three hard-coded constants in the
`ptllm_tpe` P5+ pipeline with principled, adaptive rules.

| Constant today | Where | What it decides |
| --- | --- | --- |
| `H2` = fixed Bernoulli p=0.7 | `examples/benchmark/hybrid_variants.json` | how often we call the LLM instead of TPE |
| `experience_self_start_trial` = 50 | `examples/benchmark/prompt_variants.json` | when this-run good/bad regions enter the prompt |
| `X50` = `experience_explore_prob=0.5` | same file, `fifty_fifty` mode | given regions are on: show prefer/avoid **or** coverage_map (never both) |

Baseline to beat in all cases: `ptllm_tpe__P5plusA5T50X50__H2`, seed 1, 80 trials,
40 slots in `benchmark_data/real_hpo_v2_mltest20_algo40/tasks_40.json`.
Never overwrite that run directory; every idea below gets a new prompt/hybrid id.

---

# Part A — LLM call frequency (replacing H2)

## Why fixed p=0.7 is the wrong shape

Two structural facts break the assumptions behind both `H2` (fixed Bernoulli) and
`HUCB` (plain UCB1):

1. **Non-stationarity.** LLINBO ([arXiv:2505.14756](https://arxiv.org/html/2505.14756v2))
   makes this its central claim: LLM proposals help most early and "systematically
   degrade as more data are collected and surrogate models strengthen." UCB1 assumes
   fixed reward distributions. As the adaptive-operator-selection literature puts it,
   "although UCB keeps exploring all arms, it would need quite some time to detect
   that the best operator has changed."
2. **Asymmetric cost.** A gpt-4o-mini call costs dollars and seconds; a TPE ask is
   free. No UCB variant knows this unless told.

## What we already have, and its lineage

`mindware/components/optimizers/hybrid_selector.py` is better grounded than the
docstring suggests:

- `mode="pucb"`: `Q + C2 · ω · sqrt(ΣN / (1 + N_a))` is the **PUCT** form (Rosin's
  predictor-UCB, later AlphaGo), with the ω schedule acting as the policy prior.
- `pucb_q="window_minmax"`: very close to Fialho's **extreme-value credit
  assignment** over a sliding window, which that line of work found *more stable*
  than average-based rewards.
- `pucb_q="breakthrough"` / `"arm_best"`: `(S+1)/(N+2)` is the mean of a Beta(1,1)
  posterior — i.e. we are already one line away from Thompson sampling.

## Candidate methods

| Method | Rule | Fit |
| --- | --- | --- |
| **LLINBO-Transient** | GP w.p. `p_t`, LLM w.p. `1-p_t`, with `1-p_t ∈ O(1/t)`, e.g. `p_t = 1 - 1/t²` | Only one with a no-regret bound for this exact setup: `R_T ≤ B·O(√T) + β_T·O(√(T·γ_T))` |
| **D-UCB / SW-UCB** ([Garivier & Moulines](https://doi.org/10.48550/arxiv.0805.3415)) | discounted/windowed mean: `X̄_t(γ,i) + 2B·sqrt(ξ·log n_t(γ) / N_t(γ,i))` | Matches the non-stationary lower bound up to a log factor; beats EXP3.S empirically |
| **DMAB** (da Costa/Fialho) | UCB1 + Page-Hinkley change test → **restart the bandit** | Built for "best operator changes mid-run" |
| **Adaptive Pursuit** ([Thierens 2005](https://gpbib.pmacs.upenn.edu/gecco2005/docs/p1539.pdf)) | `P_a* += β(P_max − P_a*)`, others `+= β(P_min − P_a)`, `P_max = 1 − (K−1)P_min` | Winner-take-most *probability*, not argmax; `P_min` floor means the LLM arm can recover |
| **Discounted Thompson** ([Raj & Kalyani](https://doi.org/10.48550/arxiv.1707.09727)) | `θ_k ~ Beta(S_k+α₀, F_k+β₀)`; `S_k ← γS_k + r`, discount **unplayed** arms too | Most sample-efficient at small T; no exploration constant to tune |
| **GP-Hedge** ([Hoffman et al. 2011](https://doi.org/10.48550/arxiv.1009.5419)) | Hedge portfolio; reward *every* arm each round via surrogate mean `r_t^i = μ_t(x_t^i)` | Converts partial → **full information**; large win at T=80 |
| **Cost-cooling** ([CArBO](https://doi.org/10.48550/arxiv.2003.10870)) | `EI(x)/c(x)^α`, `α = (τ − τ_k)/(τ − τ_init)` | Principled way to charge the LLM arm for tokens, fading as budget burns |

## Caution from our own results

LLINBO's schedule drives LLM usage toward zero because it assumes LLM decay. But
`X50/H2` — 70% LLM — is our rank-1 arm out of 15 in
`benchmark_data/real_hpo_v2_mltest20_algo40/analysis/p5plus/gpt_p5_p5plus_four_arm.csv`.
On this benchmark the LLM is still earning its slot late in the run.

**Therefore prefer the adaptive families over the scheduled ones.** D-UCB, Adaptive
Pursuit and discounted TS *learn* the direction per task; a hard decay-to-TPE
schedule would fight our best current result.

## Part A implementation order

1. **`HDTS` — discounted Thompson.** Sample `Beta(S+1, F+1)` instead of taking its
   mean; decay both arms by `γ≈0.95` every trial. No `C` to tune, which matters:
   at T=80 with 2 arms, `1.4·sqrt(log t / N_a)` is the same order as Q itself and
   likely dominates the decision.
2. **`HAP` — adaptive pursuit.** `P_min=0.1`, `β=0.3`, reward = existing
   extreme-value window Q. Degrades toward H2-like mixing instead of locking on.
3. **`HDTS_auc` / `HAP_auc` — rank-based credit.** Min-max Q is scale-sensitive and
   our score scales differ hugely across 30 datasets (bal_acc vs negated MSE).
   Fialho's later AUC/rank-based credit exists for exactly this reason. Q-swap only.
4. **`HHEDGE` — full-information Hedge.** Score the *rejected* arm's proposal with
   the per-algo GP from `mindware/components/experience/surrogate_landscape.py`,
   doubling effective samples per trial. Known flaw: GP-Hedge never forgets early
   rewards, so one bad first LLM call can suppress it all run;
   [No-PASt-BO](https://doi.org/10.48550/arxiv.1908.00361) fixes this with
   normalized rewards + forgetting.

Orthogonal and worth reading: LLINBO-Justify (reject LLM point if
`α_UCB(x_LLM) < max_x α_UCB(x) − ψ_t` with `ψ_t ∈ O(1/√t)`) and LLINBO-Constrained
(no tuning parameter at all). These change *what we do with the answer* rather than
*how often we ask*, so they don't save API cost — but they're more defensible than
probabilistic mixing.

---

# Part B — When to insert this-run good/bad regions (replacing trial 50)

## The problem with 50

Trial 50 is a **budget clock**: "we've spent most of 80 trials." What we actually
want is a **readiness clock**: "this run now has a map of good/bad regions worth
saying out loud."

`build_experience_self_block` (`mindware/components/optimizers/llm_prompt_blocks.py:984`)
already knows a lot about when a region is junk. For a numeric HP,
`_marginal_prefer_avoid_numeric` (same file, line 806) requires ≥4 observations,
≥2 samples per bin, ≥2 scored bins, and resolves overlap to `neither`. The trial-50
gate in `pt4opt_llm_optimizer.py:423` ignores all of that. A 5-HP CatBoost slot may
be ready at trial 28; a 13-HP `gradient_boosting` slot may still be mush at 50.

**A real flaw worth fixing at the same time:** `prefer_frac=0.3, avoid_frac=0.3`
selects prefer/avoid bins **by rank**, so once two bins are scored it *always* emits
something — even when the bin means are statistically indistinguishable. Rank-based
selection cannot say "no signal yet."

## Two things should become dynamic

- **(a) WHEN to start** — the readiness gate.
- **(b) HOW STRONGLY to assert** — interval width and prompt language, indexed by
  confidence, instead of one binary switch.

## Math families

### 1. Anytime-valid confidence sequences — the correct statistical frame

We would re-evaluate this gate at **every trial**. That is textbook peeking: a
fixed-horizon t-test applied repeatedly inflates Type I error. Confidence sequences
([Howard, Ramdas, McAuliffe, Sekhon, *Ann. Statist.* 2021](https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-2/Time-uniform-nonparametric-nonasymptotic-confidence-sequences/10.1214/20-AOS1991.full))
give `P(∀t: μ ∈ C_t) ≥ 1 − α` — coverage uniform over time and valid at *arbitrary
stopping times*, which is exactly our situation.

Gate: emit prefer/avoid for HP `h` only when the CS for
`(mean rel_score in prefer bins − mean rel_score in avoid bins)` excludes 0.

`rel_score` is already min-max normalized to [0,1] per run, so a bounded
empirical-Bernstein CS applies directly. Cheap approximation if we don't want the
full martingale machinery — spend δ by union bound over checks, as racing does:

```
width(n) = B · sqrt( log(2 · n_checks · m / δ) / (2n) )
```

### 2. Hoeffding / empirical-Bernstein races — replaces the rank heuristic

[Maron & Moore 1993](https://proceedings.neurips.cc/paper/1993/file/02a32ad2669e6fe298e607fe7cc0e1a0-Paper.pdf);
tighter [Bernstein races](https://faculty.wharton.upenn.edu/wp-content/uploads/2015/07/alt2013_final_2.pdf).

Treat each bin as a racing candidate:

- **prefer** = bins whose LCB exceeds the UCB of the worst bin
- **avoid** = bins whose UCB falls below the LCB of the best bin
- **neither** = everything else

The `neither` dead-zone then *emerges from evidence* instead of being imposed by
`OVERLAP_POLICY`. Bernstein is tighter than Hoeffding because it uses the observed
variance, which matters at n≈25.

### 3. fANOVA — per-HP gate

[Hutter, Hoos & Leyton-Brown, ICML 2014](https://proceedings.mlr.press/v32/hutter14.pdf):
decompose performance variance into per-HP components, importance `F_U = V_U / V`.
Fit a small random forest on this run's `(config, score)` pairs — trivial at n≤80.

Gate: only speak about HP `h` if `F_h ≥ 0.05–0.10`. This directly addresses
"13 HPs, 3 of them matter" on `gradient_boosting`. Their finding that "most
performance variation is attributable to just a few hyperparameters" is the whole
argument for this gate. Note we already compute an `importance_by_role` vector for
retrieval in `mindware/components/experience/retrieve.py`; fANOVA is the rigorous
version and the two could share one estimator.

### 4. TuRBO trust state — makes width dynamic, not just on/off

[Eriksson et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf).
Keep a trust length `L` per run:

```
success = new trial beats incumbent by > 1e-3 · |best|
τ_succ consecutive successes  → L ← min(2L, L_max)      # τ_succ = 3, L_max = 1.6
τ_fail  consecutive failures  → L ← L / 2               # τ_fail = ceil(max(4/b, d/b))
L < L_min (= 0.5^7)           → discard region, restart
```

Two gifts for us:

- The **width** of the prefer interval becomes adaptive rather than the raw bin
  envelope — a natural way to say "prefer near here, ± this much."
- `τ_fail` **scales with dimension `d`**, automatically demanding more evidence on
  13-HP GBM than on 5-HP CatBoost. That is precisely the per-task readiness we want,
  and it comes free.

It also gives a principled *un*-latch: when the region goes stale, stop asserting it.

### 5. πBO prior weight — and the twist that matters for us

[Hvarfner et al., ICLR 2022](https://arxiv.org/abs/2204.11051), building on
Souza et al. 2021 (BOPrO) and Lee et al. 2020:

```
α_π,n(x, D_n) = α(x, D_n) · π(x)^(β/n)
```

The prior's influence decays because "the model improves with data while the user
prior remains fixed," guaranteeing convergence even under a misleading prior.

**The twist:** our prefer/avoid block is *re-derived from this run's own data every
trial*. It does **not** go stale the way a fixed user prior does — it *sharpens*.
So the correct analogue is an **increasing or confidence-indexed exponent**, not a
decaying one:

```
γ_n = min(1, n / n₀)        or        γ_n = 1 − p_value(h)
```

expressed in the prompt as language strength (`hint` → `prefer` → `strongly prefer`)
or as interval width. This is a genuine adaptation rather than a copy, and it is the
"mathematical dynamic version" the fixed trial-50 switch is missing.

**Corollary for the RC/RS arms now running:** the *cross-task retrieved* block IS a
fixed external prior, so πBO's `π^(β/n)` decay applies to it literally. That
retroactively justifies the `experience_retrieve_fade` we disabled for this bake-off,
and makes "fade the retrieved block, sharpen the self block" a clean testable pair.

### 6. Page-Hinkley stability latch — unifies Part A and Part B

Reuse the DMAB change detector on the sequence of prefer-interval midpoints:

```
e_t = r_t − r̄_t + δ,   m_t = Σ e_i,   trigger when max_i |m_i| − |m_t| > γ
```

Latch ON when no change is detected for K consecutive LLM turns (K=3); recompute on
trigger. Same statistic governing both the selector and the prompt gate is a tidy
story for the paper. `δ = 0.15` is the value that line of work settled on.

### 7. Multiple testing

With `m` HPs × `B` bins we run many comparisons per trial. Benjamini–Hochberg:
sort `p_1..p_m`, take the largest `i` with `p_i ≤ i·q/m`, reject those (`q = 0.1`).
This controls how many spurious prefer/avoid lines land in the prompt. Alternatively
spend δ by union bound as in racing.

## Cheap fallbacks (if we don't want the full machinery)

- **Contrast gate** — turn on when the top-bin vs bottom-bin mean gap exceeds
  `delta` (already computed as `0.05 · span` in `build_experience_self_block`). If
  every config scores the same, there are no regions.
- **Stagnation gate** — turn on when the incumbent hasn't moved for
  `max(5, 0.15n)` trials (already computed as `stagnation`). Early, while scores are
  still climbing, P5's best/worst list is enough; the region map earns its tokens
  once the run is stuck.

Keep a **floor** of ~20–24 successful trials in all cases (matches the retrieval
rerank threshold), and a **cap** so a noisy slot doesn't wait forever:
`on = latched OR trial ≥ 50`.

Whatever gate we pick, X50's 50/50 explore/exploit split must key off **the same
latch**, not a separate number. Otherwise we start flipping to `coverage_map` while
prefer/avoid is still empty.

---

# Part C — Explore vs exploit inside the LLM prompt (replacing X50's coin flip)

## What X50 actually does

After `experience_self` is allowed, every LLM turn is an **XOR**:

- **exploit** → inject `build_experience_self_block` (this-run prefer/avoid)
- **explore** → inject `build_coverage_map_block` (untried cats, unobserved numeric range)
- never both; P5 top/worst/recent still always present

The pick is `_experience_explore_branch`: a deterministic Bernoulli with `p=0.5`
from `md5(seed, trial, mode)`. It does **not** look at stagnation, coverage,
incumbent beats, or whether the region block is empty.

That is the same shape of bug as H2 and T50: a budget-blind coin, not a
readiness or credit rule.

## Two sub-decisions (do not fuse them)

| Sub-decision | Today | Reuse from |
| --- | --- | --- |
| **(C1) WHEN** the XOR is allowed | same clock as T50 | **Part B latch.** Already required above. Before latch there is nothing to exploit, so a coverage XOR is just a second prompt block with no counterpart. |
| **(C2) HOW** to pick explore vs exploit on a latched LLM turn | Bernoulli 0.5 | **Part A selector**, new instance, arms `("exploit","explore")`. Same UCB / PUCB / DTS / HAP / DMAB code. |

We do **not** need a third mathematical family for the core problem. We **do**
need a couple of *signals* that Part A does not have, because the two prompt
operators are not LLM-vs-TPE: they are intensify-vs-diversify.

Do **not** share the LLM-vs-TPE `HybridSelector` instance. Different timescale
(only LLM turns after latch), different reward (a TPE trial must not credit a
prompt arm), different cost (both arms are an LLM call — unlike Part A, cost is
symmetric). Symmetric cost is why Part A's cost-cooling/CArBO does **not** apply
here and why a fair coin was a reasonable first try.

## Can we just reuse Part A/B? Verdict

**Yes for the machinery. No for the schedule.**

- Reuse: latch (C1), `HybridSelector` class (C2), Page-Hinkley restart, TuRBO
  success/fail counters, Bernstein “is the region real?”, fANOVA (which HPs
  coverage should talk about).
- Do **not** reuse: LLINBO-Transient / ε-decreasing / “explore → 0 as t grows.”
  That is the same trap as Part A. X50 50/50 *is* our rank-1 prompt. Late-run
  coverage is still earning its slot. A schedule that anneals explore away
  fights the best current result, just like decaying LLM probability fought H2.

The HPO literature's actual intensification/diversification switch is the
**opposite** of ε-greedy annealing: explore *more* when the incumbent is stuck,
exploit *more* right after a breakthrough.

## What the papers do in this exact slot

This is intensification vs diversification, not “which proposer to call.”

| Method | Rule | Papers | Maps onto X50 |
| --- | --- | --- | --- |
| **SMAC intensification** | After a new incumbent, keep racing similar configs (exploit); otherwise sample from the RF/EI (diversify) | [Hutter et al. LION 2011](https://doi.org/10.1007/978-3-642-25566-3_40); [SMAC3 JMLR 2022](https://www.jmlr.org/papers/volume23/21-0888/21-0888.pdf) | Force **exploit** for `K=2–3` LLM turns after a beat; otherwise allow explore |
| **irace** | Race elite configs; restart the elite when it is statistically unstable | [López-Ibáñez et al. 2016](https://doi.org/10.1016/j.orp.2016.09.002) | Force **explore** when Part B's Bernstein race says prefer/avoid collapsed to neither |
| **TuRBO length** | `τ_succ` consecutive beats → double L (exploit harder); `τ_fail` → halve L, eventually restart (explore) | [Eriksson et al. NeurIPS 2019](https://arxiv.org/abs/1910.01739) | Same counters as Part B TRdyTurbo, but the action is the XOR, not interval width |
| **IPOP / BIPOP-CMA-ES** | Restart with a larger population when stuck | [Auger & Hansen 2005](https://hal.inria.fr/inria-00287351) | Coverage_map is our “bigger population”: look at untried values |
| **GP-UCB / EI / PI** | `μ + βσ` (explore via σ); PI over-exploits; EI `ξ` knob | [Srinivas et al. 2010](https://las.inf.ethz.ch/files/srinivas10gaussian.pdf); Kushner PI; Mockus EI | We do not pick `x`; we pick a *prompt operator*. Analogue: exploit when region gap is large (high μ-signal), explore when coverage holes are large (high σ-signal) |
| **Adaptive operator selection** | Two prompt operators = two arms; UCB / DMAB / sliding-window TS | [Fialho/Da Costa AOS](https://www.msr-inria.fr/files/inria-00519579/file/banditaos-amai10.pdf); [DYTS on MOEA/D](https://arxiv.org/abs/2004.10874); [RIDER windowed TS for prompt ops](https://doi.org/10.23919/fruct70069.2026.11506558) | **Byte-for-byte Part A**, new instance. RIDER is the LLM-prompt version of this |
| **Entropy / MES** | Explore the query that most reduces entropy of the maximizer | Hennig & Schuler; Hernández-Lobato PES | Coverage_map is a cheap discrete stand-in: untried cats have infinite local entropy |
| **LLM-as-policy-mediator** | LLM *chooses* explore vs exploit from stagnation/uncertainty metrics, then a separate generator proposes `x` | [arXiv:2603.28959](https://arxiv.org/abs/2603.28959) | Warning: they find LLMs **over-diversify when progress slows**. So “ask the LLM whether to explore” is a bad idea; keep the XOR outside the model |

## Signals we already compute and do not use for X50

`llm_prompt_blocks.py` already has a deterministic explore/refine rule that X50
ignores:

```
explore = n < 15 or stagnation >= 8 or coverage < 40.0   # build_explore_refine_summary
```

plus `since_last_improvement`, `mean_coverage_pct`, and the Part B stagnation
threshold `max(5, 0.15n)`. That is enough to ship C2 without a new estimator.

| Signal | Already in code | Suggested use |
| --- | --- | --- |
| `since_last_improvement` | yes | `p_explore = clip(stag / max(5, 0.15n), 0.2, 0.8)` |
| `mean_coverage_pct` | yes | `p_explore = clip(1 - coverage/100, 0.2, 0.8)` |
| incumbent beat | traj / selector | SMAC: next K LLM turns = exploit |
| Bernstein neither | Part B race | irace: if no region survives, **must** explore (XOR becomes forced coverage) |
| fANOVA `F_h` | Part B | coverage lines only for important HPs with holes |
| region-stability latch | Part B | C1: XOR off until latched |

Floor 0.2 / cap 0.8 is the Part A lesson: never kill an arm. X50's 0.5 sits in
the middle of that interval, which is why a fair coin could win.

## Candidate C2 methods (one factor at a time, prompt id only, H2 fixed)

Reuse first; only the last two are new combinators.

| Id | Rule | New math? |
| --- | --- | --- |
| `P5plusA5TRdyX50` | C1 only: XOR starts at Part B latch; still p=0.5 | no — free with TRdy |
| `P5plusA5T50XStag` | C2: stagnation-adaptive p_explore, clock still 50 | no — Part B formula |
| `P5plusA5T50XCov` | C2: p_explore from `1 - mean_coverage` | no — existing coverage_map stats |
| `P5plusA5T50XInt` | C2: SMAC intensify — exploit for 3 LLM turns after a beat, else 50/50 | no — SMAC rule |
| `P5plusA5T50XRefine` | C2: hard switch using existing `n<15 or stag≥8 or cov<40` | no — already coded |
| `P5plusA5T50XHAP` | C2: new `HybridSelector` on (exploit, explore), HAP P_min=0.2 | no — Part A class |
| `P5plusA5T50XDTS` | C2: discounted Thompson on the two prompt arms | no — Part A class |
| `P5plusA5T50XUCB` | C2: UCB C=1.4 on window Q of incumbent beats | no — Part A class |

Do **not** launch an ε-decreasing `p_explore = 1/sqrt(t)` arm first. Same
rationale as skipping hard LLINBO decay in Part A.

## Implementation notes

- Keep XOR. Papers that put both prefer/avoid *and* coverage in one prompt
  (static `coverage_map` block when `ee_mode != fifty_fifty`) let the LLM
  ignore one; X50's never-both is the cleaner experiment.
- Reward for the prompt-arm selector: beat-incumbent on that LLM trial, same
  as Part A `pucb_q=breakthrough`. Window min-max is optional.
- Log `experience_explore_branch`, `p_explore`, `stagnation`, `coverage_pct`
  on every LLM traj row (C1/C2 replay needs this).
- Offline replay on finished X50 runs: we can reconstruct what p_explore each
  rule *would* have used, but we cannot know what config the LLM *would* have
  proposed. Live arms are required for C2. C1 (latch-coupled XOR) is the only
  one that is a true no-new-math change bundled with TRdy.

---

# Test plan

## Phase 0 — offline replay (zero API cost) ← do this first

prefer/avoid is computed **from the trajectory**, so every gate above can be
replayed on the *already-finished* `ptllm_tpe__P5plusA5T50X50__H2` runs with no new
LLM calls. For each of the 40 slots, record:

- the trial at which each candidate gate would have fired
- how many prefer/avoid lines each gate would emit, and on which HPs
- how often the trial-50 gate fires *before* evidence exists (false-confidence rate)
- how often it fires *long after* (wasted-signal rate)

Deliverable: `analysis/gate_replay/gate_firing_trials.csv`. Use it to pick **two**
finalists before spending any API budget. Only arms that change the *proposals* need
live runs.

## Phase 1 — live arms

One change at a time; all extend `P5plusA5T50X50`; seed 1, 80 trials, 40 slots, H2.

| New id | Change vs X50 | Question |
| --- | --- | --- |
| `P5plusA5TRdyX50` | evidence + stability latch, floor 24, cap 50 | Is a readiness clock better than trial 50? |
| `P5plusA5TRaceX50` | Bernstein-race prefer/avoid/neither, replacing rank fractions | Is evidence-based region selection better than top-30%? |
| `P5plusA5TRdyFanovaX50` | + fANOVA per-HP gate (`F_h ≥ 0.05`) | Does silencing unimportant HPs help? |
| `P5plusA5TRdyTurboX50` | + TuRBO width adaptation and staleness reset | Does adaptive width beat raw bin envelopes? |
| `P5plusA5TRdyStrengthX50` | + confidence-indexed language strength (πBO analogue) | Does hedged phrasing beat a binary switch? |

Part C (explore/exploit XOR). Hybrid stays H2. Do not combine with a new
Part A selector. `TRdyX50` already includes C1 (XOR keys off the latch).

| New id | Change vs X50 | Question |
| --- | --- | --- |
| `P5plusA5T50XStag` | p_explore from stagnation, still trial-50 clock | Does stuck → more coverage beat a fair coin? |
| `P5plusA5T50XCov` | p_explore from 1−coverage | Does hole-filling beat a fair coin? |
| `P5plusA5T50XInt` | SMAC: exploit 3 LLM turns after a beat | Does intensification-after-success beat a fair coin? |
| `P5plusA5T50XRefine` | hard switch: existing explore_refine_summary rule | Deterministic vs coin |
| `P5plusA5T50XHAP` | HAP on (exploit, explore), P_min=0.2 | Does Part A pursuit work on prompt operators? |

Selector arms from Part A (`HDTS`, `HAP`) are a **separate** axis — do not vary a
prompt gate, a prompt XOR, and a proposer selector in the same arm.

## Decision rule (pre-registered, same as X50R)

Unique-slot mean rank **and** head-to-head vs `X50`; adopt only if both win.
Unique slots must filter trajectories by algorithm — see
`examples/benchmark/report_x50r_unique_slots.py`.

## Cost

Each live arm is 40 slots × 80 trials against one API key. At `PARALLEL=4` the
cascade-vs-surrogate bake-off currently in flight gives the runtime datapoint to
extrapolate from. Do not launch anything here until it finishes.

## Instrumentation to add before Phase 1

Log per trial into the traj record so gates can be analyzed post-hoc:
latch state, `n` at latch, which HPs passed, CS width / p-value per HP,
count of prefer and avoid lines, Page-Hinkley triggers, TuRBO `L`.

---

# Reading list

- LLINBO — <https://arxiv.org/html/2505.14756v2>
- Garivier & Moulines, D-UCB / SW-UCB — <https://doi.org/10.48550/arxiv.0805.3415>
- da Costa, Fialho, Schoenauer, Sebag, DMAB — <https://www.msr-inria.fr/files/inria-00278542/file/pap333s1-dacosta.pdf>
- Fialho et al., extreme-value credit assignment — <https://www.msr-inria.fr/files/inria-00377401/file/aoslion.pdf>
- Thierens, adaptive pursuit — <https://gpbib.pmacs.upenn.edu/gecco2005/docs/p1539.pdf>
- Raj & Kalyani, discounted Thompson — <https://doi.org/10.48550/arxiv.1707.09727>
- Hoffman, Brochu, de Freitas, GP-Hedge — <https://doi.org/10.48550/arxiv.1009.5419>
- No-PASt-BO — <https://doi.org/10.48550/arxiv.1908.00361>
- Lee et al., CArBO cost-cooling — <https://doi.org/10.48550/arxiv.2003.10870>
- Howard et al., confidence sequences — <https://doi.org/10.1214/20-AOS1991>
- Maron & Moore, Hoeffding races — <https://proceedings.neurips.cc/paper/1993/file/02a32ad2669e6fe298e607fe7cc0e1a0-Paper.pdf>
- Hutter, Hoos & Leyton-Brown, fANOVA — <https://proceedings.mlr.press/v32/hutter14.pdf>
- Eriksson et al., TuRBO — <https://proceedings.neurips.cc/paper_files/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf>
- Hvarfner et al., πBO — <https://arxiv.org/abs/2204.11051>
- SLLMBO (closest sibling to `ptllm_tpe`; its LLM-TPE sampler is a fixed 50/50) — <https://arxiv.org/html/2410.20302v2>
- Hutter, Hoos, Leyton-Brown — SMAC intensification, LION 2011 / SMAC3 JMLR 2022
- López-Ibáñez et al. — irace, 2016
- Srinivas et al. — GP-UCB, 2010
- Fialho / Da Costa — bandit AOS, AMAI 2010
- RIDER — windowed Thompson over prompt operators, 2026
- LLM policy mediator for BO explore/exploit (over-diversifies when stuck) — <https://arxiv.org/abs/2603.28959>
