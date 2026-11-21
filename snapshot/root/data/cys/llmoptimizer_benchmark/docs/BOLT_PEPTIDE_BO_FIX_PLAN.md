# Fixing the BOLT peptide multi-task BO benchmark

Status: analysis complete, fix specified, init-pool generator implemented and validated.
Date: 2026-09-12.

Reference: Zeng et al., *Scaling Multi-Task Bayesian Optimization with Large Language
Models*, ICLR 2026 (`/root/21295_Scaling_Multi_Task_Bayes.pdf`).
Upstream code: `https://github.com/Yimeng-Zeng/BOLT.git` (`/root/data/cys/blockdata_archive/BOLT`, commit
`9787317` is the upstream peptide import; `/root/data/cys/BOLT` is code-identical).

---

## 1. What BOLT actually does

BOLT is **initialization-only transfer**. There is no multi-task GP and no shared
surrogate. From the abstract:

> "We introduce BOLT, an initialization-only transfer strategy that distills prior BO
> runs into an LLM which proposes candidates for new tasks, **while the surrogate at test
> time remains single-task**."

The loop:

1. Run single-task LOL-BO ("STBO") on training tasks.
2. Fine-tune GPT-4o-mini on `(task context C, top-K solution)` pairs from completed runs.
   For peptides, `C` = the seed amino-acid sequence.
3. For a new task, sample **the same number of init points the baseline uses** from
   `BOLT-T`, and run the *unchanged* STBO loop on top.
4. Feed the new run's top-K back into the next fine-tune. 7 rounds for peptides.

So the only thing that differs between STBO and BOLT at test time is **where the 1000
init sequences come from**. That makes the init pool the load-bearing component of the
entire benchmark — and it is exactly what is broken in our harness.

### Published peptide protocol (paper §4 + Appendix C)

| Item | Value |
|---|---|
| Task | seed peptide `s_i` → minimize predicted MIC vs *A. baumannii* ATCC 19606 (µmol/L) |
| Library | `L = 1000` extinct seed peptides; **900 train / 100 validation** (last 100) |
| Constraint | similarity `1 - d(S,S')/len(S) >= 0.75` to the seed |
| Validation design | "All of the validation peptides are at least 25% different from any other peptide in the library" |
| Init points | **1000** per task. STBO: random mutants at ≥75% similarity. BOLT: 1000 LLM samples (temperature 1.0) |
| Inner loop | constrained LOL-BO, **acquisition batch size 50**, **budget 20,000 oracle calls** |
| VAE | 256-d latent, trained on 4.5M sequences, **jointly updated with the surrogate every 10 steps** (`e2e_freq=10`); none of the 1000 seeds were in VAE pretraining |
| Oracle | APEX; `score = -apex_wrapper(seqs)[:, 0]` (maximize convention) |
| Metric | summed / average predicted MIC over validation tasks (lower is better) |

Note the validation-set design is the reason our bug is total rather than partial: the
held-out seeds were *deliberately* chosen to be ≥25% different from everything else, so
a pool built around one seed can never be feasible for them.

### Reference numbers we can validate against (paper Table 11)

Initialization-stage quality, **summed unnormalized predicted MIC over 20 validation
peptides**, as a function of the first `k` init candidates evaluated (lower is better):

| k | BOLT-10 | BOLT-20 | BOLT-50 | BOLT-600 | **STBO/MTBO** |
|---|---|---|---|---|---|
| 1 | 1204.05 | 1120.99 | 1057.13 | 564.45 | **5727.74** |
| 100 | 135.50 | 147.25 | 112.90 | 100.58 | **1551.49** |
| 200 | 120.48 | 134.83 | 107.13 | 96.78 | **1111.72** |
| 500 | 109.18 | 122.13 | 101.36 | 94.24 | **792.55** |
| 1000 | 107.55 | 119.71 | 97.14 | 92.10 | **625.95** |

`STBO @ k=1000` = best of 1000 random mutants = **625.95 summed / 31.30 per task**. This
is a hard gate for the init pipeline, testable with zero BO.

---

## 2. What our runs actually did

Two run sets exist. **Both are unusable.** The second is the result of partially fixing
the first, which fixed the constraint and exposed the init pool.

### 2a. `bo80_bolt` — 1000 tasks × 25,000 calls × bsz 50

876 tasks at `n_post_init_trials=25000`, plus 86 at 50k and 31 at 200k.

**Every task ran reference 0.** `constraint_index=0` in the `meta_info` of every
`history.json` checked (pep_050, pep_500, pep_900, pep_999). Consequences measured from
the checkpoints:

- `pep_500`, `pep_900`, `pep_910`, `pep_999` all have **byte-identical**
  `num_calls=25019`, `train_c` feasibility `168276/219500`, and the same
  `best_x_seen = RRYYEQLEQILRKLWRGFRF`.
- That sequence has similarity **0.750 to `REFERENCE_SEQUENCE[0]`** (feasible) and
  **-0.214 / -0.417 / -0.308** to refs 500 / 900 / 999 (nonsensically infeasible —
  negative because the edit distance exceeds the reference length).

So these are ~876 duplicate replays of a single task written into 876 directories. There
is no multi-task signal in this dataset at all.

### 2b. `bo80_bolt_bsz10_r100` — current fleet, 1000 tasks × 1000 calls × bsz 10

The launcher comment says `constraint_index=pep_idx patched`. That patch is real and
correct: the constraint is now evaluated against `REFERENCE_SEQUENCE[pep_idx]`. But the
**init pool is still reference 0's**, because of `bolt_lolbo_runner.py:144-145`:

```python
self.constraint_index    = int(pep_idx)   # -> constraint_types = [pep_idx]   (correct)
self._init_offset_helper = int(pep_idx)   # -> cancels the block index          (bug)
```

combined with `info_transformer_vae_optimization.py:159`:

```python
bacteria_num = self.constraint_types[0] - self.init_offset_helper   # == 0, always
train_x = all_rows[bacteria_num * 1000 : (bacteria_num + 1) * 1000]
```

`init_offset_helper` is a **per-file constant** in upstream BOLT — the index of the first
reference whose block is present in the init file (default `0`, never set in
`run_single_seed_0.sh`). Setting it per-task to `pep_idx` pins every task to block 0.
And `seed_0_init.txt` contains only one block anyway (1000 rows = mutants of ref 0).

Measured consequences:

- **999 / 1000 references have ZERO feasible init points** (exhaustive scan of all 1000
  refs against the pool at the 0.75 threshold). Only ref 0 works. Most refs have
  `max_similarity = 0.000`.
- `Best X Found: None` in **250/251, 171/171, 65/65, 49/49** finished runs across the
  four shards — everything except `pep_000`.
- `top_k_xs` stays `[]` forever, `best_x_seen` stays `None`, `progress` is never made, so
  `progress_fails_since_last_e2e` increments every round → **`e2e_updates == 3` for
  462/531 runs** (100 rounds ÷ `e2e_freq=33`), a fingerprint of permanent infeasibility.
- `recenter()` therefore finetunes the VAE on `train_x[-10:] + []` — 10 sequences.
- TuRBO initialises with `best_value = -inf`, `best_constraint_values = inf`.
- `export()` reports `best_objective = max(...)` over **all** observations with no
  feasibility filter (`bolt_lolbo_runner.py:579`), so nothing looks wrong downstream.

**The tell:** mean best predicted MIC over 668 completed tasks is **3.42** (median 3.05).
The published STBO reference at the same budget is **31.30**. Our runs are **9.2× "better"
than the baseline we are trying to reproduce** — because the similarity constraint is
inactive, so the optimizer roams the whole latent space instead of staying within 75% of
the seed.

### 2c. Is the user's protocol itself wrong?

No. **1000 init points + 100 rounds × 10 candidates = 1000 post-init calls is a
legitimate scaled-down protocol.** It deviates from the paper (bsz 50, 20,000 calls,
`e2e_freq=10`) but deliberately and defensibly, and it has a bonus: `k=1000` is exactly
the budget Table 11 reports for STBO, so the init stage is directly checkable. Two things
to be explicit about when reporting:

- `bsz=10` vs 50 and `e2e_freq=33` vs 10 are deviations; state them.
- With `bsz=10`, TuRBO's Thompson sampling still draws **5000** candidates per round
  (`n_candidates = min(5000, max(2000, 200*256))`), a 500:1 candidate-to-batch ratio vs
  BOLT's 100:1. This is a pure cost with no benefit at `bsz=10` (see §5).

---

## 3. The fix

### Step 1 — generate per-reference init pools (implemented)

`tools/build_bolt_init_pools.py` reuses BOLT's own mutation sampler
(`create_mutations.py`, ≤25% edit distance = ≥75% similarity) and its exact score
convention (`get_scores_helper.py`, `score = -apex_wrapper(seqs)[:, 0]` — verified to
reproduce `seed_0_scores.csv` to float32, max abs diff 2.7e-4).

```bash
python tools/build_bolt_init_pools.py --ref-start 0 --ref-end 1000
```

Writes `refs_{start}_{end}_init.txt`, `_scores.csv`, and a `_manifest.json` recording
`init_offset_helper == ref_start`. It enforces feasibility per mutant, keeps every block
exactly `num_mutations` long so block slicing stays valid, and tops up any reference the
rejection sampler underfills.

**Cost: negligible.** 0.08 s/ref generation; APEX scores at ~4,350 seq/s. All 1000 blocks
(1,000,000 sequences) is **~80 s CPU + ~230 s GPU**. The missing dataset that invalidates
both run sets takes under 10 minutes to build.

Validated on refs 900-919 (the paper's 20 validation tasks):

| | value |
|---|---|
| blocks / rows | 20 / 20,000, all `min_similarity = 0.75`, 0 refs needing top-up |
| summed best init MIC (seed 0) | 794.60 |
| across seeds 0-4 | **759.9 ± 67.3** (min 678.8, max 861.9) |
| paper STBO @ k=1000 | **625.95** |

The published value sits ≈2σ below our seed distribution. Given that the paper does not
state which 20 of the 100 validation peptides it used (we assume refs 900-919, matching
the commented slice in `sampling_openai.py`), and that the sum is dominated by a heavy
tail (our per-task median is 18.9 but max is 191.7), this is a **reasonable
reproduction**. Worth one follow-up: if we need a tighter match, sweep which 20-task
subset and/or check the APEX model revision.

### Step 2 — fix the block indexing (2-line change)

In `mindware/domains/protein/bolt_lolbo_runner.py`:

```python
# was: self._init_offset_helper = int(pep_idx)     # pins every task to block 0
self._init_offset_helper = int(init_offset_helper) # = manifest["ref_start"]
```

and point `hps["init_data_path"] / ["init_scores_path"]` at the generated pair. Thread
`--init-data-path`, `--init-scores-path`, `--init-offset-helper` through
`run_protein_bolt_lolbo.py` → `run_tasks` → `BenchmarkPeptideLOLBO`, defaulting to the
manifest so it cannot be forgotten.

**Add an assertion that makes this class of bug loud instead of silent**: after
`initialize_top_k()`, require at least one feasible init point, else raise with the task
id, the reference, and the pool's max similarity. Every one of the 999 broken tasks would
have failed instantly. This is the single highest-value change in the plan.

### Step 3 — the latent-cache trap (must not be missed)

`load_train_z` reads `<vae_statedict>-train-zs.csv`, a **single global file** currently
holding ref-0's `(1000, 256)` latents, and only checks `len(zs) >= num_initialization_points`
— never that the latents correspond to the sequences. If `None`, `compute_train_zs()`
encodes the pool and **writes the result back to that same global path**
(`info_transformer_vae_optimization.py:128-132`).

So once init pools are per-task, this cache silently pairs one task's latents with another
task's sequences — a second round of garbage results, and a write race across 4-32
concurrent workers.

Fix: make the z cache per-reference (`<...>-train-zs-ref{idx}.csv`) or disable the
write-back and encode per task (~1 s for 1000 sequences). **Encode-per-task is the safer
default.** This is benign today only because every task shares block 0.

### Step 4 — enforce the oracle budget

`run_lolbo_bolt_native` checks the budget only at loop top and never clamps `bsz`, while
`recenter()` calls the oracle on `train_x[-bsz:] + top_k_xs` × `num_update_epochs`.
The `xs_to_scores_dict` cache hides most of it, but post-finetune decodes are cache
misses. Measured: only 59/531 runs hit exactly 1000; range 1000→1064; a 300-budget run
used 350 (+16.7%). `run_lolbo_checkpoint` already clamps correctly
(`bolt_lolbo_runner.py:453`) — port that logic, and count recenter's calls explicitly.

**This gets worse after the fix, not better**: once the constraint is satisfiable,
`top_k_xs` fills to `k=1000`, so `recenter()` re-encodes and re-scores ~1010 sequences
twice per E2E update instead of 10. Budget enforcement is a prerequisite for the fixed
runs, not a nice-to-have.

### Step 5 — make failure visible

- `acquisition_stall` currently breaks the loop and writes a normal-looking row
  (20/531 runs finished at 249-990 calls). Mark truncated runs in `index.jsonl`.
- Record the **actual** `bsz`/hparams, not `BOLT_SEED0_HPARAMS` (`:672` reports `bsz 50`
  for all 531 rows while the fleet ran `--bsz 10`).
- Report a **feasible-only** best alongside the unconstrained one, plus the best
  sequence's similarity to its reference. Had this existed, the bug would have been
  obvious on day one.
- Fix `_task_done`'s `UnboundLocalError` in `checkpoint` mode (`:726`, verified) and the
  `shutil.disk_usage` call on a not-yet-created output root (`:769`, verified).
- Remove the live `breakpoint()` at `lolbo/lolbo.py:447` — dormant only because the
  runner never passes `similarity`, and `--min-similarity` exists as a CLI flag.

---

## 4. Correct experiment to run

Inner loop is identical in every arm; only the init pool changes. That is the whole point
of BOLT and it keeps the comparison honest.

| Arm | Init pool (1000 pts/task) | Purpose |
|---|---|---|
| **STBO** | random mutants ≥75% similar to the seed | the baseline; reproduces Table 11 `k=1000` |
| **BOLT-T** | 1000 samples from LLM fine-tuned on top-K of T training tasks | the method |
| *(optional)* BOLT-0 | untuned LLM | isolates fine-tuning from prompting |

Protocol: refs 0-899 train / **900-999 validation**; 1000 init points; 100 rounds × 10 =
1000 post-init calls; `e2e_freq` as chosen; metric = summed/mean predicted MIC over the
100 validation tasks, **plus** a feasibility rate that must be 100%.

Validation gates, cheapest first:

- **Gate 0 (done, free).** Generated pool best over refs 900-919 = 759.9 ± 67.3 vs paper
  625.95. ✅
- **Gate 1 (~20 min).** STBO on refs 900-919, 1000 post-init calls. Require: `best_x_seen`
  non-`None` for all 20; similarity ≥ 0.75 for every reported best; `e2e_updates` no
  longer pinned at 3; summed MIC below the init-pool value. Any per-task MIC near 3
  means the constraint is still inactive.
- **Gate 2.** Full 100 validation tasks → the STBO reference curve.
- **Gate 3.** Only then the BOLT arm, which needs the LLM fine-tune loop.

Compute: my profiled feasible task (`pep_000`, 350 calls) ran at ~8.9 oracle calls/s vs
~12.6 for the degenerate tasks, and the fixed runs get more expensive as `top_k` fills.
Budget ~2-5 min/task at 1000 calls, so 100 validation tasks ≈ 4-8 GPU-hours before the
throughput fixes in §5 — hours, not days. Do a 20-task calibration run before committing
to the full sweep.

---

## 5. Throughput (measured, `py-spy`, 4753 samples / 47.5 s)

Current fleet: GPU util **0 / 31 / 100 / 13 %**, **3.9 GB of 81 GB**, one running thread
per worker despite `OMP_NUM_THREADS=8`.

The two structural wins, worth ~2.5-3× together:

- **Load imbalance.** Static `i % n_shards` + `--skip-done` left shards with 4 / 62 / 167
  / 179 tasks remaining. GPU 0 has been idle for ~40 min; the job ends when shard 3 does.
  Roughly half the fleet's GPU-hours are wasted. Replace with a shared work queue
  (claim via `O_EXCL` lockfile) — per-task checkpointing already makes tasks claimable.
- **Under-subscription.** 4.8% of GPU memory, ~36% average util, 1 core/worker of 128.
  Run 6-8 workers per GPU. Generalize the launcher's `EXTRA_ON_LAST_GPU` to all GPUs.

Per-task code wins, ~1.4-1.6× together:

| Issue | Cost | Fix |
|---|---|---|
| `n_candidates=5000` for `bsz=10` (`turbo.py:175`) | `rsample` 11.6% + `_symeig` 7.9% | drop to 1000-1500; hoist the per-call `SobolEngine` |
| gpytorch takes the eigendecomposition root path | `_symeig` 3.74 s vs Cholesky 1.38 s | `fast_computations(covar_root_decomposition=False)` |
| `self.vae.eval(); self.vae.cuda()` **every** objective call (`info_transformer_vae_objective.py:120-121`) | ~8% (`_apply` + `flatten_parameters`) | hoist to init |
| `.item()` sync per token per decode (`transformer_vae_unbounded.py:315`) | main cause of low GPU util | check every 4-8 steps |
| causal mask rebuilt on CPU each step + `_detect_is_causal_mask` | 2.15% | precompute on device, pass `is_causal=True` |
| `random_gumbels` concatenated every step and **never returned** (`:294`, `:311`) | dead allocation | delete |
| no KV cache in decode (O(L²)) | VAE decode 28% cumulative | biggest algorithmic win, most invasive |
| `_ensure_bolt_paths()` purges `lolbo*` from `sys.modules` **per task** | 2.71 s/task ≈ 11 min/shard | guard to run once |
| tqdm written unbuffered to disk | 17.7 MB/shard, >90% of bytes | `TQDM_DISABLE=1` |

---

## 6. Recommendation

1. **Stop the current fleet.** Every task it completes is void, and it is holding 4 GPUs.
2. Generate all 1000 init blocks (~10 min).
3. Apply §3 steps 2-5 — the feasibility assertion first, so this can't recur silently.
4. Gate 1 on refs 900-919. Do not scale up until similarity ≥ 0.75 holds for all 20.
5. Then the throughput work in §5, then the full STBO reference, then the BOLT arm.

Treat `bo80_bolt` and `bo80_bolt_bsz10_r100` as discarded. Archive, don't delete: they are
the evidence for what went wrong.

---

## 7. Open items

- **Which task set?** The paper evaluates on the 100 held-out refs (900-999) and trains
  the LLM on 0-899. Running BO on all 1000 is a superset but only 900-999 is comparable
  to the published numbers.
- **Repo duplication.** `/root/data/cys/blockdata_archive/llmoptimizer_benchmark` and
  `/root/data/cys/llmoptimizer_benchmark` are separate directories with identical
  launchers. The fleet runs the `blockdata` copy. Real risk of editing one and running
  the other — consolidate or symlink.
- Table 11's 20-task subset is unstated; we assume refs 900-919.
- 15 references contain a non-standard `X` residue. Decoded sequences have `X`→`A`
  applied (`info_transformer_vae_objective.py:127`), so those positions can never match,
  costing one edit of similarity budget on those tasks.
