# Peptide SFT — handoff guide

Written 2026-09-12 for whoever picks this up next. Everything upstream of SFT is done: the
BO bug is fixed, all 1000 tasks have been re-run, and the SFT datasets are rebuilt and
validated. What remains is training and evaluation.

Read alongside:
- `docs/BOLT_PEPTIDE_BO_FIX_PLAN.md` — what was broken in the BO and why
- `docs/BOLT_PEPTIDE_SPLIT.md` — the pre-registered train/test split. **Do not change it.**

---

## 0. Context in one paragraph

BOLT (Zeng et al., ICLR 2026) is **initialization-only transfer**: no multi-task GP. You
run single-task LOL-BO ("STBO") on training tasks, fine-tune an LLM on
`(seed sequence → top solutions)` pairs, then use that LLM to generate the 1000
initialization points for a *new* task and run the unchanged BO loop on top. So the LLM's
only job is to propose good initial peptides for an unseen seed. That is what this SFT
trains.

## 1. State of the data

BO source of truth: `/root/data/cys/blockdata_archive/domain_protein/stbo_bsz10_r100_fixed`
(1000/1000 tasks, 61 GB, 1000 init points + 1000 acquisition calls per task).

> **`index.jsonl` contains 30 duplicate rows** from task re-runs. Always dedupe by
> `task_id` keeping the **last** occurrence. `tools/build_peptide_sft.py` already does.

SFT datasets in `benchmark_data/rl_task_selection/peptide_sft_v2/` (LlamaFactory-registered
in `dataset_info.json` there):

| dataset | train | eval | assistant target |
|---|---|---|---|
| `peptide_sft_elite10` | 897 | 100 | Rationale + JSON, 10 sequences |
| `peptide_sft_elite10_jsononly` | 897 | 100 | JSON only, 10 sequences |
| `peptide_sft_elite10_bolt` | 897 | 100 | single sequence, BOLT's minimal prompt |
| `peptide_sft_winbatch10` | 197 | 18 | Rationale + JSON (**see warning below**) |

Split is train `pep_000..pep_899`, eval `pep_900..pep_999`. 3 tasks are skipped in every
arm (`too_few_feasible_targets`) — short references whose feasible neighbourhood is too
small to yield 10 distinct sequences.

Rebuild any arm with:

```bash
cd /root/data/cys/blockdata_archive/llmoptimizer_benchmark
PYTHONPATH=. python tools/build_peptide_sft.py --arm elite10                 # rationale+JSON
PYTHONPATH=. python tools/build_peptide_sft.py --arm elite10 --no-rationale  # JSON only
PYTHONPATH=. python tools/build_peptide_sft.py --arm elite10 --prompt-style bolt
```

The builder **validates every row before writing** and will refuse to emit anything that
violates the guarantees in §4. If it raises, that is the point — do not disable the check.

## 2. The two training configs

`configs/peptide_sft/`:

| config | dataset | mirrors |
|---|---|---|
| `llamafactory_qwen35_9b_lora_peptide_s1b_elite10_ep7p5.yaml` | `peptide_sft_elite10` | winning `p5_s1b_ep7p5` (rationale+JSON) |
| `llamafactory_qwen35_9b_lora_peptide_s1a_elite10_jsononly_ep7p5.yaml` | `peptide_sft_elite10_jsononly` | winning `p5_s1a_ep7p5` (JSON-only BC) |

**Why these two.** The original ML sweep covered 2.5 vs 7.5 epochs × two output-format
arms. There is no leaderboard file on disk, so "top 2" is inferred from what the project
actually built on afterwards, which is unambiguous:

- `s1a_ep7p5` → 11 DPO variants, 4 `r0*` runs, 6 `rw_`/`prefix_` RL runs
- `s1b_ep7p5` → 11 DPO variants, `attr_t2`/`t3`, the entire GRPO line, **and the only
  merged model on disk** (`qwen35_9b_merged_p5_s1b_ep7p5`)
- `ep2p5` → one DPO variant (`s1b_ep2p5_dpo_D`), then abandoned

So 7.5 epochs beat 2.5, and both output-format arms were carried forward. Note the two
winning configs are **hyperparameter-identical** — they differ only in dataset. The shared
recipe:

```
LoRA r=32, alpha=64, dropout=0.05, target=all, loraplus_lr_ratio=16.0
lr=1.25e-4, cosine, warmup_ratio=0.03, epochs=7.5
per_device_batch=2, grad_accum=8  (effective 16 x n_gpu)
bf16, gradient_checkpointing, neftune_noise_alpha=5, seed=42
base: /root/eb-public/huggingface-models/Qwen/Qwen3.5-9B, template qwen3_5_nothink
```

The only deliberate change for peptides is **`cutoff_len: 1024`** (was 6016). Measured
token lengths: rich prompt p50 830 / max 906; bolt prompt max 76. Nothing truncates, and
it is ~6x less padding per step.

## 3. Running it

LlamaFactory lives at `/root/data/cys/LLMOptimizer/LlamaFactory`.

> The old ML configs point `dataset_dir` at `/root/LLMOptimizer/...`, which **does not
> exist** on this box (real path is `/root/data/cys/LLMOptimizer/...`). The new peptide
> configs use an absolute path that does exist. If you copy an old config, fix this first.

```bash
cd /root/data/cys/LLMOptimizer/LlamaFactory
CFG=/root/data/cys/blockdata_archive/llmoptimizer_benchmark/configs/peptide_sft

# single GPU
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  $CFG/llamafactory_qwen35_9b_lora_peptide_s1b_elite10_ep7p5.yaml

# 2-GPU DDP (how the ML sweep ran: two configs in parallel, 2 GPUs each)
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
  $CFG/llamafactory_qwen35_9b_lora_peptide_s1b_elite10_ep7p5.yaml
CUDA_VISIBLE_DEVICES=2,3 llamafactory-cli train \
  $CFG/llamafactory_qwen35_9b_lora_peptide_s1a_elite10_jsononly_ep7p5.yaml
```

Weights land in `/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_lora_peptide_*`.

Sanity checks before a long run: 897 rows × 7.5 epochs ÷ (2×8 per step) ≈ 420 optimizer
steps per GPU-pair — this is a *small* dataset, expect well under an hour. Confirm the
loss curve actually moves and that `cutoff_len` produced no truncation warnings.

## 4. What the data guarantees, and why you must not weaken it

The previous SFT export was unusable. Both failures were verified on its 900-row output:

| check | old pipeline | current |
|---|---|---|
| rows with the answer verbatim in the prompt | **900/900** | **0/897** |
| target sequences violating the ≥0.75 similarity constraint | **9000/9000** | **0/8970** |
| target similarity (min / median) | −1.000 / 0.000 | 0.750 / 0.800 |

1. **Label leakage.** `build_user_prompt()` wrote the exact 10 answer sequences *and their
   MICs* into the user message via `target_recommendations_block()`; the assistant then
   repeated them in order. The model was trained to copy from its own input. Note the ML
   pipeline it claimed to mirror (`p5_s1b_sft.jsonl`) has no such block — this was an
   implementation bug, not a design.
2. **Infeasible targets.** Every target broke the similarity constraint the prompt calls
   "hard", because the source BO run had zero feasible points (see the BO fix plan).
3. **Leaky split.** The old default `stratified10` (eval = every 10th ref) puts **18/100**
   eval references within >0.75 similarity of a training reference. The pre-registered
   last-100 split has **0/100**.

`validate_row()` now enforces, per row: no real sequence appears as a `"sequence"` value
in the prompt; no achieved MIC in the prompt; every target satisfies similarity ≥ 0.75,
length ≤ 30, ≤1 cysteine, standard alphabet; no target equals the reference or the seed;
exactly 10 distinct sequences.

**The old files in `benchmark_data/rl_task_selection/peptide_sft/` (no `_v2`) are the
broken ones. Do not train on them.**

## 5. Evaluation

The point is not SFT loss. It is: *do this model's proposals make better BO
initializations than 1000 random mutants?*

1. Sample 1000 sequences per held-out task (`pep_900..pep_999`) from the fine-tuned model,
   conditioned on that task's seed. The paper uses temperature 1.0 for peptides.
2. Filter/repair to feasible (`tools/build_bolt_init_pools.py` has the exact predicates;
   `mindware.domains.protein.bolt_constraints.repair_to_feasible` can repair).
3. Score with APEX: `score = -apex_wrapper(seqs)[:, 0]`, ~4,300 seq/s. Write init +
   scores files in the same format as `build_bolt_init_pools.py`.
4. **Init-only gate first** (nearly free): summed best MIC over the 20 tasks
   `pep_900..pep_919`, comparable to paper Table 11.
5. Then run the BO arm by passing your files via `--init-data-path/--init-scores-path/--init-offset-helper`
   to `run_stbo_fixed_fleet.sh`, and compare against the STBO reference below.

### Reference numbers already established

| quantity | value |
|---|---|
| STBO, all 1000 tasks, mean best feasible MIC | **37.23** |
| STBO, test `900-999`, mean / sum | **37.07 / 3707.0** |
| STBO, `900-919` (20 tasks), summed MIC | **774.4** |
| our random-mutant init pool, same 20 tasks | 794.6 (seed 0); 759.9 ± 67.3 over seeds 0-4 |
| **paper Table 11, STBO @ k=1000, same 20 tasks** | **625.95** |
| paper Table 11, BOLT-600 @ k=1000 | 92.10 |

Beating 774.4 is the bar. The paper's BOLT-600 reaches 92.10, i.e. ~8x better than its own
STBO baseline, so a working BOLT arm should show a large, unmistakable gap. If your
fine-tuned init only ties STBO, suspect the pipeline before believing the result.

## 6. Open issues you are inheriting

1. **We do not reproduce the paper's STBO init number.** Our pools give 794.6 vs their
   625.95 on refs 900-919 (~2.5σ of our seed spread). The gap is in the *init pool*, not
   the BO. Unresolved candidates: the paper never states which 20 of its 100 validation
   peptides it used; their mutant sampler may differ (e.g. indels, or oversample-and-keep-best);
   APEX model revision. Worth an hour before publishing any comparison.
2. **BO barely improves on the init pool at this budget.** Mean 1.08x, median 1.00x; only
   151/900 tasks improve >5%; aggregate summed MIC 35,543 → 33,963 (−4.4%). Verified to be
   budget, not breakage (`pep_006` goes 1.00x → 1.76x at 5x budget). **Consequence: at
   1000 calls this experiment is predominantly an init-quality comparison.** That is a
   legitimate and cheap experiment — it is what Table 11 measures — but it is not the
   paper's Figure 1 and must not be labelled as such.
3. **`winbatch10` is not viable at this budget.** It needs a post-init batch that improves
   on the best feasible init MIC, which happens for only ~19% of tasks → 197 rows vs 897.
   Not a fair ablation against `elite10`. Drop it or re-run at a larger budget.
4. **Budget definition is a live choice.** Runs used `--budget-mode acquisition`: exactly
   1000 acquisition calls, but *total* oracle consumption averaged 1,476/task and ranged
   590→8,800 because E2E VAE re-scoring is uncounted. BOLT's own loop terminates on total
   `num_calls`, so `--budget-mode total` is the faithful-to-paper setting. Both counts are
   in every row (`n_acquisition_calls`, `n_e2e_calls`, `n_oracle_calls`) — decide which you
   headline, but do not call 1000 acquisition calls "1000 oracle calls".
5. **69/1000 tasks are `truncated: candidate_exhaustion`** — the VAE decoder saturates the
   feasible neighbourhood of short references. Real property, not a defect. Decide whether
   to report or exclude them; they are flagged per row.
6. **Two copies of this repo exist** (`/root/data/cys/blockdata_archive/llmoptimizer_benchmark` and
   `/root/data/cys/llmoptimizer_benchmark`) with identical launchers. Everything here was
   done in the `blockdata` copy. Consolidate or symlink before editing.
