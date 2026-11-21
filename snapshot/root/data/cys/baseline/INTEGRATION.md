# Wiring the baselines into our study

Target of comparison (the "our method" run these baselines must sit beside):

```
adapter   /root/data/cys/blockdata_archive/model-weights/
            llamafactory_qwen35_9b_lora_p5_s1b_ep7p5_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12/ckpt32
prompt    P5plusA5T50X50RS5f05v4
hybrid    HPUCB_arm         planner  ptllm_tpe_qwen
label     ptllm_tpe_qwen__P5plusA5T50X50RS5f05v4__HPUCB_arm
study     /root/data/cys/blockdata_archive/llmoptimizer_benchmark/benchmark_data/
            real_hpo_v2_mltest20_algo40_p5plus_grpo_rsmix_v12_p5_s1b_attr_t2_b_prefix_grpo_rsmix_v4_pareto32_v12_ckpt32/
protocol  tasks_40.json (20 cls + 20 rgs) x 80 trials, seed 1, time_limit 10800, holdout
base LLM  /root/data/cys/blockdata_archive/model-weights/qwen35_9b_merged_p5_s1b_ep7p5  (Qwen 3.5 9B)
```

Per-dataset results land as `<study>/{cls,rgs}/<dataset>/summary.json` with one entry per
optimizer label, plus a trajectory under
`history_data/<task_type>/<metric>/<dataset>/<label>/<algorithm>.json`.
Any baseline we add has to emit the same shape to be picked up by
`plot_mltest20_algo40_avg_rank.py`.

## What already runs under the exact protocol

| Baseline | How |
|---|---|
| TPE | `run_toy_categorical_optimizers.py --optimizer tpe`, already one of `CLASSICAL_OPTIMIZERS` and already used by the harness that produced the study |
| Random Search | `mindware` accepts `optimizer='random_search'` (`base_cash.py`), and `RandomSearchOptimizer` exists — but `random` is **not** in the runner's `CLASSICAL_OPTIMIZERS = ["smac","tpe","cmaes","hebo"]`, so it is not yet reachable from the CLI. One-line registration needed (see below) |

## What each remaining baseline still needs

These are **adapter tasks**, deliberately kept out of `repos/` so upstream stays pristine.

### LOL-BO
- ML/DL HPO: per spec, no VAE — feed standardised hyperparameter vectors straight to the GP
  and keep the adaptive trust region (i.e. TuRBO-style local GP-BO). Needs a
  `LatentSpaceObjective` subclass whose `vae_decode`/`vae_forward` are identity over the
  normalised config vector, plus round-trip to/from `ConfigSpace`.
- Peptide/query: full LOL-BO on the shared VAE. `repos/BOLT` and `repos/BayesQO` already
  contain working peptide and query-plan LOL-BO wirings to copy from.

### FSBO
- Group train tasks by **identical search-space schema** (same arity, per-dim semantics,
  types, ranges, conditional activation), one meta-trained FSBO per group, ML and DL kept
  separate. Feed `{task_id: {"X": [...], "y": [...]}}` (verified format).
- At meta-test, `DeepKernelGP.observe_and_suggest(X_obs, y_obs, X_pen)` is the entry point.
- Spec explicitly disables FSBO's own evolutionary warm start — use the shared warm start.

### RGPE
- Future ML fits use `examples/benchmark/rgpe_exact.py` through the default
  `rgpe_worker.py` entry point. Source training and online target fitting use
  streamed, bounded-memory likelihood gradients, including categorical and
  conditional spaces. Keep existing source caches; kernel formulas, priors,
  normalization, ranking, acquisition and optimizer restarts are unchanged.
  This preserves upstream gradient conventions, with floating-point tolerance
  rather than a guarantee of identical configurations in every optimization run.
  Validation: `/root/data/logs/ml_rgpe_exact_check_20261028_final/README.md`.
- Build `{task_id: {"configurations": [Configuration], "y": np.ndarray}}` per schema group;
  **all** compatible source tasks enter the set (no top-K prefilter).
- Run `RGPE(TAF)` with the settings verified in `tests/smoke_rgpe.py`:
  `sampling_mode="bootstrap"`, `normalization="Copula"`,
  `weight_dilution_strategy="probabilistic-ld"`, `num_posterior_samples=1000`.
- Confirmed working: uniform weights below 3 target observations, then pairwise-misranking
  weights with per-round base-model dropping.
- RGPE lives in an isolated py3.7 env, so it must be driven as a **subprocess** from the
  harness, not imported into it.

### OptFormer  (spec marks this 先跳过 / deferred)
- Train one model per scenario on our own ordered trajectories; metadata carries the
  trajectory-generating algorithm's identity. Continuous/integer dims quantised to
  `Q=1000` tokens; objectives normalised to larger-is-better with a range frozen from
  train tasks only.
- Needs a seqio Task built from our trajectories (`optformer/data/tasks.py` is the template),
  then `t5x/train.py` with `--gin_file=runs/train.gin`.
- Public checkpoints/datasets at `gs://gresearch/optformer/...` are reachable if we want a
  sanity reference, but the spec has us train from scratch.

### ToSFiT
- Peptide scenario only. Swap their `protein_stability` (Biopython instability index) for
  our APEX mean-MIC oracle, and their `nferruz/ProtGPT2` generator for our peptide LLM.
  Entry point `scripts/run_tosfit.py --config experiments/tosfit/protein/...`.
- `flash_attn` is not installed; set `attn_implementation="sdpa"` or install it.

### BORA
- `Bora(experiment, api_key, log_path, llm_model=...)` speaks the OpenAI API. Point it at a
  local OpenAI-compatible server for our Qwen 3.5 9B base so the LLM backbone matches our
  method; the harness's `examples/benchmark/hf_qwen_client.py` is the existing Qwen path.
- Note BORA **maximises** — our objectives are OpenBox-style minimisation, so negate.
- Its three actions (`BO`, `ASSISTANT_SUGGESTS`, `ASSISTANT_SELECTS`) already match the
  spec's "pure BO / LLM proposes / LLM re-ranks" scheduling.

### BOLT
- Already wired for peptide and query in `llmoptimizer_benchmark/examples/benchmark/domains/`
  (`run_protein_bolt_lolbo.py`, `run_query_bolt_lolbo.py`). For ML/DL HPO it needs the
  top-K-config SFT → LLM warm start → LOL-BO handoff.

### SemanticOpt
- No upstream code; build from spec in `impl/semanticopt/`. Two pieces:
  1. **Offline replay** to recover `(mu, sigma)` — for each decision step `t` of each
     train-task trajectory, refit a LOL-BO surrogate on the causal prefix `D_t` only and
     score the already-recorded `x_{t+1}`. Never re-optimise the acquisition or reselect
     `x_{t+1}`. Add `p_fail` only after a failure has actually been observed.
  2. **Assistant-only LoRA SFT** with Unsloth: r/alpha 16, lr 1e-5 cosine, bf16, bs 64,
     1 epoch, ctx 24576. Verified working on our Qwen 3.5 9B base
     (`tests/smoke_semanticopt_lora.py`, loss 1.01 over 2 steps).
- At test time sample 2 candidates/step, score externally by `(1-p_fail) * EI`, and fit
  **no** external surrogate.
- Must be reported as a *backbone-controlled protocol reproduction* (Qwen 3.5 9B, not the
  paper's Qwen 3.5 27B).

## Registering Random Search in the runner

`examples/benchmark/run_toy_categorical_optimizers.py` needs `random_search` added to
`CLASSICAL_OPTIMIZERS` and `TRAJ_LABELS`; `mindware`'s CASH dispatch already accepts it.
Not done yet — that file is part of the harness that generated the study's numbers, so it
is worth doing as a reviewed change rather than silently.
