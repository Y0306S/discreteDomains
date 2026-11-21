# Baseline → repo → env → scenario

Scenario codes: **M** = ML HPO, **D** = DL HPO, **P** = peptide optimization, **Q** = query optimization.
Scenario applicability is taken from `pt4opt-baselines.md`, not from what each repo happens to ship.

| # | Baseline | Upstream repo (exact) | Local path | Env | Scenarios |
|---|---|---|---|---|---|
| 1 | Random Search | — (no upstream; trivial) | `../llmoptimizer_benchmark/mindware/components/optimizers/random_search_optimizer.py` | existing `openbox_qwen_eval` | M D P Q |
| 2 | LOL-BO | `nataliemaus/lolbo` | `repos/lolbo` | `envs/bl_lolbo_exact` | M D P Q |
| 3 | TPE / Latent-TPE | Optuna `TPESampler` (paper has no repo; Latent-TPE is our own sanity baseline per spec) | `../llmoptimizer_benchmark/mindware/components/optimizers/pt4opt_tpe_optimizer.py` | existing `openbox_qwen_eval` | M D P Q |
| 4 | FSBO | `releaunifreiburg/FSBO` | `repos/FSBO` | `envs/bl_fsbo` | M D P Q |
| 5 | RGPE | `automl/transfer-hpo-framework` | `repos/transfer-hpo-framework` | `envs/bl_rgpe` | M D P Q |
| 6 | OptFormer | `google-research/optformer` @ **`neurips22`** branch | `repos/optformer-neurips22` (+ `repos/optformer` main, `repos/t5x`) | `envs/bl_optformer` | M D P Q |
| 7 | ToSFiT | `IBM/thompson-sampling-via-fine-tuning-of-llms` | `repos/tosfit` | `envs/bl_tosfit` | P (per spec) |
| 8 | BORA | `Ablatif6c/bora-the-explorer` | `repos/bora-the-explorer` | `envs/bl_bora` | M D P Q |
| 9 | BOLT | `Yimeng-Zeng/BOLT` | `repos/BOLT` → `/root/data/cys/BOLT` | existing `bolt_clean_venv` | M D P Q |
| 10 | SemanticOpt | **no public code** — implement from paper spec | `impl/semanticopt/` | `envs/bl_semanticopt` | M D P Q |

Also linked: `repos/BayesQO` → `/root/data/cys/BayesQO` (`Speculative/BayesQO`), the query-plan
LOL-BO variant already vendored here; useful as the query-scenario reference.

## Why some entries have no upstream repo

- **Random Search** — nothing to reproduce; the spec defines it directly (uniform sampling in the
  raw space for M/D, uniform latent sampling + decode for P/Q).
- **TPE / Latent-TPE** — the spec states outright that `Latent-TPE` is *"a custom sanity baseline
  run in the latent space starting from the same initial VAE as LOL-BO, not an independent method
  from the literature."* Plain TPE = Optuna's `TPESampler`.
- **SemanticOpt** — the paper (Meindl et al., MIT / MIT-IBM) ships **no code**; its only GitHub
  references are to `unslothai/unsloth` (their trainer) and `equinor/neqsim` (a benchmark).
  So it must be built from the spec. The spec is explicit about the data format, supervision
  fields and inference rules, and permits a different backbone as long as those are preserved
  (label it *backbone-controlled protocol reproduction*).

## Deviations from upstream pins, and why

| Baseline | Deviation | Reason |
|---|---|---|
| LOL-BO | none — exact pins (`torch==1.11.0`, `gpytorch==1.6.0`, `botorch==0.6.3.1`, `numpy==1.22.3`, `selfies==2.0.0`) | `torch 1.11.0` on PyPI is the cu113 build, which covers the A800's sm_80 |
| FSBO | `numpy` pinned `<1.23`; torch/gpytorch newer than the README's 1.8.1/1.4.2 | upstream calls `np.asscalar` (removed in numpy 1.23). `torch==1.8.1` on PyPI is a cu102 build with **no sm_80 support**, so it cannot run on this A800 host |
| RGPE | drops `HPOlib1.5` | only used by `rgpe/test_functions.py` (their own AdaBoost/SVM/OpenML benchmarks). The methods we need (`rgpe/methods/rgpe.py`, `taf.py`, `rgpe/utils.py`) do not import it |
| OptFormer | `neurips22` branch, not `main` | `main`'s `optformer/original/` is a rewrite; the spec asks for the original T5 encoder–decoder paper implementation, whose code is on `neurips22` |
| ToSFiT | `flash_attn` not installed | needs a long local CUDA compile; `attn_implementation` can be overridden to `sdpa`. Everything else is at the pinned versions |
| SemanticOpt | Qwen backbone instead of Qwen 3.5 27B | compute; the spec explicitly allows this and requires it be reported as a backbone-controlled protocol reproduction |

## Data / checkpoint dependencies still outstanding

| Baseline | Needs | Status |
|---|---|---|
| FSBO | HPO-B benchmark data (`hpob-data/`) | **only for reproducing the repo's own figures** — not needed for our four scenarios, which supply their own tasks |
| RGPE | AdaBoost + SVM grids from `nicoschilling/ECML2016`, LCBench | same — only for upstream's benchmarks |
| OptFormer | `gs://gresearch/optformer/{model_checkpoints,datasets}` | GCS is reachable. Per spec we train one OptFormer per scenario on **our own** trajectories, so the public checkpoints are optional (useful as a sanity reference) |
| ToSFiT | protein LLM + reward model weights | pull via `hf-mirror.com` |
| BORA | an OpenAI-compatible endpoint | point at a local vLLM Qwen server so the LLM backbone matches our method |
