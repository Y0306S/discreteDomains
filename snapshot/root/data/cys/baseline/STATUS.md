# Setup status — 2026-09-15

Current ML status: [Approved policy and remaining work, 2026-09-25](../llmoptimizer_benchmark/docs/ML_FIXED_READINESS_2026-09-25.md). LOLBO and random search are stopped; TPE continues under its existing launch settings. The prepared policy is 48 hours per evaluation, no overall search timeout, and 100 attempts including failures. Reusing clean results leaves 231 method/task results still needed on the 77-dataset/132-pair split. The [initial recovery report](../llmoptimizer_benchmark/docs/ML_FIXED_RECOVERY_2026-09-25.md) preserves earlier launch details.

Current experiment problems: [Run issue register, 2026-09-25](../llmoptimizer_benchmark/docs/RUN_ISSUES_2026-09-25.md). This register covers the ML split mismatch, overlapping random-search jobs, resume and timeout problems, result summaries, TaskSetBO CPU affinity, and BOLT budget accounting. The setup checks below describe the earlier environment setup.

`bash /root/data/cys/baseline/verify_all.sh` → **10 passed, 0 failed**
(full log: `logs/verify_final.log`)

| # | Baseline | Repo | Env (all under `/root/data`) | Verified by |
|---|---|---|---|---|
| 1 | Random Search | in-harness | `openbox_qwen_eval` | `RandomSearchOptimizer` + openbox SMBO import |
| 2 | LOL-BO | `nataliemaus/lolbo` | `bl_lolbo_exact` | 5 acquisition rounds, 25 oracle calls, trust region + DKL-GP |
| 3 | TPE / Latent-TPE | optuna (in-harness) | `openbox_qwen_eval` | 25-trial TPE study converges to x≈0.29 |
| 4 | FSBO | `releaunifreiburg/FSBO` | `bl_fsbo_exact` | meta-train writes `weights`; meta-test `observe_and_suggest` picks a candidate |
| 5 | RGPE | `automl/transfer-hpo-framework` | `bl_rgpe` | RGPE(TAF): uniform weights <3 obs, then rank weighting + base-model dropping |
| 6 | OptFormer | `google-research/optformer` @ `neurips22` | `bl_optformer` | jax 0.4.23 sees all 4 GPUs; t5x + seqio + optformer + t5.data import |
| 7 | ToSFiT | `IBM/thompson-sampling-via-fine-tuning-of-llms` | `bl_tosfit` | all 29 `src.*` modules import; protein reward computes |
| 8 | BORA | `Ablatif6c/bora-the-explorer` | `bl_bora` | 3 actions match spec; UCB + EI suggest on a fitted GP |
| 9 | BOLT | `Yimeng-Zeng/BOLT` | `bolt_clean_venv` | peptide LOL-BO imports; uniref VAE ckpt present |
| 10 | SemanticOpt | none (built from spec) | `bl_semanticopt` | **real LoRA SFT on Qwen 3.5 9B**: r/alpha 16, lr 1e-5 cosine, bf16, 2 steps, loss 1.01 |

## Resolved during setup

| Problem | Fix |
|---|---|
| `github.com` / `pypi.org` unroutable | `gh-proxy.com` for git; aliyun + nju for pip; bfsu for conda. `git insteadOf` rewrite added so `pip install git+https://github.com/...` works (this is what unblocked t5x's `airio`/`jestimator`) |
| LOL-BO on PyPI `torch==1.11.0` is a **cu102** build → no sm_80, unusable on A800 | installed `torch-1.11.0+cu113` from `mirrors.aliyun.com/pytorch-wheels/cu113`; `get_arch_list()` now includes `sm_80` |
| FSBO calls `np.asscalar` (gone in numpy ≥1.23) | env pinned to `numpy==1.22.3` (+ `sklearn 1.1.3`, `xgboost 1.7.6`, `matplotlib 3.5.3` to match) |
| RGPE: `pyrfr` build failed — `swig` not found | `swig` was installed *inside* the env but its `bin/` was not on the build PATH |
| RGPE: `scikit-optimize 0.10.2` needs py≥3.8 (env is py3.7) | pinned `scikit-optimize==0.8.1`, `scikit-learn==0.22.1` per upstream `install.sh` |
| OptFormer `main` is a rewrite, not the paper's model | used the `neurips22` branch via a git worktree |
| modern `flax` requires py≥3.12; t5x `main` tracks git HEAD | pinned a Jul-2023 release stack and a `t5x` worktree at `45df405` (Jul 31 2023) |
| `clu==0.0.10` pins `numpy==1.23.1` vs `chex` needing ≥1.24.1 | `clu==0.0.12` + explicit `numpy==1.24.3` |
| `t5==0.9.4` hard-depends on `seqio-nightly`, which evicted `seqio==0.0.13` and dragged in TF 2.21 | install `t5 --no-deps` **last**, with its real deps installed beforehand, under a pip constraints file |
| `tensorflow_hub` needs `pkg_resources` (removed in setuptools ≥81) | `setuptools<81` |
| jax 0.4.23 needs `scipy.linalg.tril` (removed in scipy 1.13) | `scipy==1.11.4` |
| `jax[cuda12_pip]` resolved cuDNN 9.26, but `jaxlib 0.4.23+cuda12.cudnn89` needs 8.9 → silent CPU fallback | pinned `nvidia-cudnn-cu12==8.9.7.29`; jax now reports `cuda(id=0..3)` |
| overlay `/` hit **95 % (1.6 G free)** from a 13 G pip cache | moved `pip`/`huggingface`/`unsloth`/`torch` caches to `/root/data/cys/baseline/cache/` with symlinks; conda `pkgs_dirs`+`envs_dirs` repointed there too. Overlay now 53 % |

## Everything is on `/root/data` — nothing on the system disk

```
/root/data/cys/baseline/envs    34G   7 envs, all standalone (no env depends on the overlay)
/root/data/cys/baseline/repos  335M   upstream source, unmodified
/root/data/cys/baseline/cache   15G   pip / hf / unsloth / torch / conda pkgs
```

`/root/.cache/{pip,huggingface,unsloth,torch}` are now symlinks into the above.
The two redundant layered venvs from the first attempt (`bl_lolbo`, `bl_fsbo`) were removed;
`bl_bora` was rebuilt as a standalone conda env so no baseline env inherits site-packages
from the method's own eval env.

## Not done yet (deliberately)

1. **Scenario adapters** — none of the baselines is wired to our 4 scenarios yet; that is the
   next phase and it is where the real work is. See `INTEGRATION.md` for the per-baseline
   list.
2. **Random Search CLI registration** — `mindware` already accepts `optimizer='random_search'`,
   but `run_toy_categorical_optimizers.py` does not expose it. Left alone because that file is
   part of the harness that produced the study's numbers.
3. **`flash_attn`** for ToSFiT — needs a long local CUDA compile; `sdpa` works meanwhile.
4. **Upstream benchmark data** (HPO-B for FSBO, AdaBoost/SVM/LCBench grids for RGPE,
   `gs://gresearch/optformer` checkpoints) — only needed to reproduce each paper's own
   figures, not for our four scenarios.
