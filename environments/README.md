# Captured environments

These are installed-package exports from 21 November 2026, not guesses from
Python imports. Each Python environment has `installed.json`, exact
`requirements.txt`, `pip-only.txt`, `direct-installs.json`, and, for Conda,
`conda-explicit.txt`. The latter pins Linux package builds and download URLs.
`index.json` records original prefixes, Python versions and overlay parents.

| Environment | Role | Python | Key versions |
| --- | --- | --- | --- |
| `bolt_clean_venv` | Modern peptide VAE/FSBO, query optimizer and services | 3.11.15 | torch 2.5.1+cu124, gpytorch 1.15.2, botorch 0.18.1, numpy 1.26.4 |
| `openbox_qwen_eval` | Parent of the above overlay venv | 3.11.15 | Same modern stack; transformers 5.6.0 |
| `llamafactory_qwen35` | ORPO/GRPO and warm-start training | 3.11.15 | torch 2.5.1+cu124, transformers 5.6.0, LLaMA-Factory 0.9.5.dev0 |
| `bl_fsbo_exact` | Legacy FSBO subprocess | 3.9.25 | torch 1.11.0+cu113, gpytorch 1.6.0, numpy 1.22.3 |
| `bl_lolbo_exact` | Legacy LOLBO baseline | 3.9.25 | torch 1.11.0+cu113, botorch 0.6.3.1 |
| `bl_bora` | BORA baseline | 3.11.16 | torch 2.4.1, numpy 1.26.4 |
| `bl_rgpe` | Legacy RGPE reference environment | 3.7.16 | torch 1.5.0, gpytorch 1.1.1 |
| `bl_semanticopt` | SemanticOPT baseline | 3.11.16 | torch 2.12.1, transformers 5.5.0 |
| `bolt_qp` | Auxiliary native query baseline environment | 3.12.14 | torch 2.14.0+cu126 |
| `pg16` | PostgreSQL binaries | — | PostgreSQL 16.15 |

Keep the modern and legacy stacks separate: their GPyTorch, NumPy, Torch and
checkpoint interfaces differ. In particular, the legacy FSBO code depends on
NumPy APIs removed in later releases. A driver supporting the required CUDA
runtime and the target GPU architecture is needed. The captured local server
has four A800 80 GiB GPUs and NVIDIA driver 580.65.06; remote campaign hosts have
different allocations recorded in their policies.

## Preferred: archive the current environments

Run on the old server, choosing storage outside Git with enough free bytes:

```bash
python migration/pack_environments.py --output /root/data/discreteDomains-env-packs
# Inspect the plan, then create the archives:
python migration/pack_environments.py --output /root/data/discreteDomains-env-packs --apply
```

The default packs the modern parent, overlay, training, legacy FSBO and
PostgreSQL environments. Add `--names ...` to include the optional baseline
environments. Archives and their SHA256 manifest remain outside this repository.
The older archives in `/root/data/env-packs` are from September; they are not
an exact replacement for these November environments.

On the new server, extract each archive into the **original prefix** recorded
in its manifest. For each Conda environment, then run `PREFIX/bin/conda-unpack`.
Extract the overlay after restoring its `openbox_qwen_eval` parent. The overlay
uses `--system-site-packages`, and its executable depends on that parent path.
Packing ignores editable-project checks because those sources are restored
from Git. Re-register LLaMA-Factory in the two parent environments after restore:

```bash
/root/data/cys/envs/openbox_qwen_eval/bin/python -m pip install --no-deps -e /root/data/cys/LLMOptimizer/LlamaFactory
/root/data/cys/blockdata_archive/miniconda3/envs/llamafactory_qwen35/bin/python -m pip install --no-deps -e /root/data/cys/LLMOptimizer/LlamaFactory
```

## Alternative: rebuild from specifications

With Conda available on the new server, the helper previews each environment
creation and refuses to replace an existing prefix:

```bash
python migration/rebuild_environment.py openbox_qwen_eval
python migration/rebuild_environment.py openbox_qwen_eval --apply
python migration/rebuild_environment.py bolt_clean_venv --apply
python migration/rebuild_environment.py bl_fsbo_exact --apply
python migration/rebuild_environment.py llamafactory_qwen35 --apply
python migration/rebuild_environment.py pg16 --apply
```

Restore the source layout first. Direct archive installs, including the legacy
cu113 Torch wheel, are recorded separately from package-index installs.
Conda build-host `file://` entries belonging to Conda-installed packages are
provenance and are not project directories to recover. The rebuild path has
not been executed on a clean server; old wheels, compiled libraries or package
URLs may require the current archives instead.

PostgreSQL also requires the `pg_hint_plan` extension matching the running
PostgreSQL major version. The Conda package spec alone does not install it.
Verify `LOAD 'pg_hint_plan'` in the restored query database and record its
extension version. If the extension was installed outside the packed prefix
on a remote host, transfer/build it there separately.
