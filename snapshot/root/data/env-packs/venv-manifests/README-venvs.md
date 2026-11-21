# The two non-conda venvs

These are **thin overlay venvs** built on top of the conda envs. `conda-pack`
cannot handle them (they are stdlib `venv`, not conda), so they are shipped as
plain tarballs plus the notes below.

## bolt_clean_venv  (7.4M packed, 29M raw)

- Parent env: **openbox_qwen_eval**, with `include-system-site-packages = true`
  — i.e. it sees everything in `openbox_qwen_eval` and adds almost nothing.
- `bolt_clean_venv.requirements.txt` is **intentionally empty**. That is not a
  failure: the only dist-info in its own `site-packages` is pip + setuptools.
  The `dotenv/` and `wandb/` directories present there carry no package
  metadata, so `pip freeze --local` correctly reports nothing.
- Its real payload is **`sitecustomize.py`** — a compat shim making upstream
  BOLT's `approximate_gp.py` work on `gpytorch >= 1.12` by restoring
  `gpytorch.lazy.LazyTensor`. Losing this file breaks BOLT runs, so keep it.
- Referenced by: `BOLT/launch_ceb3k_fleet.sh`.
- `/root/data/cys/blockdata_archive/bolt_shims/` is **empty** — nothing to migrate from it.

## fanova  (148M packed, 513M raw)

- Parent env: **hpo_benchmark**, `include-system-site-packages = false`
  (fully isolated), so it must carry its own numpy/matplotlib/etc.
- `fanova.requirements.txt` lists all 24 packages; this venv is cleanly
  reproducible from that file if relocation is ever a problem:

      <hpo_benchmark>/bin/python -m venv fanova
      fanova/bin/pip install -r fanova.requirements.txt

  Note `fanova` needs a C toolchain to build; prefer relocating the tarball.

## Relocating them

Deploy the parent conda env first, then run `fixup_venvs.sh` (in the parent
directory) so `pyvenv.cfg` and the `bin/` shebangs point at the parent's new
location. The original `pyvenv.cfg` files are saved here for reference.
