# Portable envs for the second server

The other server mounts `/root/data` but **not** `/root/data/cys/blockdata_archive`, where these
envs actually live. This directory carries them across.

They are shipped as **`conda-pack` tarballs, not raw directory copies.** Three
reasons that matters:

1. A conda env hardcodes its own absolute prefix into thousands of files
   (every `bin/` shebang, `conda-meta/*.json`, `.pc` files, some RPATHs). A
   plain `cp`/`rsync` to a new path leaves all of those pointing at
   `/root/data/cys/blockdata_archive/...`, which does not exist on the target — so every console
   script fails. `conda-unpack` rewrites them on arrival.
2. It works no matter where the other server mounts the share. Nothing depends
   on the path being identical on both machines.
3. `/root/data` is Lustre at 97% full. The five envs are **418,487 files /
   28G** raw; as tarballs they are **7 files / 13G**. Copying and storing 7
   large files on a near-full network filesystem is dramatically kinder than
   400k small ones.

## Contents

| Pack | Raw | Packed | Notes |
|---|---|---|---|
| `packs/openbox_qwen_eval.tar.gz` | 8.0G | 3.9G | env of the 4 live protein/BOLT jobs |
| `packs/hpo_benchmark.tar.gz` | 7.3G | 3.7G | |
| `packs/llamafactory_qwen35.tar.gz` | 6.9G | 3.5G | |
| `packs/openbox_eval.tar.gz` | 3.0G | 1.2G | |
| `packs/taskset_eval.tar.gz` | 2.3G | 741M | python 3.7 / tensorflow 1.15 |
| `packs/fanova.venv.tar.gz` | 513M | 148M | overlay venv on `hpo_benchmark` |
| `packs/bolt_clean_venv.venv.tar.gz` | 29M | 7.4M | overlay venv on `openbox_qwen_eval` |

`deploy_envs.sh` — unpack + fix the conda envs.
`fixup_venvs.sh` — unpack + repoint the two overlay venvs (run after the above).
`venv-manifests/` — pip requirements + original `pyvenv.cfg` for the venvs.

## Deploying on the other server

```bash
cd /root/data/env-packs

# conda envs -> local disk (preferred; see "Where to unpack" below)
DEST=/root/envs ./deploy_envs.sh

# just one env
DEST=/root/envs ./deploy_envs.sh openbox_qwen_eval

# then the overlay venvs
ENVS=/root/envs DEST=/root/venvs ./fixup_venvs.sh
```

Then either call the interpreter directly:

```bash
/root/envs/openbox_qwen_eval/bin/python your_script.py
```

or register the directory with an existing conda:

```bash
conda config --append envs_dirs /root/envs
conda activate openbox_qwen_eval
```

No conda installation is required on the target for the direct-interpreter path.

### Where to unpack

Unpack to **local disk** (`DEST=/root/envs`), not onto `/root/data`. Python
imports touch tens of thousands of small files, which is Lustre's worst case —
interpreter startup and imports will be markedly slower from the share, and it
would add 28G to a filesystem already at 97%. Use `DEST=/root/data/envs` only
if the target server genuinely has no local scratch space.

## Two pre-existing breakages fixed during this migration

Worth knowing, because both were already broken on the source box:

1. **`import llamafactory` was failing** in `openbox_qwen_eval` and
   `llamafactory_qwen35`. Their `_editable_impl_llamafactory.pth` pointed at
   `/root/LLMOptimizer/LlamaFactory/src`, a path that does not exist. It now
   points at `/root/data/cys/LLMOptimizer/LlamaFactory/src`, which is real and
   — conveniently — on the share, so it resolves on both servers. The originals
   are saved alongside as `*.pth.bak-premigration`.

2. **`pip` and ~80 other `bin/` scripts had a dead prefix.** These envs were
   created through symlinks under `/root/miniconda3/envs/`, so their shebangs
   read `#!/root/miniconda3/envs/<name>/bin/python` — but only `hpo_benchmark`
   and `llamafactory_qwen35` still have such a symlink. In the other three,
   `pip` fails with "required file not found" today. `conda-unpack` alone does
   **not** fix this (it only rewrites the prefix conda-pack recorded), so
   `deploy_envs.sh` does a second rewrite pass and verifies `pip` afterwards.

## What was deliberately NOT moved

Only environments were migrated. The `blockdata` references in your scripts are
overwhelmingly **data** paths, not env paths — of 113,872 references scanned,
just **125** point at envs. The rest point at `llmoptimizer_benchmark/`
(154G of `benchmark_data/`, `history_data/`) and `model-weights/`, which are a
separate, much larger migration question and were left untouched.

`/root/data/cys/blockdata_archive/bolt_shims/` is empty; there was nothing to move.

The source envs under `/root/data/cys/blockdata_archive/miniconda3/envs/` are **untouched and
still working** — nothing was deleted, so the 4 running jobs were never at risk.
