# Moving to another server

This kit prepares the source and records the dependencies. No destination
server was supplied, so it has not transferred artifacts, stopped workloads,
packed the current environments or restored a database. These operations are
explicit steps below. Live experiments on the original machines were left
running.

## 1. Keep the original layout

Use Linux x86_64 and retain the `/root/data/...` and `/root/eb-public/...`
paths where possible. They occur in Python imports, saved recipes, subprocess
commands and checksum manifests. A different layout requires an explicit
configuration migration and revalidation of run/checkpoint identities.
Install Git, Python 3, SSH and rsync on both hosts; the package/restore steps
also need tar, Conda/conda-pack and the matching NVIDIA driver.

Clone the repository into `/root/discreteDomains` on the new server and check it:

```bash
git clone https://github.com/Y0306S/discreteDomains.git /root/discreteDomains
cd /root/discreteDomains
python migration/verify_snapshot.py
python migration/restore_layout.py --destination /
# On the new server, after reviewing the preview:
python migration/restore_layout.py --destination / --apply
```

The restore tool checks every snapshot hash, copies files and refuses conflicts
instead of overwriting existing paths. A scratch preview can use
`--destination /tmp/discreteDomains-stage`. Scripts retain their original
absolute paths, so that scratch directory is for inspection, not execution.

## 2. Transfer the scientific inputs and saved state

`migration/inventory/artifact-roots.txt` lists the non-Git payload roots,
relative to `/`. `exact-artifacts.json` lists existing checkpoint/data file
references and sizes. Original configs contain their scientific hashes.
`primary-inputs.json` records availability of the 25 absolute paths referenced
by the main peptide/query model recipes; all existed on the capture host.
`path-references.txt` is an additional audit list of paths seen in sources,
including historical/output paths; it is not a list of all required inputs.

On the old server, preview the copy to a staging directory on the new one:

```bash
cd /root/discreteDomains
bash migration/transfer_artifacts.sh root@NEW-SERVER:/migration-stage/
# Review storage requirements and then perform the copy:
bash migration/transfer_artifacts.sh root@NEW-SERVER:/migration-stage/ --apply
```

The destination then contains `/migration-stage/root/data/...` etc. No remote
command starts an experiment. Restore those payload paths into the corresponding
original paths on the new server after inspection. Scientific input symlinks
are preserved and their targets are listed explicitly. Existing source/config
files already in the Git snapshot are excluded from the payload copy.

The plan covers the peptide/query campaign directories, source histories,
native query data, author VAEs, FSBO weights, APEX oracle assets, SFT/ORPO
checkpoints and Qwen3.5-9B base. Old variants and comparison inputs are retained
because code and reports refer to them. It can therefore require substantial
storage; the Git tree is only the source/configuration portion. It excludes
machine SSH/API credentials, runtime service tokens and raw PostgreSQL
relation/WAL directories. Provision fresh service credentials on the new hosts.

The transfer helper does not make a global, atomic snapshot of an actively
changing campaign. Copy immutable inputs first. Before the final cutover,
arrange a normal checkpoint boundary, make consistent journal/database backups,
and repeat the final state transfer. Keep one authoritative running copy of
each experiment. Do not treat a copied PID, lock file, `status.json` or old
health report as evidence that a new-server process is healthy.

## 3. Preserve environments

Follow [environments/README.md](../environments/README.md). Prefer fresh
Conda-pack archives of the current environments, with their SHA256 manifest.
The helper previews archive creation until `--apply` is specified. Store and
transfer these archives outside Git, restore the original prefixes and run
`conda-unpack` for Conda environments. Restore the modern base environment
before its `bolt_clean_venv` overlay. Re-register editable LLaMA-Factory source.

Package exports also support a specification-based rebuild if the old server
is no longer available. That rebuild has not been validated on a clean host;
preserving the current archives avoids dependence on old wheel availability.

## 4. Back up SQLite journals consistently

For query state, the SQLite online backup API produces consistent per-journal
copies while the database is readable:

```bash
python migration/backup_sqlite.py \
  --campaign /root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117 \
  --output /root/data/discreteDomains-journal-backup
# Add --apply to create the backup and its checksum manifest.
```

It checks each backup with `PRAGMA quick_check`. Run it on the host that can
access the journals. On storage configurations with host-local journals,
repeat on each worker host. Backups are individually consistent; a running
controller may advance other files after the snapshot. Final resume requires
the matching journal, saved model state, pool, recipe and source pins from
the same checkpoint boundary. Do not fabricate missing observations or relabel
warm-start measurements as new evaluations.

## 5. Restore PostgreSQL/IMDb separately

The current query topology includes PostgreSQL instances with data directories
in `/dev/shm` and remote disk/cache paths. RAM directories disappear on reboot.
Ordinary rsync of live relation/WAL files is not a valid database backup.

Use a logical dump of the authoritative IMDb database, preserving schema,
data, indexes and extension requirements. A single dump can initialize
equivalent worker replicas when their dataset/schema identities agree:

```bash
PGHOST=127.0.0.1 PGPORT=WORKER_PORT PGUSER=imdb PGDATABASE=imdb \
  bash migration/backup_postgres.sh /outside/git/imdb.dump
```

Supply authentication through a private `PGPASSFILE` or the existing server
authentication policy. The helper invokes PostgreSQL 16 binaries, writes a
custom-format dump and records its contents and SHA256. It does not create
server-wide roles. Create the required database user/ownership on the new
server, install the matching `pg_hint_plan` extension, then restore:

```bash
/root/data/cys/envs/pg16/bin/pg_restore --exit-on-error \
  --host=127.0.0.1 --port=NEW_PORT --username=imdb --dbname=imdb \
  /outside/git/imdb.dump
```

The target database must already exist. Recreate the intended tablespaces,
temporary directories and per-worker instances for the new storage layout.
If using a physical backup instead, use PostgreSQL's supported backup and
WAL/recovery procedure for the matching major version.

Check `LOAD 'pg_hint_plan'`, query schema/index identities, codec equivalence,
query correctness, cache/preload policy, timeout and `EXPLAIN` hints before
live evaluation. Runtime measurements on different hardware/storage are a new
measurement environment; record that boundary when comparing trajectories.

## 6. Reconfigure hosts and run validation

Read [docs/EXPERIMENTS.md](EXPERIMENTS.md) for the versioned entry points.
The stored resource allocation names multiple hosts (`LLMOptimizer2`,
`LLMOptimizer3`, `LLMOptimizer4`), GPU indices, CPU affinity sets, ports and
temporary storage paths. Replace them with an explicit allocation for the new
machines before starting a controller. Existing SSH configuration/keys are
not included; configure your own access if using several hosts.

Retain algorithm parameters and training/test splits. Keep the new query
FSBO/VAE features separate from the original retrieval encoder/GP features.
Provision the base Qwen model, exact adapters, native codecs, APEX oracle and
fresh service-token files. If a recipe/code identity must change, issue a new
validated run identity; do not bypass the checksum/resume checks.

Run the CPU checks in the experiment guide, then the campaigns' existing live
preflight, model reload, oracle, transport and checkpoint/resume gates on the
new hardware. Begin with a separately identified smoke run. Historical
validation JSON files preserved here describe the old deployment and do not
certify the new server.

Existing monitor/recovery implementations are preserved in campaign code.
Register their scheduling on the new hosts explicitly after validating
process identities; personal Codex configuration and live scheduler sessions
are not part of this repository.

## GitHub publication

The intended repository is `Y0306S/discreteDomains`, private by default. The
server needs an authenticated GitHub session to create it and push:

```bash
gh auth login --hostname github.com --git-protocol https --web
bash migration/publish.sh
```

The helper checks the account, refuses an uncommitted working tree, creates
the repository if necessary and pushes `main`. It does not force-push. A
local commit remains usable if GitHub authentication is unavailable.
