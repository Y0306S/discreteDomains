# TaskSet Completed-Run Archives

The TaskSet CPU campaign writes each completed run to
`<output_root>/<task_id>/<method>/seed1.tar.gz`. The archive uses the verified
prototype's version-1 manifest format. Its members include:

- `run/`: all history, optimizer checkpoints, pending state, trajectories and oracle evidence.
- `runtime/`: the job's runtime source, generated configurations and other runtime files.
- `job.log` and `completion.json`: the original job log and completion record.
- `campaign/`: split, source hashes, baselines, execution plan and runner provenance.
- `manifest.json`: names, types, modes, timestamps, sizes and SHA-256 hashes.

Active runs retain their rotating checkpoints and raw evaluator files. They are
not packed mid-run. Budgets and scoring remain 100 evaluations, 1,000 training
steps per evaluation, one replica, and `min_normalized_valid_outer`.

Completed data occupies one archive. The small existing completion JSON remains
as an operational index containing the archive path and checksum. Existing job,
worker and core lock files remain in place. Shared datasets, environments,
campaign source and model checkpoints are external dependencies; runtime symlinks
are archived as links and never traversed during cleanup.

## Publication and Recovery

The packer holds the same exclusive job lock as the evaluator. It verifies job
identity, the completed budget and recorded success counts. Both clean completion
and completion with recorded evaluation failures can be archived; failed outcomes
are preserved.

It writes a temporary archive, verifies every member and its SHA-256 hash, checks
that source files have not changed, fsyncs, then atomically publishes the archive.
Only then can cleanup remove job-owned originals. Before cleanup, every remaining
source is compared with the manifest. Unexpected or changed files stop removal.
An interruption during cleanup is recovered by rerunning the packer; the published
archive remains authoritative throughout. Corrupt archives raise an error rather
than being interpreted as an empty run.

The production campaign is
`/root/data/logs/taskset_major_families_cpu_20260929`.
Use its deployed command with
`/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python`:

```bash
python code/examples/benchmark/domains/archive_taskset_campaign.py pack \
  --campaign /root/data/logs/taskset_major_families_cpu_20260929 \
  --job test.random.mlp_vae_family_seed19 --cleanup

python code/examples/benchmark/domains/archive_taskset_campaign.py verify \
  --archive test/mlp_vae_family_seed19/random/seed1.tar.gz

python code/examples/benchmark/domains/archive_taskset_campaign.py restore \
  --archive test/mlp_vae_family_seed19/random/seed1.tar.gz \
  --destination /tmp/taskset-recovered-example
```

Run these commands from the campaign directory. Omitting `--job` processes the
backlog, skipping locked, incomplete and already-compacted jobs. SIGTERM/SIGINT
asks the migration process to finish its current archive and then stop. `restore`
requires a new destination and verifies the restored files. It does not overwrite
a live run. The recovered `run/` directory includes the original optimizer state;
the `runtime/` directory includes its evidence and symlinks.

## Readers

```python
from pathlib import Path
from mindware.domains.history import iter_history_paths, load_openbox_history

for logical_path in iter_history_paths(Path("test")):
    history = load_openbox_history(logical_path)
```

The loader accepts a logical `.../seed1/history.json` path or the archive itself.
The former may no longer exist as a loose file. Readers must use the loader and
archive-aware discovery rather than `Path.read_text()` or raw `history.json`
globs. `read_archive_member()` provides verified access to individual evidence
files. TaskSet campaign status, completion checks, the BO CLI, the queue helper,
split audit and winner rescoring use the archive-aware readers.

Completed archives are immutable. The BO CLI skips their completed budgets and
refuses to extend an archive in place. Active-run resume retains the existing
checkpoint and pending-configuration logic.

## Deployment Evidence

`distributed/archive_rollout_20261002/` contains the deployment backup, source
hash revisions, pre-restart observations, post-restart verification and migration
logs. `migration.json` identifies the single low-priority background packer.
The distributed plan enables automatic packing after each new completion.

Verification covered 14 archive tests and four existing TaskSet protocol/resume
tests, including random pending-evaluation recovery, TPE continuation and LOLBO
GP/RNG/pending-batch recovery. A real 100-observation run was fully restored and
recognized as complete by the evaluation CLI. The 542 completed histories at
deployment matched their original SHA-256 hashes after the worker restart.
