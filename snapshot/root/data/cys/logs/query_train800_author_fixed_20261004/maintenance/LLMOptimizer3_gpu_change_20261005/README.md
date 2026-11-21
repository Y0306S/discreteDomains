# LLMOptimizer3 Paused For GPU Replacement

## Resumed October 5 At 14:33 Shanghai Time

The user authorized restarting the paused worker after the hardware change.
LLMOptimizer3 now has four A800-SXM4-80GB GPUs, a 40-core CPU quota, and a
400 GiB memory limit. The database archive checksum and restored contents
matched; all 3,168 pinned campaign files and 421 saved artifacts were verified.
The existing Python package versions and a CUDA matrix operation passed validation.

LLMOptimizer3.w00 is running on GPU 0, with BO/supervisor CPU 0 and PostgreSQL
CPUs 1-3 on port 5435. Supervisor PID: 233; optimizer PID: 234; PostgreSQL PID: 220.
It resumed CEB_11B16 from 420 saved proposals and committed its next batch,
reaching 430. The first ten new rows include two new database calls and eight
cache hits; all contain finite positive runtimes. The original 420 proposal
files remain unchanged. No additional workers were launched; GPUs 1-3 were idle.

See database_restored.json and worker_resumed.json for verification. The GPU/host
change occurred after proposal count 420; retain this boundary when comparing
runtime measurements within the resumed task. The records below describe the
earlier shutdown and are preserved as its history.

User requested stopping this machine before changing it to four A800 GPUs.
The shutdown operation itself did not perform a restart or hardware change.

## Stopped Work

- QueryOpt supervisor LLMOptimizer3.w00, PID 20140, and optimizer PID 50085.
- Current task CEB_11B16: 420 of 1,000 proposals durably saved, plus 49 Bao initial observations.
- The checkpoint contains all 420 completed proposals and no pending batch.
- No unresolved SQL evaluation remained at the stopping boundary.
- Dedicated PostgreSQL service on port 5435, PID 18808, cleanly shut down.
- Old disk-usage scan and its sort/tail pipeline, process group 41274.

The worker and claim are marked paused with the user's hardware-change reason.
Readable task history/status are also marked paused. The normal supervisor's
SIGTERM exception is an intentional stop, not evidence of a new experiment failure.
Other machines' workers and training jobs were not stopped.

## Preserved State

Results and checkpoint:

`/root/data/cys/logs/query_train800_author_fixed_20261004/outputs/CEB_11B16/lolbo/seed0/`

The checkpoint loaded successfully on CPU. Its saved observations match the
individual proposal files; the trained GP parameters are finite and the RNG
state is present. No evaluations, model updates, or GPU initialization were
performed during verification. GPyTorch's three positive constraints retain
their expected infinite upper-bound buffers.

The live database was in volatile RAM at
`/dev/shm/queryopt_pg_20261004/pg_imdb_d`. Its cleanly shut-down contents were
archived to persistent shared storage:

`pg_imdb_d.clean_shutdown.tar`

- Size: 9,318,154,240 bytes, approximately 8.68 GiB.
- SHA-256: `21224ca680ecf5b8e8be8686132ec94b5444a1e24cfd44e35431e942c588a634`.
- GNU tar compared the archive with the stopped database successfully.
- PostgreSQL control data reports `shut down`.
- A stale Unix socket was omitted by tar; sockets are recreated at server startup.
- The older source database at `/root/data/cys/pg_imdb_d` was preserved.

Before-stop snapshots, proposal checksums, checkpoint verification and database
archive verification are recorded in the adjacent JSON files.

## Before Resuming

1. Confirm the new SSH endpoint/hostname, shared-storage mount, CPU quota,
   memory limit, four A800 devices, and available RAM-backed storage.
2. If the volatile database directory was lost, verify the archive checksum
   and restore it under `/dev/shm/queryopt_pg_20261004`, preserving numeric ownership.
   Inspect an existing target before restoring; never overwrite a live database.
3. Revalidate Python/PostgreSQL environments and database libraries. Shared
   code, author checkpoint, task split, results, and proposal budget stay the same.
4. Review the machine's CPU/GPU allocations and update host identity if the
   resized instance changes it. The old worker used GPU 0, BO CPU 0 and DB CPUs 1-3.
5. Start and validate its dedicated database before resuming the same worker
   and task from the saved checkpoint. Do not start a fresh task or delete results.

No automatic restart was configured. The controller's existing read-only
monitor may report this worker as stopped or the resized host as unreachable;
the pause records identify this as scheduled maintenance.

## Final Remote State

After the experiment jobs and backup finished, only SSH, platform shells and
an idle Jupyter server remained. No notebook kernel or experiment process was
present. All eight existing A16 GPUs reported zero memory use and zero utilization.
Filesystem writes were flushed with sync.
