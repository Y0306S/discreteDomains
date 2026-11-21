# QueryOpt Random Search Storage Recovery

On October 15, 2026, `LLMOptimizer4.random01` and `random03` were recovered
after PostgreSQL returned DiskFull (`53100`) at proposal indices 207 and 111.
The 207 and 111 earlier completed proposals were preserved and checksummed.

## Storage Change

Eight private databases shared the 128 GiB `/dev/shm` filesystem. Their base
data occupied about 72 GiB, leaving temporary SQL spill files competing for
the remaining capacity. The failed statements had no saved evaluation result.

Each database now has the `query_random_disk_temp` tablespace under:

`/root/data/cys/queryopt_temp/LLMOptimizer4_random_20261015/<worker>`

`ALTER DATABASE imdb SET temp_tablespaces` applies this location to new
connections. The database contents remain in tmpfs. No running query was
terminated to apply the setting. PostgreSQL removes its temporary query files
when their sessions or statements finish; these are not permanent run artifacts.

All eight databases passed a real external-sort test writing approximately
21 MiB to disk, plus a check that temporary relations use the new tablespace.
Probe-specific memory, parallelism and timeout settings were confined to the
probe connections.

The VAE, random sampler, seed, search space, 49 initial observations, 1,000
proposal budget, 100-second SQL timeout and database work-memory settings are
unchanged. Shared-storage I/O can change runtimes for spill-heavy SQL queries.
Per-worker records identify the task, saved count and time at the change.

## Recovery Evidence

- `before.json`: original controller, worker states and available storage.
- `*.storage.json`: per-database settings and disk-spill verification.
- `storage_policy.json`: deployed policy and runtime-comparison caveat.
- `*.recovery.json`: original failed attempt, task claim and completed checksums.
- `verification.json`: most recent worker liveness and replay verification.

Only the two failed proposals were authorized for replay after verifying their
DiskFull errors, completed call timestamps, missing results and absence of
other database clients. Their original error attempts remain in the per-task
SQLite journals. The existing runner performs the replay from its checkpoint.

The handover controller was briefly restarted to serialize the two launches.
The other workers and PostgreSQL instances were not restarted. The handover
controller continues to own these workers and their original CPU reservations.

Run the read-only result verification on LLMOptimizer4:

```sh
/root/data/cys/envs/bolt_clean_venv/bin/python \
  /root/data/cys/logs/query_random_storage_20261015/recover.py verify
```

`configure` installs or revalidates the tablespaces. `restart` is specifically
for the two reviewed failures and deliberately refuses when their saved state
has moved on; it is not a general restart command.
