# Query ORPO + FSBO: RAM storage remediation

The user authorized greater RAM use to reduce Lustre input/output errors and short reads on 18 November 2026. The campaign is `query_our_fsbo_vae80_test200_20261117`.

## Changes

| Host | Database placement | Temporary query files |
|---|---|---|
| LLMOptimizer2 | Six databases moved from Lustre to `/dev/shm/query_orpo_fsbo_20261118/w0`–`w5`; 52.32 GiB total | RAM, inside the new database directories |
| LLMOptimizer4 | Fourteen existing databases remain on RAM; four Lustre databases now have 10 GiB PostgreSQL shared buffers each, preloaded with their tables, indexes and TOAST data | Existing Lustre tablespaces retained |

Twenty of the 24 database instances are now RAM-backed. The remaining four have explicitly preloaded database caches. They still depend on Lustre for uncached reads, writes and restart recovery.

LLMOptimizer4's 128 GiB RAM filesystem had only 9.97 GiB free, so additional full database copies or unrestricted spill files would risk exhausting it. Enlarging PostgreSQL's memory caches uses the host's remaining memory without filling that filesystem. After preparation its cgroup used approximately 191.91 of 256 GiB. LLMOptimizer2 used approximately 211.07 of 400 GiB, with 147.68 GiB free on its 200 GiB RAM filesystem. These are observations before resumed workload growth, not permanent reservations.

## Verification and preservation

- All optimizers were drained and the existing recovery scheduler was held during migration.
- Every copied file was checked with SHA-256 against its stopped original; PostgreSQL control data matched. Originals remain intact at their original Lustre paths.
- All 24 database identities, ports, CPU assignments and hint support were checked.
- Reads of the three previously failing blocks passed on all 24 databases. Full counts of `name`, `cast_info`, `movie_info` and `title` matched on the six migrated and four newly cached databases.
- All 26 optimization SQLite journals passed integrity checks and were backed up locally. Their 14,604 committed optimization measurements, plus 6,000 warm-start measurements, account for all 20,604 saved measurements. Warm-start result files were also backed up.
- Only the three reviewed historical storage errors were archived after validation. The recovery policy still holds on an unexpected corruption error. Unscored interrupted operations follow the existing quiescence audit; no score is inferred.
- Models, SQL, proposal journals, saved scores, batch size and round budget are preserved. The resource allocation hash was updated with a recorded storage-change boundary.

## Interpretation and remaining dependencies

RAM reduces the database reads and local temporary writes exposed to the failing shared filesystem. Successful validation establishes that the checked data was readable at the time; it does not establish that Lustre faults are repaired.

Remote query spill files and durable score/proposal journals remain on Lustre. RAM database copies and preloaded caches are volatile. Original database copies and durable result journals remain available, and RAM copies can be recreated after a machine restart.

Changing storage and cache state can change measured query runtime. Analysis must disclose the boundary at 20,604 saved measurements and must not treat the resulting timings as a uniform storage configuration. Exact provenance is recorded in `storage_boundary.json` and the campaign's `evidence/ram_storage_boundary_20261118.json`.

The existing five-minute recovery watchdog on LLMOptimizer4 is retained; no duplicate scheduler was created. It performs checks and bounded recovery and does not send future chat notifications.

Resume verification is recorded in `resume.json`; a subsequent live-progress observation is recorded in `post_resume.json` when available.

## Verified resumption

At the post-resume check, all 24 workers were alive and all 24 databases were reachable. The campaign was running with 20657 saved measurements (53 new measurements since resuming). No new task error files were present. See `post_resume.json` for timestamped evidence.
