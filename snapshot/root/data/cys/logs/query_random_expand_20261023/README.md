# Random Search Capacity Expansion

The user authorized additional QueryOpt Random Search workers using freed ML
capacity. The 132-task ToSFiT campaign is complete. Its five remaining idle
model actors were checked for idle health and stopped with PID/start/boot identity
checks. Completed ML results are preserved.

| Machine | Additional workers | CPU allocations (optimizer first) | Total query workers |
| --- | ---: | --- | ---: |
| LLMOptimizer2 | 3 | 0,1,2,3; 4,5,6,7; 16,17,18,19 | 9 |
| LLMOptimizer3 | 1 | 0,1,2,28 | 9 |
| LLMOptimizer4 | 3 | 0,1,2,3; 4,5,6,7; 8,9,10,11 | 11 |

Each worker has one optimizer CPU and three private PostgreSQL CPUs, with no
physical core shared between worker allocations. Machine2/3 use 36 worker cores
within their 40-core quotas. Machine4 uses 44 of 48 physical cores, within its
64-core quota. SMT siblings are not counted as additional physical capacity.
These CPU leases supersede the completed ToSFiT allocations; do not restart its
old actors or schedulers onto the leased cores.

The new workers drain the existing locked test200 Random Search queue. They use
the same author VAE, seed0, 49 Bao initial observations, 1000 additional unique
recovered query plans, decoding recovery, 100-second SQL timeout and per-task
SQLite artifact store. Existing healthy workers and their pinned numerical
runner were not changed. Machine2/3 use their GPUs for the VAE; machine4 uses
the same CPU VAE execution as its existing Random Search workers.

Each worker restores a checksum-verified private database to tmpfs. SQL spill
uses its own shared-NVMe tablespace, verified with a real disk-spilling sort.
Spill-heavy query timing can depend on shared-storage load. Database settings
remain shared_buffers=1GB, work_mem=64MB, and two PostgreSQL parallel workers.

Every new worker must complete the existing 20-proposal smoke test, pause after
10, reload its checkpoint and finish 20 before claiming production work. Failed
validation stops that worker for inspection. Smoke results are outside the
production task queue. Successful workers automatically proceed through the
remaining queue and stop their private databases when it drains.

`expand.py` provides prepare/launch/worker/status entry points. `manifest.json`
pins sources; allocations, launch records, retired actor identities, storage
checks and smoke records preserve provenance. `audit.py` verifies live methods,
CPU affinities, physical-core ownership, cgroup capacity and storage headroom.
The per-host audit JSON files are snapshots, not continuous monitors.
