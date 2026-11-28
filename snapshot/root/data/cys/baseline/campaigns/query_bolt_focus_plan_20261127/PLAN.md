# BOLT focus and LLMOptimizer2 shutdown: draft deployment plan

Prepared 27 November 2026 after live read-only host checks. This document is a
plan; no workers, services or databases have been stopped or reallocated.

## Recommendation

Pause SemanticOPT at durable measurement boundaries, release its 20 live LLM
replicas, retain all saved initialization and continuation state, and devote
LLMOptimizer3 to the frozen query VAE. Run the LOLBO optimizer and SQL on the
CPU-only hosts. Move the controller and service forwarding off LLMOptimizer2
before shutting that machine down.

Start with 16 VAE service processes, one per A16 GPU. Test 24 if decoder queues
remain material. Use 4 CPU equivalents per concurrent SQL/LOLBO task, preserving
the current one-thread optimizer, PostgreSQL leader plus two parallel workers,
50-second SQL limit, checkpoint and counted-evaluation budget. The exact final
task count requires the identities and live resource limits of both CPU-only
hosts. Only LLMOptimizer4 is currently identified as CPU-only.

## Observed capacity and dependencies

| Host | Live cgroup CPU quota | Memory limit | GPU hardware | Current roles |
| --- | ---: | ---: | --- | --- |
| LLMOptimizer2 | 40 CPU equivalents | 400 GiB | 4 A800 80 GiB | Both fleet controllers, 8 SemanticOPT task slots, 12 SemanticOPT LLM replicas, SSH launchers/forwarding |
| LLMOptimizer3 | 28 CPU equivalents | 112 GiB | 16 A16, about 15 GiB each | 8 SemanticOPT LLM replicas on GPU pairs, all 8 current BOLT VAE services, 5 BOLT SQL/LOLBO slots |
| LLMOptimizer4 | 64 CPU equivalents | 256 GiB | No NVIDIA runtime/GPU detected | 12 SemanticOPT SQL slots, 3 BOLT SQL/LOLBO slots |
| Second CPU-only server | Not yet identified | Not yet checked | User reports CPU-only | Admission pending host identity and read-only inspection |

CPU quotas are scheduler CPU-time equivalents, not dedicated physical-core
counts. LLMOptimizer3 exposes 64 hardware threads and LLMOptimizer4 exposes 96;
these listings do not override their 28/64 quotas. Affinity should prefer distinct
physical cores before adding sibling threads, while aggregate usage stays within
quota. A five-second sample used about 10.5 CPU equivalents on Optimizer3 and
5.0 on Optimizer4 without throttling; this short asynchronous sample is not a
capacity or throughput benchmark.

Optimizer3 memory.current was about 97.3 GiB of 112 GiB; removing its eight LLM
replicas will release substantial host RAM as well as GPU memory. Optimizer4
memory.current was about 161 GiB of 256 GiB. /dev/shm was about 44/56 GiB on
Optimizer3 and 119/128 GiB on Optimizer4. Never use the host-level free-memory
display as the container's available allocation, and do not create additional
RAM database clones in these nearly full mounts.

BOLT already uses saved 30-plan warm starts. It needs zero live LLM replicas.
The eight current VAE processes are on Optimizer3 GPUs 15,4,6,3,8,10,12,5,
ports 23300–23307. Their observed NVIDIA resident footprint is about 192–200 MiB
per process and their reported Torch peaks about 72–79 MiB. These measurements
support adding replicas; they do not prove an unlimited throughput benefit.
The original service's allocator policy remains in force.

The VAE checkpoint SHA256 is
eebbe1ee70209aec6b8236d3c01d43c0d3591e52f9bbc23e311c610712e0204d.
The continuation uses that frozen train80-finetuned representation, CPU DKL/GP
and Thompson sampling. This migration does not introduce end-to-end training,
different latent sampling, fallback acceptance, or regenerated warm starts.

## GPU deployment

1. Keep the existing eight VAE processes through the first cutover. After the
   SemanticOPT pause is verified, stop only its owned LLM services and broker.
2. Add eight identical VAE services on the eight uncovered GPUs: 0,1,2,7,9,11,13,14.
   This gives 16 processes, one per physical A16 GPU. Pin each process's CPU
   thread separately and keep host/controller reserve within the 28-CPU quota.
3. Move all five SQL/LOLBO task slots off Optimizer3 before treating it as a
   dedicated decoder host. Until then, the existing 20 task CPU slots plus
   16 busy decoder threads would exceed its quota; do not activate that layout.
4. With SQL elsewhere, use 16 CPU equivalents for the 16 services and up to
   12 for controller, communication, OS and reserve. Measure actual usage.
5. If VAE queueing limits SQL throughput, test 24 services (add second processes
   on eight cards), allowing 24 active decoder threads plus four CPU equivalents
   of reserve. This is a practical CPU-budget ceiling for this layout, not a
   VRAM limit or a measured optimum. Do not promise that 24 beats 16.
6. Additional idle replicas fit GPU memory but do not create additional CPU
   capacity. Admit more active work only with measured headroom and improved
   end-to-end valid-plan/evaluation throughput.

Keep the existing per-vector decode seed and response ordering. Adding endpoints
must actually expand client routing: the current pinned pool lists only eight
URLs. Use a recorded operational endpoint override and audit the decoder
dispatcher; merely starting new services leaves them unused. Balance requests
across the pool and bound aggregate concurrency, while retaining per-service
RNG isolation. Decode requests can take longer than a short /health timeout
because the health handler shares the model mutex; a timeout alone is not proof
of a dead replica.

## CPU worker allocation

A worker has one independent PostgreSQL cluster, one single-threaded LOLBO
process and exclusive task ownership. Keep BLAS/OpenMP/Torch at one optimizer
thread. More optimizer threads do not automatically improve the small-GP work.

For CPU quota Q and reserve R, the initial CPU admission ceiling is:

    workers = floor((Q - R) / 4)

This is only a CPU ceiling. Memory, private database availability, temporary
storage, network and VAE feed rate can lower the admitted count.

| CPU quota on a CPU-only host | Reserve | CPU-based worker ceiling |
| ---: | ---: | ---: |
| 32 | 4 | 7 |
| 40 | 4 | 9 |
| 64 | 4 | 15 |
| 96 | 8 | 22 |
| 128 | 8 | 30 |

On verified LLMOptimizer4: release its 12 SemanticOPT slots and reuse the
existing clusters. Expand BOLT from three to 12 workers, then test 14 and 15.
Fifteen uses 60 CPU equivalents plus four reserved; memory/storage admission
must still pass. Retaining the same slot-to-database mappings avoids unnecessary
copies and permits later SemanticOPT resumption after those databases are idle.

For example, if the second CPU-only host also has a verified 64-CPU quota and
sufficient memory/storage, the two-host CPU ceiling is 30 workers total. This
is conditional, not the currently verified capacity. With only the known
Optimizer4 available, use up to 15 SQL/LOLBO workers plus a dedicated 16-service
VAE host; do not invent a second machine or count Optimizer3 twice.

Keep a 4-CPU per-task baseline to preserve existing SQL parallelism. After
profiling, a separate 3-CPU affinity admission experiment can test whether
optimizer/SQL phase alternation improves aggregate throughput. It must not
silently reduce PostgreSQL parallelism or claim 20/30/etc. workers are faster
solely because they start. CPU-service changes and measured runtime execution
conditions must be recorded.

## Storage and failures before scaling

Six BOLT tasks were paused in the latest inspection. Four have decoder novelty
shortages (including the originally deferred CEB_10A32); two have PostgreSQL
temporary-file I/O errors. More GPU replicas do not fix a neighborhood that
cannot produce ten novel feasible plans. Preserve those partial proposals and
follow the user's deferred-work ordering; do not silently accept duplicates or
fallback plans to fill batches.

For temporary-file failures, check the actual tablespace target and filesystem
on each CPU host. Provision healthy local scratch when available, or a sized
tmpfs with explicit per-host admission limits. Existing root disks have only
about 7.6/11 GiB free on Optimizer3/4, and /root/data is shared Lustre despite
its shared-NVMe backing; neither is automatically suitable for all workers'
spill allowance. The present per-query temp_file_limit is 8 GiB; concurrent
parallel SQL processes, PostgreSQL shared buffers, query work memory, GP RSS,
database RAM copies and OS reserve all enter the memory/storage calculation.
Measure per-worker peak PSS and temp use, then cap simultaneous spill demand.
Do not allocate 15 times a nominal 8-GiB limit without checking the actual
PostgreSQL process-level limits and host budget.

Record any scratch change because storage affects the measured query runtime.
Keep SQL timeout, parallelism and work_mem fixed for the initial migration.

## Checkpoint-safe cutover order

1. Inventory PID/start ticks, active leases, database mappings, current saved
   evaluation counts and pending measurement markers. Back up SemanticOPT's
   SQLite databases with the SQLite backup API, preserving pending batches,
   requests, RNG state and the 800-budget amendment. Do not copy a live WAL file
   as an assumed consistent backup. Store checkpoint and prefix hashes.
2. Set SemanticOPT's global PAUSE and let its evaluator reach its existing safe
   loop boundary after the current SQL/batch commits. Verify no in-flight SQL
   before stopping its owned model processes or repurposing databases. Pause
   may take minutes, particularly with ten 50-second SQL slots per batch or
   an outstanding generation. Do not SIGKILL active measurements to speed it up.
3. Stop/reconfigure any controller that would restart those services. Preserve
   its process identities and the 800-evaluation task states for later resume.
   Keep local databases until all live SQL exits; if the container will be
   destroyed, make a clean DB backup since /dev/shm is ephemeral.
4. At BOLT's full saved round boundary, preserve its existing task journals,
   partial SQL results and proposal records. Reassign worker/database placement
   through an operational allocation amendment; keep pinned scientific recipe
   files unchanged. Audit ambiguous started SQL before any replay.
5. Put the new controller on a surviving CPU host, preferably Optimizer4. Make
   it host-aware: the present fleet assumes Optimizer2 is local and stores
   SSH-launcher PIDs locally, so running it unchanged on another host is unsafe.
   Launch remote workers detached under a host-local supervisor. Verify remote
   PID/start ticks and task locks rather than treating local SSH PIDs as workers.
6. Establish direct CPU-host-to-Optimizer3 VAE connectivity with independently
   owned, restartable forwarding if direct routing is unavailable. Forwarding,
   authentication, health checks and logs must not depend on Optimizer2's SSH
   masters. Preserve the eight existing local URL aliases as needed; add and
   verify the new endpoints explicitly.
7. Copy required local-only operational evidence into shared persistent storage.
   Verify all recipe/model/token/input paths from the surviving hosts. /root/data
   is currently a shared mount; confirm its lifetime is independent of stopping
   Optimizer2. Do not copy local secrets wholesale.
8. Run integrated decode/encode checks with the pinned checkpoint and a live
   continuation batch on a selected existing task, retaining every new SQL
   observation in its journal. Validate duplicate/fallback rejection, counters,
   lease exclusivity, database ownership and isolated failure behavior.
9. Scale VAE 8 -> 16 and SQL workers to the first admitted count, then step up.
   Verify continued saved SQL commits without any Optimizer2 forwarding before
   shutting down Optimizer2. Shutdown is the final operation after all checks.

## Throughput selection

Use fixed probe inputs for decode-only comparisons and record all production
SQL during end-to-end admission. Preserve per-latent seeds and avoid selecting
tasks by favorable performance. Compare VAE 16 versus 24, SQL 12 versus 14 versus
15 on Optimizer4, then add the verified second CPU host in stages.

Measure successful completed SQL evaluations/hour, newly accepted unique feasible
plans/minute, VAE queue/execution latency, GP time, SQL time, CPU throttling,
memory/temp peaks and task failures. Queue-only throughput and GPU utilization
alone are not the objective. Short probes select candidates; observe at least
30–60 minutes of end-to-end behavior per admitted stage before declaring a
throughput winner. Hold or reduce concurrency when work completed per hour
falls, SQL timeouts increase, or resource limits are approached. There is no
measured speedup or reliable completion ETA for the proposed layout yet.

## Information still needed

The SSH alias/address of the second CPU-only server (and confirmation that
LLMOptimizer4 is the first). Inspect its cgroup CPU/memory limits, private
database inventory and scratch storage before giving a final total worker
count. No shutdown, pause or reallocation was performed while drafting this plan.
