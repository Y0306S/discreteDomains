# Query SemanticOPT and BOLT resource plan

Updated November 23, 2026. Status: launched and verified. SemanticOPT runs 20 workers with 20 full-context LLM replicas; BOLT runs seven workers with eight VAE services after passing its SQL canary. Actual campaign paths and initial results are recorded in launch.json and LAUNCH.md.

## Budgets and method

- SemanticOPT: retain the existing 200 proposals per query; continue all 200 queries to 400 proposals (40 batches of 10), with the fixed first ten continuing to 800 proposals (80 batches). This requires 44,000 additional SQL evaluations. The original 49 Bao observations remain separate from the proposal budget. Preserve the original model, generation settings, validation and constrained-random top-up policy.
- BOLT: reuse exactly the saved 30 measured plans per query, then add 770 evaluations to reach 800 total (80 batches including the three warm-start batches). Across 200 queries this is 154,000 additional evaluations. The older paused BOLT recipe instead specified 800 NEW evaluations, or 830 including warm starts; the new isolated recipe must explicitly correct this to 77 continuation batches.
- BOLT here means the latent BO/LOLBO continuation after LLM warm starts. The earlier version of this proposal incorrectly reserved two live full-SFT LLM replicas for BOLT. The full-SFT checkpoint is provenance for the saved 30 plans; regenerating those plans or retaining that LLM service is unnecessary.
- The available continuation uses the frozen, train80-finetuned query VAE and CPU DKL-GP/Thompson sampling. It is not automatically an implementation of every end-to-end VAE update in upstream BOLT. Pin and report this variant explicitly; FSBO-specific algorithm changes must not silently become BOLT changes.

## CPU and PostgreSQL allocation

Live cgroup quotas are 40, 64 and 28 CPU equivalents: 132 total. Each SQL task receives four CPU slots: one coordinator/optimizer and a PostgreSQL leader plus two parallel workers. GPU service CPU usage must fit the service budget too. CPU identifiers in proposal.json are candidate affinities, not proof of available physical cores.

| Host | CPU quota | SemanticOPT task workers / private DBs | BOLT task workers / private DBs | Task CPU slots | Service and reserve CPU slots |
|---|---:|---:|---:|---:|---:|
| LLMOptimizer2 | 40 | 8 | 0 | 32 | 8 |
| LLMOptimizer4 | 64 | 12 | 3 | 60 | 4 |
| LLMOptimizer3 | 28 | 0 | 5 target; 4 initially | 20 target; 16 initially | 8 target; 12 initially |
| Total | 132 | 20 | 8 target; 7 initially | 112 target | 20 target |

This provides exactly 20 independent SemanticOPT databases and up to eight additional BOLT databases. Reuse existing exclusive clusters. Existing completed-hybrid slots provide ten local, 23 on Optimizer4, and four on Optimizer3; the eighth BOLT worker requires a fifth usable Optimizer3 cluster. Admit it only after memory and CPU measurements; otherwise stay at seven. Never have two active tasks sharing a private DB.

Optimizer4 currently has only 9.9 GiB free in /dev/shm. Reuse clusters rather than creating more RAM clones there. Stop/release surplus clusters through their owning lifecycle only after checking ownership. Preserve database settings, the 50-second SQL cap, and storage behavior for comparable measurements. Maintain at least 12 GiB RAM headroom per host, including actual resident/shared database memory and inference host memory.

## GPU services: 20 LLM replicas plus eight BOLT VAE replicas

| Host / GPUs | Planned services | Initial execution policy |
|---|---|---|
| Optimizer2, A800 GPUs 0-3 (80 GiB each) | Three SemanticOPT query LLM replicas per GPU: 12 total | Admit up to two ordinary generations per card initially; serialize large memory requests; increase to three only after peak-memory and throughput checks |
| Optimizer3, sixteen A16 GPUs (15 GiB each), eight existing GPU pairs | One SemanticOPT query LLM replica per pair: 8 total | Validate the original query context/output envelope on the pair; route oversized requests to A800 without shortening the prompt |
| Optimizer3, one card in each pair | One BOLT VAE service per pair: 8 total | One decoder owner per BOLT worker; increase parallel decoding only when queue time justifies it |

The A16 pairs are [14,15], [1,4], [2,6], [0,3], [7,8], [9,10], [11,12], [13,5]. Candidate VAE cards are 15,4,6,3,8,10,12,5. Final placement depends on measured memory after the old model services exit. Existing hybrid services have demonstrated 12 resident replicas on A800 and eight paired replicas on A16, but SemanticOPT has different output/context/cache requirements; this is a capacity target, not a passed SemanticOPT canary.

There are 20 task workers and 20 loaded LLM replicas, but not necessarily 20 simultaneous generations. Route consecutive requests to the cache-owning replica. Keep SemanticOPT's 1.5 GiB retained prompt cache and original generation envelope. The old service configuration admits 15,360 input tokens and up to 16,384 output tokens; derive actual generation settings from the pinned recipe, rather than substituting the hybrid's 1,024-output envelope. Pair sharding, GPU memory admission and overflow routing require an isolated, fingerprinted service copy. If a pair cannot satisfy the envelope, reduce replica count and share validated A800 services rather than changing the experiment's prompts.

BOLT's current DKL-GP fitting and acquisition are on CPU. More GPU VAE replicas do not move GP computation to GPU. Begin with this unchanged optimizer. A separately validated GPU GP implementation is optional only if profiling shows acquisition dominates; match results within appropriate numerical tolerances and record backend changes.

Eight A16 LLM services plus eight VAE services share Optimizer3's service CPU budget. Cap active generation/decoding work to that budget, leaving room for the broker and SSH. Do not assign one whole simultaneously busy CPU to every service process. If VAE work is CPU-bound, reduce A16 LLM concurrency or move decoder replicas onto A800 after memory checks; keep the 20 SemanticOPT SQL workers fed. On Optimizer2 likewise fit 12 replicas into the eight service CPU slots through measured concurrency/admission. Enforce aggregate budgets rather than relying on affinity alone.

## BOLT readiness issue

The previous BOLT continuation is paused and has zero new SQL measurements. Its canary produced only two distinct feasible novel plans from 200 sampled latents, short of the required ten. The service reported zero transport/model errors: the failure was proposal acceptance, not simply insufficient VAE workers. The canary measured about 437.6 seconds of decoding execution versus 3.95 seconds of queue time across 212 calls, so adding workers alone cannot repair this acceptance problem.

The existing client also sends one latent at a time, sequentially, and always prefers the same dedicated endpoint. Merely adding URLs will not use the new pool efficiently. Before scaling, dispatch independent decodes concurrently in bounded batches, assign endpoints across workers, and preserve per-latent seeds and response ordering. Do not batch together RNG streams in a way that silently changes sampling semantics. Audit the latent scale, canonicalization, cross-join rejection, duplicates and fallback flags to locate the acceptance failure. Any proposal-policy change needs its own recipe/version and an explicit method description.

BOLT must complete a real ten-new-plan batch with valid, unique, non-fallback plans and durable SQL results before admitting the full pool. Retain failures and rejected attempts in the audit. Do not substitute saved plans or label decoder fallback as a newly successful proposal. SemanticOPT can start independently while BOLT's proposal issue is resolved.

## Handoff and scheduling

1. The hybrid has completed 200/200 tasks and 160,000 total measurements. Freeze its completion report, verify no unresolved SQL calls, then retire its monitor/recovery lifecycle and inference services before assigning the same GPUs or DBs. Resident old services currently occupy roughly 60 GiB per A800 and 7.7-12 GiB per A16 despite zero sampled GPU utilization. Check ownership and preserve unrelated workloads.
2. Create an isolated SemanticOPT continuation. Back up all 200 SQLite checkpoints, preserve the exact 200-record prefix, Bao pools and RNG state, and verify no pending evaluations. Amend the explicit budget policy and fingerprints rather than editing the original campaign in place.
3. Load and canary the SemanticOPT pool, then admit all 20 SemanticOPT workers. Give the fixed first ten priority through 400 and 800; keep the other ten workers progressing through the remaining queries, scheduling historically slow tasks early. A task has at most one owner.
4. In parallel, resolve the BOLT acceptance issue using one task/DB, then expand to seven BOLT workers and eight decoder services. Admit the eighth SQL worker only when its fifth Optimizer3 DB and host headroom pass admission. No new 30-plan LLM warm starts are needed.
5. As SemanticOPT queries finish at 400 and the pilot finishes at 800, transfer freed task/DB slots to BOLT. Retire idle SemanticOPT inference services too. The same conservative allocation supports up to about 28 BOLT task workers after the handoff, subject to CPU, memory and decoder capacity. Reprofile decoder concurrency at that stage.

## Measurement, plateau report and ETA

Report separately: LLM attempts, valid unique LLM proposals, constrained-random SemanticOPT top-ups, BOLT latent attempts, first-decode feasibility, feasibility after decoder retries, duplicates, decoder fallback, and successful uncensored SQL evaluations. Keep numerators and denominators explicit.

For the fixed first ten SemanticOPT queries, report best uncensored runtime at 400, 500, 600, 700 and 800 proposals; paired 400-to-800 improvement; geometric mean and median; per-query wins; and incremental SQL, inference and wall time. The first ten are a convenience pilot, not a representative random sample of templates. The historical 40,000 SemanticOPT proposals contain 32,690 constrained-random top-ups, 7,295 labeled LLM proposals and 15 early unlabelled records, so attribution matters when interpreting the plateau. A 1% geometric-mean improvement threshold is only a suggested operational decision rule, not a user-agreed threshold or a significance test.

Historical SemanticOPT SQL duration averaged 31.935 seconds per proposal. At 20 fully occupied SQL workers, 44,000 additional evaluations imply about 19.5 hours of SQL capacity time, before inference, imbalance and service overhead. Publish a real ETA after the first hour of measured continuation throughput. BOLT has no defensible ETA until its proposal canary passes and its novel-plan throughput is measured.
