# Query BOLT warm starts + LOLBO

This is an isolated continuation on the same frozen 200-query held-out split as
the active Query campaign. It imports the already measured 30-plan BOLT pool for
each task, then runs 80 LOLBO rounds with a batch size of 10 (800 new plans per
task). The existing ORPO+FSBO experiment keeps its own ledger and database
instances.

The optimizer uses the frozen query VAE posterior-mean representation and the
CPU LOLBO DKL-GP / BoTorch Thompson-sampling implementation. Censored SQL results
remain in the audit history but do not train the GP or update incumbents. Every
proposal batch is durably journaled before evaluation; an interrupted SQL call
is not replayed automatically.

Five additional private PostgreSQL clones, worker slots 24-28 on
LLMOptimizer4, are reserved for this run. The run starts with a real 10-plan
round on the first held-out query; that round is part of the 80-round result.

Run `python run_lolbo.py status` for progress. The source BOLT pool and VAE
checkpoint are hash-pinned in `recipe.json`.

Dedicated VAE capacity is running on LLMOptimizer3 GPU5, CPU63, port21900,
using the same hash-pinned checkpoint and a 2% GPU allocator cap. Private SSH
forwards expose it as `http://127.0.0.1:21900` on LLMOptimizer4. Set
`QUERY_LOLBO_VAE_ENDPOINTS=http://127.0.0.1:21900` when launching the controller;
this endpoint is preferred for each decode, with the original pool as fallback.

The dedicated-endpoint canary on 2026-11-18 failed at proposal generation:
20 retries of 10 latents yielded only 2 distinct feasible novel plans. It made
zero SQL measurements. `PAUSE` remains present, and the five-worker campaign
has not launched. Preserve the existing SQLite journal when fixing proposal
generation; rerun the ten-measurement canary before launching all workers.
The service health and journal audit are in `evidence/gpu3_canary_report.json`.
