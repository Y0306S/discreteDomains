# Peptide FSBO Live Audit

Snapshot: October 29, 2026, approximately 10:24 China time.

## Progress

- FSBO: 0/200 completed; 48 tasks running; 56,350 proposals and 51,624 fresh evaluations.
- No saved-history integrity errors, changed pinned code, or recorded worker crashes were found.
- The completed LOLBO and TPE campaigns each contain 200 completed-task archives.

## Findings

1. **Target fitting dominates elapsed time.** GP adaptation accounts for 97.1% of the recorded
   FSBO round time. Of 1,127 completed fits, 1,061 reached the 10,000-step maximum. The median
   latest fit took 4,565 seconds, approximately 76 minutes, for one batch of 50 proposals.
   The released FSBO consecutive-loss patience rule and 10,000-step limit are retained.
   GPyTorch 1.6.0 switches away from Cholesky above its 800-point threshold and uses ten
   stochastic trace samples, with deterministic probes disabled. A controlled fixed-parameter,
   850-observation GP reproduced a loss range of 0.00727; the same test using Cholesky had zero
   variation. This is much larger than the 0.0001 patience tolerance and is a credible mechanism
   for persistent full-length fits. Production settings were not changed by this diagnostic.

2. **New-proposal feasibility is extremely low.** Only 25/51,624 fresh peptides, 0.0484%,
   satisfy the 0.75 normalized edit-similarity constraint. Thirty of 48 tasks have no feasible
   new peptide, and only six have improved their feasible incumbent. Actual sequence distances
   match the recorded feasibility flags. Infeasible scores are not being used as feasible
   incumbents. This is a search-quality concern, not evidence of a score/accounting error.
   The frozen author VAE and native constrained acquisition remain unchanged.

3. **VAE services are healthy, but are not the main runtime bottleneck.** The existing services
   report zero errors. Eight additional services passed parity checks for posterior-mean
   encoding, seeded decoding, and real oracle scores. CPU worker routes are migrated after
   their current batch is durably saved, then the same task resumes. No model-training recipe
   or peptide evaluation budget is changed by the routing update.

## Expanded Services

LLMOptimizer4 currently exposes no NVIDIA GPU devices. All GPU VAE processes therefore run
on LLMOptimizer2; four of the additional services serve each machine's CPU worker pool.

| Worker Pool | New Ports | Service CPU Cores | GPUs |
|---|---|---|---|
| LLMOptimizer2 | 18881-18884 | 3, 4, 6, 7 | One service on each of 0-3 |
| LLMOptimizer4 | 18885-18888 | 9, 10, 21, 22 on LLMOptimizer2 | One service on each of 0-3 |

This brings the LLMOptimizer2 inference pool to sixteen VAE processes, four per GPU,
including its older unassigned service. The worker and service CPU reservations total
38/40 cores. GPU use after parity checks is approximately 4.2-4.6 GiB per 80 GiB GPU.
The routing pools contain seven local and eight remote services. Updated supervisors retain
the existing FSBO-then-RGPE queue and preserve task checkpoint identities during rerouting.

## Checks

- Correct held-out task membership and separation from train800.
- Pinned code and FSBO parent checkpoint identities.
- Proposal/fresh/cache budgets and cached score consistency.
- Peptide alphabet, maximum length, cysteine rule, and actual similarity constraints.
- Higher-is-better score versus MIC signs and feasible incumbents.
- Source-GP reload verification metadata.
- Twelve scheduler tests, including planned checkpoint rerouting and genuine-error parking.
- Nine-service parity check, including the existing baseline service and all eight additions.
- Reverse-tunnel health checks from LLMOptimizer4 for all four new remote endpoints.

## Artifacts

- Full task audit: `/root/data/logs/peptide_vae_expansion_20261029/audit.json`
- Controlled likelihood diagnostic: `/root/data/logs/peptide_vae_expansion_20261029/likelihood_check.json`
- Inference parity: `/root/data/logs/peptide_vae_expansion_20261029/validation.json`
- Expansion launcher and records: `/root/data/logs/peptide_vae_expansion_20261029/`
- Current routing: `/root/data/logs/peptide_transfer_test200_20261028_v1/workers/vae_routes.json`
- Live migration state: `queue_local.json` and `queue_remote.json` in that workers directory.
