# Query inference capacity expansion — November 17, 2026

The resumed ORPO + FSBO run uses eight LLM replicas and sixteen VAE services on four A800 GPUs: two LLM replicas and four VAE services per GPU. The 21 optimizers and their 21 private PostgreSQL instances retain their original CPU allocations.

ORPO uses a shared inference queue so the original tasks can use all eight replicas. Its prompts, sampling, constraints, acquisition method, budget, checkpoints, and saved measurements are preserved. BOLT retains task affinity for its existing cache policy.

The identical training-prompt comparison measured 1.785 times the inference throughput with eight replicas versus four. After resuming, a five-minute production sample completed 105 evaluations: about 1,260 evaluations/hour, compared with about 445/hour in the preceding baseline hour. This short observation is an early rate, not a guarantee for the full run. Mean LLM queue wait fell from about 68 seconds to 2.8 seconds.

All sixteen VAE services passed checkpoint, encoding-equivalence, and strict plan-validity checks. Their simple microbenchmark was too short to establish their separate contribution to overall throughput. The final VAE health check reported no errors.

The existing independent recovery watchdog still checks every 600 seconds. Resource checks and recovery now derive service counts from the allocation file. Deep health requests allow up to 90 seconds because VAE health shares the decoding mutex. There were 30 passing targeted tests.

Evidence is recorded in `capacity_report.json`, `production_rate.json`, `llm_benchmark.json`, `final_vae_health.json`, `resource_only_changes.json`, and `busy_health_grace_change.json`. Pre-change resources and pins are retained in `before/`. Rollout commands require a paused, verified quiescent campaign and refuse to change active SQL workers.
