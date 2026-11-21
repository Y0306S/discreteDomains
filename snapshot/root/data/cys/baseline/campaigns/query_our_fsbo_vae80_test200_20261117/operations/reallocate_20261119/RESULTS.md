Query ORPO + FSBO resource expansion completed

- 37 evaluators and 37 private PostgreSQL databases: LLMOptimizer2 10, LLMOptimizer3 4, LLMOptimizer4 23.
- 20 verified LLM replicas: 12 A800 replicas and 8 A16 replicas using all 16 A16 cards in pairs.
- 24 Query VAE services; independent peptide adaptation and VAE jobs preserved.
- Current measurements: 60147 / 160000; completed tasks: 56 / 200.
- Six A16 pairs passed 32616 input tokens with a 1024 output-token allowance; measured peak 14.083984375 GiB per card.
- Two shared pairs passed 16316 tokens locally, measured peak 11.328125 GiB; 32616-token requests were forwarded intact to A800s and verified.
- All 20 LLM services healthy; all 37 optimizer launchers running; host 3 worker process identities verified.
- 63 tests passed cumulatively across serving, batching, query infrastructure and recovery.
- Recovery enabled; 345 file pins verified.
- BORA and RGPE remain stopped; saved checkpoints retained.

One temporary spill-file EIO on host 3 was audited at a fully drained database boundary. Its journal was backed up, exactly one unscored SELECT was marked replayable, and its plan and all saved results were retained. The exact temporary-read path is now covered by bounded retries and regression tests.

No measured campaign speedup is claimed yet. Weights, prompts, sampling, SQL timeout (50 seconds), and budget (800 evaluations per task, 200 tasks) are unchanged.
