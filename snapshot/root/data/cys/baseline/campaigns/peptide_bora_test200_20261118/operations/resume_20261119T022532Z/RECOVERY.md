# BORA checkpoint recovery, 19 November 2026

The original OOM was reproduced with the assigned GPU pair otherwise empty. During the 22,814-token prompt prefill, logical GPU1 had 10.75 GiB of model parameters and reached its 14 GiB per-process allocator cap. The failed allocation was 180 MiB. This was a placement/headroom problem rather than evidence of competing jobs.

Recovery retains all 2,076 committed production oracle evaluations across ten tasks. All checkpoints were audited for unique sequences, consecutive evaluation indices, exact physical-call accounting, matching campaign identity and absence of unresolved in-flight oracle calls. Original checkpoints, source files, manifest and inference journal are archived in before/ and requests_before.sqlite.

The text model is explicitly split at layer16 across each original GPU pair, leaving more even activation headroom and avoiding automatic text-weight offload. No model weights, peptide scoring, task split or 4,000-fresh-evaluation budget changed.

The tokenizer API returns a mapping in these package versions. The old prompt check counted mapping keys rather than input tokens and prevented the intended history summary from activating. Worker.prompt_tokens now counts the same rendered token IDs as inference. A 4,176-row context fixture correctly invokes the existing best40/recent40 summary and produces a 3,700-token prompt. This repairs the existing documented policy; earlier observations are retained and the recovery boundary is recorded in checkpoint_audit.json.

Resume reuses the passed original real-model/oracle branch validation, requires fresh checkpoint hashes and memory/context regression evidence, prioritizes the failed task and admits two workers before ten newly saved evaluations permit ten-worker concurrency. The original asset manifest is preserved; asset_revision.json records each compatible repair while retaining the task checkpoint identity.

LLM requests may be retried after recorded infrastructure failure. Completed journal responses are retained. Peptide oracle calls are never replayed without durable accounting.
