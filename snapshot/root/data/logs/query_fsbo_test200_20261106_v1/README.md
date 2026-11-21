# Query FSBO: frozen VAE, original peptide Thompson wrapper

Prepared only; the full held-out campaign has not been launched.

Uses the new query-specific FSBO weights and the original peptide CPU
TuRBO / batch Thompson implementation. The author query VAE remains frozen.
Stalls only affect the ordinary trust region: no VAE training and no extra
recentering phase. FSBO GP and feature-network adaptation still run each round.

Scope: held-out test200 queries, disjoint from all query source training.
Per task: 49 official Bao imports, then 1,000 charged proposals in batches of
10, selected from 5,000 candidates. SQL executions have a 100-second cap;
censored bounds are preserved and excluded from exact FSBO fitting.

All 16 admitted workers use their existing private PostgreSQL copies on
LLMOptimizer4. The ten existing query VAE services supply frozen inference.
worker_plan.json contains the concrete commands; none have been launched.

Implementation and operation:
/root/data/cys/llmoptimizer_benchmark/docs/query_fsbo_frozen_wrapper.md

Separate training-query component validation:
/root/data/logs/query_fsbo_wrapper_check_20261106_v1

recipe.json pins the implementation, split, models, normalization and assets.
tasks.json preserves canonical Bao imports and SQL identities. Per-task
checkpoint.pt, artifacts.sqlite3, history.json and status.json support
resume without repeating a committed or unresolved SQL execution.
