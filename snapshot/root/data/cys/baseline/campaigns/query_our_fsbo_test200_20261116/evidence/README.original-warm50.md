# Query BOLT50 → ORPO + FSBO, held-out200

Run the trained BOLT full SFT checkpoint-1224 to propose 50 valid, distinct plans for each of the fixed200 held-out CEB tasks. Accept the first50 in generation order; up to500 generations; no Bao initialization or random replacement. Execute each plan on PostgreSQL with pg_hint_plan and a fixed50-second timeout. Stop and report any underfilled pool.

Freeze all200 measured pools, then run80 independent ORPO/FSBO attempts per query. ORPO is checkpoint-103, NOT the RL checkpoint. The later RL pass is disabled and can reuse the immutable pool without continuing ORPO history.

Use ML/peptide-style PUCB: first8 post-initialization attempts from ORPO; afterwards p1=.7, q=arm_best, c2=1.4, schedule index=attempt−9, denominator91. In attempts10..20, use FSBO if the preceding selected arm was LLM. Five invalid/duplicate LLM generations fall back to FSBO. Five failed FSBO decodes consume an attempt with NO_VALID_PROPOSAL, without a fabricated score. Initialization does not update selector counts.

FSBO uses frozen source query transfer weights, canonical posterior means from the frozen author64D VAE, minmax negative runtimes, source-reset target refit (maximum10000 updates, patience16, delta.0001, lr.001), and native joint Thompson sampling over5000 candidates. Following the prepared peptide hybrid, start a fresh native local trust region at the current incumbent at each FSBO selection. This is a distinct hybrid from the historical continuous-trust-region FSBO baseline. No VAE training, recentering or end-to-end updates. Censored queries remain recorded bounds and are excluded from exact GP fits and incumbent scores; ORPO receives the same uncensored good5/bad5/recent50 training prompt.

All proposal and SQL operations use durable per-task SQLite journals. Model requests have deterministic seeds and replayable request IDs. Known terminal PostgreSQL temporary-file EINTR and connection-timeout failures retry the same proposal up to4 calls. Exhaustion saves a safe-failed proposal and pauses admission without charging an attempt; ambiguous interruptions require audit before replay. Any worker failure pauses new admission; already-running tasks drain. Every successful operation remains checkpointed.

Allocation: 8 private reused PostgreSQL instances on ports20402..20409 (3 CPU cores each), 8 coordinator/FSBO CPU workers (1 core each), 3 BOLT replicas on GPUs0..2, 1 verified ORPO replica on GPU3, and1 frozen VAE service sharing GPU3. Brokers use ports21810/21811. No database copy or new model copy is made. Current progress is in status.json and per-task progress.json; logs are in logs/.

Start: bolt_clean_venv/bin/python code/run.py controller. Inspect: same Python code/run.py status. The controller requires passed training-only preflight, tests and pinned dependencies. It resumes committed history and refuses ambiguous measurements. Peptide is not queued here.
