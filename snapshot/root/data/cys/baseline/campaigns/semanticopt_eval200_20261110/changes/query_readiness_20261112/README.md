# SemanticOPT query readiness

Query is queued after peptide with batches of 50 and 20 rounds (1000 proposals per query). The 200-task prompt/validator audit passed without new oracle evaluations.

The query host controller refreshes query/progress.json every 60 seconds when the query phase starts. No additional scheduler or active peptide restart is needed. Timing excludes historical observations and requires at least 5 minutes and 25 new committed evaluations before publishing provisional ETA. See completion.json, constraint_audit.json and timing_tests.log.

The previous training-only batch probe returned 50 duplicates, so its all-fallback result is not evidence of productive LLM proposal quality. Query production fallback counts will remain visible in progress.json. Query budgets already equal 20 rounds; further reduction is not part of this change.
