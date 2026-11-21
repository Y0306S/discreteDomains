# Validation on 20 November 2026

- Source syntax/bytecode compilation runs in memory via `verify_bundle.py`, which also checks all source file hashes and the exact manifest file set.
- The bundled peptide CPU suite passed 17 tests covering rollback of parameters/buffers, source retention, fixed target anchors, numerical and infrastructure failures, RNG isolation, 20-round scheduling, and the two-attempt cap.
- The selected query CPU suite passed 26 tests covering request routing, inference replica isolation, storage/transport recovery, and retrieval representation separation. They use local mocks and fixtures, not SQL measurements or GPU inference.
- The broader query discovery command could not import `test_batch` and `test_fsbo_batch`: those tests expect a prepared campaign layout and external query data. They remain included for deployment validation.
- Full inference, model retraining, peptide oracle scoring, PostgreSQL query evaluation, and a clean machine installation were not exercised for this source bundle.

Run `python experiments/run_checks.py` with the benchmark Python environment to reproduce the selected checks. `environment_versions.cpu_checks.json` records the observed installed package versions. Historical JSON under `peptide_fsbo/tests/` records earlier campaign checks; it is distinct from the checks listed above.
