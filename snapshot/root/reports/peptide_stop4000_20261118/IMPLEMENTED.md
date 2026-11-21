# Peptide execution cap: 4,000

Implemented for SemanticOPT and the current online VAE + scratch FSBO run on the same ten held-out peptide tasks. Batch size remains 50, with an 80-round proposal limit and a 4,000-fresh-evaluation ceiling, excluding 100 initialization measurements. Cached duplicates and recenter calls mean online proposal rounds and fresh evaluations can differ; it stops when either cap is reached.

Both campaigns resumed saved progress. All ten SemanticOPT workers report budget 4,000 and preserve their saved observations byte for byte. Online workers loaded the capped configuration; completed batches now publish explicit cap metadata, while an in-progress scratch FSBO update resumes its training checkpoint. Existing monitors continue running.

Validation: 13 batch/resume tests passed. Effective budgets, dependency hashes and unchanged model/configuration identity verified. Final reports now accept the 4,000 cap. Original control files are retained in before/. See live_verification.json for evidence.
