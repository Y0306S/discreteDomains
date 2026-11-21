# Bundle adjustments

The live campaigns were not edited. Bundled peptide `common.py` resolves the shared helpers and native peptide sources relative to this experiments folder, with environment overrides.

The historic `train_bolt_query_vae.py` had two byte-identical scripts concatenated, causing an illegal second future import. The bundled file retains the first copy. This trainer is historical; the active query VAE adaptation uses `query_fsbo/training/train.py`.

Configuration templates replace machine paths (including pinned-file dictionary keys) with environment variables. Historical source/content hashes are preserved as provenance; changed code needs new deployment pins and preflight evidence.
