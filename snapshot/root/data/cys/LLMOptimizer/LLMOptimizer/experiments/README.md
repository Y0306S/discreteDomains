# Reproducible Query and Peptide Experiments

This is a source bundle for the query and peptide campaigns. It preserves the campaign runners and helper modules, adds deployment templates, and records provenance. It does not include weights, measured datasets, SQL databases, API credentials, or live process state, so it is not a standalone runnable deployment until those assets and services are supplied.

## Query

`query_fsbo/batch_code/` is the active batch runner from the fine-tuned-VAE campaign. `query_fsbo/code/` preserves the earlier single-proposal runner. The `query_training/` and `query_warmstart/` folders hold their training scripts. The peptide folders preserve peptide warm-start, ORPO/RL-FSBO evaluation, and training source. `benchmark/tools/` and `benchmark/examples/` contain shared campaign helpers. The `vendor/` folder contains selected upstream Python source and license/provenance files.

## Peptide

`peptide_fsbo/code/` is the target-loss-gated online VAE/FSBO implementation. Its retraining scheduler waits for 20 acquisition rounds without a feasible MIC improvement, allows at most two attempts per peptide including rejected/interrupted attempts, and keeps the target/source rollback gates. Historical validation records are in `peptide_fsbo/tests/`; they are provenance, not a substitute for rerunning tests after deployment. `configs/config.template.json` is a sanitized template and must be rendered with local asset paths.

The production campaign was run on LLMOptimizer3 with batch size 50 and a 4,000 proposal cap. Run entry points still expect campaign runtime directories, corpus snapshots, model checkpoints, service configuration, and a compatible CUDA environment. Consult each campaign README and `ENVIRONMENT.example`; inspect and render every config before launching. Do not run the historical deployment scripts as-is on a new host.

## Verification

```bash
python experiments/verify_bundle.py
```

The verifier parses and compiles every bundled Python file in memory and checks required entry points. The peptide CPU unit tests can be run with:

```bash
PYTHONPATH=experiments/vendor/peptide_native \
  python -m unittest discover -s experiments/peptide_fsbo/code -p 'test_*.py' -v
```

These checks do not run model inference or contact remote hosts. Query end-to-end checks require the external query corpus and DB services.

## External assets

The source campaigns recorded checkpoint and corpus hashes in their original manifests. Keep those assets in a separate artifact store and provide paths through deployment configuration; never commit them or credentials to this source tree. The original campaign paths embedded in Python modules are deployment-specific and need to be replaced with your host's configured paths before a new deployment.

See [ASSETS.md](ASSETS.md) for the deployment sequence, [SOURCE_ADJUSTMENTS.md](SOURCE_ADJUSTMENTS.md) for bundle-only edits, and `manifest.json` for file hashes and source provenance. `semanticopt_query/` and `semanticopt_peptide/` preserve the comparison runners; `semanticopt_training/` preserves their training entry points. `query_bolt_lolbo/` holds the query BOLT continuation, and the shared benchmark helpers include historical TPE/LOLBO/FSBO/RGPE front ends.

Run `python experiments/run_checks.py` in a compatible benchmark environment to execute source compilation and the 43 selected CPU tests. The broader historical test suites include tests requiring a prepared campaign and external assets; they are not included in that command.
