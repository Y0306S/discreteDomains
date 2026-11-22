# discreteDomains

Code and server migration kit for the **peptide** and **query-plan** experiments,
captured from the working server on 21 November 2026.

The repository includes locally modified experiment implementations, training
and evaluation scripts, SQL workloads, configuration, environment exports,
baseline implementations and upstream licenses. Model weights, large training
histories, database contents and environment binaries have a separate transfer
plan. **A Git clone alone does not contain those experiment inputs.**

## Where to start

| Item | Location |
| --- | --- |
| Server migration steps | [docs/MIGRATION.md](docs/MIGRATION.md) |
| Experiment versions and commands | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| Baseline methods, environments, payloads and acceptance gates | [docs/BASELINES.md](docs/BASELINES.md) |
| Python/CUDA/PostgreSQL environments | [environments/README.md](environments/README.md) |
| Original source layout | [snapshot/root](snapshot/root) |
| File hashes, upstream commits, symlinks | [migration/inventory](migration/inventory) |
| Restore, package and transfer tools | [migration](migration) |
| Verification performed here | [docs/VALIDATION.md](docs/VALIDATION.md) |

The `snapshot/` tree mirrors the original absolute paths. For example,
`snapshot/root/data/cys/llmoptimizer_benchmark/tools/query_fsbo_model.py`
restores to `/root/data/cys/llmoptimizer_benchmark/tools/query_fsbo_model.py`.
This preserves the path assumptions and checksum identities in existing runs.
Work from the restored copy when running experiments; do not run controllers
directly from the Git snapshot.

The main peptide version is
`/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119`.
The main query version is
`/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117`.
Their parent experiments, ORPO/RL preparation, warm-start training, baseline
implementations and analysis scripts are included with their original names.

Eight shared-helper files had embedded API credentials replaced with
environment-variable lookups in this copy. See `.env.example` and the
`sanitized` entries in `migration/inventory/sources.json`. The production
source files were not edited. Existing historical status files, hostnames,
process IDs and validation reports are provenance, not new-server readiness.

## Local checks

```bash
python migration/verify_snapshot.py
python migration/restore_layout.py --destination /tmp/discreteDomains-preview
```

For a new server, follow the migration guide before running the experiments.
The original scientific budgets, splits and checkpoint identities are retained;
hardware assignments and local service credentials must be provisioned there.

## Third-party code

Upstream licenses remain beside their respective source trees. Original Git
remotes and commit IDs are recorded in `migration/inventory/upstream.json`.
The source includes local modifications beyond those upstream commits; the
snapshot hashes identify the files preserved here. No new blanket license is
applied to third-party code or research data.
