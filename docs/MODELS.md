# Exact final models for query and peptide

Scope: both discrete domains and every included method. The machine-readable
inventory is [final-models.json](../migration/inventory/final-models.json).
It records selected original paths, file sizes, SHA256, roles and dependencies.
Final trained variants remain distinct even when only one was used by a given
evaluation. Intermediate checkpoints, optimizer states and training RNG are
excluded from these model bundles.

Hugging Face publication is **pending the destination namespace and write
authentication**. No model has been uploaded by this migration yet. Prepared
bundles live on shared storage at `/root/data/discreteDomains-final-models/hf`;
ordinary immutable model files are linked to their original paths so preparation
does not duplicate tens of GiB. Transfer the actual files, not these local links.
New model repositories are private. GitHub code is already public at
<https://github.com/Y0306S/discreteDomains>.

## Method and model mapping

| Method / component | Query | Peptide |
| --- | --- | --- |
| Our warm start | `query-warmstart`: selected full checkpoint1224 | `peptide-warmstart`: selected full checkpoint1500 |
| ORPO proposals | `query-orpo`: checkpoint103 adapter; current evaluation has RL disabled | `peptide-orpo`: final ORPO adapter |
| Trained ORPO+RL variant | `query-orpo-rl`: final round1 adapter, separate from current ORPO-only evaluation | `peptide-orpo-rl`: final round1 adapter |
| SemanticOPT | `query-semanticopt`: completed domain adapter | `peptide-semanticopt`: completed domain adapter |
| Adapter base / BORA / pure Qwen | `qwen35-9b-exact-base`, shared exact local Qwen3.5-9B shards | Same base; no separately trained BORA adapter |
| New representation + FSBO | `query-vae80` + `query-fsbo-vae80` | `peptide-adapted-vae` + `peptide-adapted-fsbo` |
| Frozen FSBO baseline | `query-author-vae` + `query-frozen-fsbo` | `peptide-author-vae` + `peptide-frozen-fsbo` |
| BOLT / LOLBO | New BOLT30 continuation uses our selected query warm start and frozen `query-vae80`. Historical native VAE variants are separately preserved. DKL/GP trains within each task. | Original `peptide-bolt-warmstart` full SFT model, author VAE and APEX. Use the campaign's own measured starting pool. |
| TPE / random / GP / HEBO / SMAC | No separate universal trained checkpoint; retain the recipe's codec/VAE, initialization and SQL oracle | No separate universal trained checkpoint; retain the recipe's VAE, initialization and APEX oracle |
| RGPE | `query-rgpe`: 788 immutable source GPs, not 800 | `peptide-rgpe`: 800 immutable source GPs |
| Retrieval | `query-retrieval-gp-means` plus frozen original encoder | `peptide-retrieval-gp-means` plus recipe-specific encoder |
| Online peptide variants | Query evaluator's matching resume journals/state are separate scientific artifacts | Final per-task parameter exports for original online, rollback and feasible-search versions, ten tasks each |
| Oracle | PostgreSQL/IMDb, indexes, pg_hint_plan and measurement configuration, transferred separately | `peptide-apex-oracle`: eight exact predictor weight files; source/preprocessing assets are in the snapshot |
| ToSFiT / OptFormer | Upstream code included; no completed discrete-domain model established | Same limitation; no fabricated final checkpoint |

Adapters load against the exact shared base, not against the warm-start model.
Keep the tokenizer and chat template from each selected checkpoint. Current
query ORPO-only evaluation must not silently be replaced by the trained RL
variant. Small source GP mean files do not reproduce RGPE posterior covariance;
the full bank is required.

## Prepare and verify on the source server

Use Python 3.11+ for SQLite streaming BLOB export. The deployed
`bolt_clean_venv` interpreter is required only for extracting trusted online
PyTorch checkpoints. Extraction runs on CPU and verifies every exported tensor.

```bash
cd /root/discreteDomains
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/data/cys/envs/bolt_clean_venv/bin/python migration/export_online_models.py
python migration/model_catalog.py --reuse-hashes
python migration/hf_models.py prepare
python migration/hf_models.py prepare --id query-rgpe --id peptide-rgpe --export-rgpe
python -m unittest discover -s migration -p test_hf_models.py
```

The catalog hashes all normal file contents. `--reuse-hashes` uses a previous
digest only when size and nanosecond mtime match. Upload rehashes prepared
payloads before publishing. RGPE source-row hashes are initially recorded from
SQLite, then verified against the actual bytes during export. Export is resumable:
existing per-task files are retained only after size and SHA256 validation.
Its temporary files remain on `/root/data`, which has room for the large bank.

The shared model set is approximately 391 GiB before online parameter exports;
the peptide RGPE payload alone is 314.3 GiB. The original SQLite container is
larger because of database overhead. RGPE exports are one file per task
(roughly 149–427 MB each for peptide), plus original row metadata and SHA256.
Reconstruction preserves model rows and their exact payloads; SQLite container
bytes can differ. Scientific data, SQL dumps, environments and resume backups
require additional storage.

## Publish after Hugging Face access is available

Authenticate on the server without putting tokens in source, command arguments
or chat. Supply the destination account or organization namespace.

```bash
hf auth login
python migration/hf_models.py upload --namespace YOUR_NAMESPACE
```

Repositories are named `YOUR_NAMESPACE/discrete-domains-MODEL_ID`. The uploader
checks that every catalog bundle was prepared; partial publication can be
explicitly selected with repeated `--id`. Existing public model destinations
are refused. It verifies remote file sizes and LFS SHA256, downloads small
non-LFS files for SHA256 checks and verifies the provenance manifest at the
returned commit. Verified revisions are recorded outside Git in
`/root/data/discreteDomains-final-models/hf-publication.json`. Keep that receipt
with your migration backup and use its exact commits for downloads.

An upload that fails before remote verification is not recorded as complete.

## Restore on the new server

Install `huggingface_hub` (source-server version 0.34.4), authenticate to the
private destination and clone the code. Restore source layout as described in
[MIGRATION.md](MIGRATION.md), then use each verified publication receipt:

```bash
python migration/hf_models.py restore \
  --repo-id YOUR_NAMESPACE/discrete-domains-query-warmstart \
  --revision FULL_VERIFIED_COMMIT_SHA --destination /
# Review the paths, then perform the verified restoration:
python migration/hf_models.py restore \
  --repo-id YOUR_NAMESPACE/discrete-domains-query-warmstart \
  --revision FULL_VERIFIED_COMMIT_SHA --destination / --apply
```

Repeat for each dependency in the manifest. Downloads require immutable commit
revisions. Restore previews paths by default, hashes downloaded payloads before
writing, accepts identical existing files and refuses conflicting files.
RGPE restoration streams payloads into a fresh SQLite database and runs
`PRAGMA quick_check`; it refuses to replace an existing bank. Hugging Face cache
files and restored payloads may both occupy storage, so provision space for both.

Online exports contain VAE, FSBO source prior/normalization and constraint-model
parameters with original checkpoint identities. They exclude task histories,
RNG and trust-region state. Target FSBO is refitted from the matching observations
in the original evaluator; these exports alone do not reproduce a fitted target
posterior or resume a run. Preserve the full original checkpoint and matching
journals with the separate consistent scientific-state backup.

## Scientific inputs and validation boundary

A model download does not supply source800 corpora, measured warm-start pools,
immutable task splits, full histories, PostgreSQL data or environment binaries.
Use the artifact transfer/backup procedures in [MIGRATION.md](MIGRATION.md).
Peptide initialization pools already included in Git have their own integrity
check in [PEPTIDE_INITIALIZATION.md](PEPTIDE_INITIALIZATION.md).

Source hashes and transfer fixtures were checked here. A fresh server still
needs actual encode/decode, APEX/SQL, service transport, model reload and
interruption/resume validation with the chosen campaign. No new experiment
results are implied by preparing or publishing its code and weights.
