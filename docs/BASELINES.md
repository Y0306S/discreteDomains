# Baseline code and new-server requirements

Audited 22 November 2026. The actual directory is **`/root/data/cys/baseline`**
(singular). This guide covers the peptide and query experiments in this repo.
The shared harness also contains ML/TaskSet implementations; migrating their
complete datasets and campaigns is a separate scope from this discrete kit.

All paths below are restored absolute paths. Their source copies live at
`../snapshot/` followed by the same path without its leading slash. Restore
with `migration/restore_layout.py`; do not execute a controller from `snapshot`.
The machine-readable map is `migration/inventory/baselines.json`.

## What is included and which runner to use

Abbreviations: `T=/root/data/cys/llmoptimizer_benchmark/tools`,
`D=/root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains`,
`C=/root/data/cys/baseline/campaigns`, `R=/root/data/cys/baseline/repos`.
These are documentation abbreviations, not environment variables set by the kit.

| Method | Actual implementation / entrypoints | Environment and external inputs |
| --- | --- | --- |
| TPE | `T/peptide_test200.py`, `T/peptide_local_bo.py`; trust-region variant `T/peptide_tpe_tr.py`; query `D/run_query_bolt_tpe.py` | Modern `bolt_clean_venv` + parent; Optuna version and campaign recipe must match; frozen author VAE, saved initialization and domain oracle |
| Random / ordinary GP / HEBO / SMAC | `T/peptide_local_bo.py` exposes these method choices; shared implementations in `mindware/domains/protein` | Modern stack and method-specific dependencies; these available choices are not evidence of completed comparison runs |
| LOLBO / BOLT | `D/run_protein_bolt_lolbo.py`, `D/run_query_bolt_lolbo.py`, native BOLT/BayesQO trees; newer `C/query_bolt_lolbo_test200_20261117/run_lolbo.py` | Modern runner/VAE/oracle; legacy `bl_lolbo_exact` retained separately. BOLT additionally requires its trained warm-start checkpoint and measured pool |
| FSBO | `T/peptide_transfer_eval.py`, `T/peptide_transfer_models.py`, `T/query_fsbo_eval.py`, `T/query_fsbo_model.py`, `T/query_transfer.py`, `R/FSBO/fsbo_modules.py` | Legacy `bl_fsbo_exact` model subprocess + modern query/peptide process; trained source weights, corpus and matching VAE coordinates |
| RGPE | `T/peptide_rgpe_exact.py`, `T/peptide_rgpe_fit.py`, `T/peptide_rgpe_merged_worker.py`; latest screening variant `C/peptide_rgpe_screen200_test200_20261119/code` | Legacy `bl_rgpe` reference and modern accelerated services; all 800 immutable source GPs, source normalization and frozen author VAE; CPU/FP64 GPU services and endpoints per campaign |
| BORA | `R/bora-the-explorer`; peptide adaptation `C/peptide_bora_test200_20261118/code/{scheduler,worker,validate}.py` | `bl_bora` export for upstream; deployed scheduler uses modern oracle plus **`taskset_qwen35_fast_20261030`** for inference. Untouched Qwen3.5-9B, author VAE, APEX, initialization, broker journals |
| SemanticOPT | `C/semanticopt_discrete_20261109/code/{prepare,train,training_queue}.py`; `C/semanticopt_eval200_20261110/code/queue.py`; ten-task continuation `C/semanticopt_peptide_test10_5000_20261117/control/controller.py` | Training/reference `bl_semanticopt`; runtime uses modern oracle plus fast Qwen overlay. Exact domain LoRA, base Qwen, replay/SFT inputs, broker and task SQLite journals |
| ToSFiT | `R/tosfit/scripts/run_tosfit.py`, `R/tosfit/src/tosfit.py`, upstream configs | `bl_tosfit`; upstream protein objective is not the APEX MIC benchmark. Source and environment are included, but no peptide/query production adaptation is established by this kit |
| OptFormer | `R/optformer-neurips22/optformer/data/tasks.py`, `R/t5x-2023-07/t5x/train.py`; both current and older upstream trees retained | `bl_optformer` (Python 3.10, JAX 0.4.23, TensorFlow 2.13.1). Discrete adaptation was deferred; a trained scenario checkpoint and trajectory task/tokenization are still required |

The original `baseline/INTEGRATION.md` and `STATUS.md` are historical planning
notes, including ML adapter work. Later campaign code/policies and pinned
recipes determine what actually ran. Do not interpret a cloned upstream repo,
an installed environment, or a successful import as a complete baseline run.

## Versions that must remain distinct

- The **completed historical** query FSBO/LOLBO/TPE comparison uses Bao49
  initialization. The 17 November four-method report records 1,000 proposals
  for FSBO/LOLBO/TPE and 200 for SemanticOPT (batch ten), with differing seeds
  and SQL timeout policies. See the captured
  `/root/reports/query_convergence_four_methods_20261117/{README.md,audit.json}`
  for exact histories and provenance.
- The **newer BOLT30 + LOLBO** continuation is a different campaign. Its README
  records a failed feasible-decode canary with zero SQL measurements and a
  retained `PAUSE` file. Fix/revalidate proposal generation before launch;
  do not label it as the completed historical LOLBO baseline.
- The main query method is the VAE80 + rebuilt FSBO campaign documented in
  [EXPERIMENTS.md](EXPERIMENTS.md). Retrieval retains its original encoder.
- BORA peptide caps 4,000 fresh calls after 100 initialization measurements.
  The screening RGPE variant caps both acquisition proposals and fresh logical
  APEX calls at 4,000 and screens 1,500 candidates to 200 before joint sampling.
  These are different mechanisms even where plots share a budget axis.
- SemanticOPT has several policy amendments and continuations. The ten-task
  peptide extension preserves the first 1,000 observations and extends to
  5,000; it is not a fresh 5,000-budget rerun. Read effective policies and
  checkpoint histories, rather than using an early README budget paragraph.
  Export fallback counts separately; historical query SemanticOPT has substantial
  constrained random fallback use.

## Environments: restore the processes, not just a baseline name

`conda activate base` is useful for migration tooling. Experiment subprocesses
use absolute interpreter paths; activating base does not supply their packages.
Every exported environment has exact package metadata in `environments/NAME`.
Old `build_env_*.sh` scripts are preserved as provenance: several delete an
existing environment, and some installed versions have changed since those
scripts were written. Use current archives or captured exports instead.

Restore in this dependency order:

1. `openbox_qwen_eval` at `/root/data/cys/envs/openbox_qwen_eval`.
2. `bolt_clean_venv` at `/root/data/cys/envs/bolt_clean_venv` (overlay on 1).
3. `llamafactory_qwen35` at
   `/root/data/cys/blockdata_archive/miniconda3/envs/llamafactory_qwen35`.
4. `taskset_qwen35_fast_20261030` at
   `/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030` (overlay on 3).
5. `bl_fsbo_exact`, `bl_rgpe`, and selected reference/optional environments
   at the prefixes in `environments/index.json`; PostgreSQL `pg16` separately.

To preview packing the deployed peptide/query runtime plus baseline references
on the **old server**:

```bash
cd /root/discreteDomains
python migration/pack_environments.py --output /root/data/discreteDomains-env-packs \
  --names openbox_qwen_eval bolt_clean_venv llamafactory_qwen35 \
  taskset_qwen35_fast_20261030 bl_fsbo_exact bl_rgpe bl_lolbo_exact \
  bl_bora bl_semanticopt bl_tosfit bl_optformer pg16
```

Add `--apply` to create archives. Additional auxiliary environments can be
added by name: `bolt_qp`, `bl_lolbo_harness`,
`bl_lolbo_harness_cu118`. Transfer archives and their manifest outside Git.
Verify SHA256 before extraction, extract into the recorded prefixes in the
order above, then run `PREFIX/bin/conda-unpack` for each Conda archive.
The two overlay archives use tar and depend on their parents' paths. Do not
run conda-unpack on these venv overlays. Restore editable project sources and
re-register LLaMA-Factory as described in [environments/README.md](../environments/README.md).
The fast Qwen overlay has three local wheel archives under its own wheels/
directory. The artifact transfer plan includes them for specification rebuilds;
the packed overlay already contains them. Its overlay-only.txt lists only
package overrides, avoiding reinstallation of the entire inherited parent.
For optional OptFormer, re-register both older worktrees with `pip install
--no-deps -e` using its own Python.

The specification rebuild alternative is:

```bash
python migration/rebuild_environment.py ENVIRONMENT_NAME
# On the new server, after inspecting the preview:
python migration/rebuild_environment.py ENVIRONMENT_NAME --apply
```

The helper refuses to replace an existing prefix and understands both overlays,
editable baseline source installs and pinned Git direct installs. Rebuilding
old package stacks has not been tested on a clean target. Legacy CUDA wheels
may be incompatible with a newer GPU architecture even if the driver supports
new CUDA; validate the actual kernels or use a compatible CPU path explicitly.

## Inputs that Git cannot replace

| Payload | Why it is required | Where to find the exact identity |
| --- | --- | --- |
| Frozen 800/200 splits and initialization | Reproduce the same held-out tasks and measured starting observations | Campaign recipes, manifests and source histories; never regenerate a split for a continuation |
| Author peptide VAE + query VAE/native codecs | Convert proposals to/from the same representation | `primary-inputs.json`, original recipes, BOLT saved models and query author `native/` tree |
| FSBO weights and RGPE source GP bank | Transfer learning depends on the exact source fit and representation | Campaign `fsbo_recipe.json`, transfer directories and checkpoint pins |
| APEX eight-model ensemble and preprocessing assets | Actual peptide MIC scoring | APEXGo/BOLT assets, campaign asset manifests and `upstream_cpu` dependencies |
| Qwen3.5-9B base, warm-start full checkpoints and domain adapters | LLM initialization/proposals and SemanticOPT/ORPO behavior | Base `/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B`, campaign model paths and identity manifests |
| Training corpus / source payloads / replay data | Needed for retraining, even if evaluation can use existing weights | `corpus`, `source_snapshot`, `deps`, `fsbo`, `vae` symlinks and their recorded targets |
| IMDb database, schema, indexes and pg_hint_plan | Query runtime oracle | Logical PostgreSQL dump and extension/schema checks; raw live PG files are excluded from rsync |
| Task state and service journals | Preserve measured results, RNG, pending proposals and resume identities | SQLite journals, model states, measured pools, checkpoint/completion manifests and broker request logs |

`artifact-roots.txt` is the transfer plan; `exact-artifacts.json` adds discovered
model/data files and sizes. `symlinks.json` records links and targets. Source
files already captured are excluded from payload transfer. Large checkpoint
**directories** and dynamic paths may not appear in the exact-file list: keep
the campaign roots and scientific symlink targets in the payload plan.
These inventories are not a byte-for-byte backup of remote hosts. Obtain
host-local journals, remote-only model assets and database backups from each
worker host. Small captured JSON may include historical runtime endpoints;
provision fresh service-token files and update the topology on the target.

## What to provision on the target

Use Linux x86_64 and preserve original absolute paths. Provide Git, Python 3,
Conda/conda-pack, rsync, SSH, tar, a compatible NVIDIA driver and sufficient RAM,
GPU memory, storage and inodes. There is no measured universal minimum server
size: source deployment used multiple hosts and GPU models. GPU indices, CPU
quotas, ports and memory reservations in campaign policies describe that
allocation, not portable requirements. Reduce concurrency through a reviewed
new resource allocation while retaining scientific budgets and model identities.

Query additionally needs PostgreSQL 16, database roles, pg_hint_plan compiled
for that installation, restored IMDb replicas, configured permissions, disk/temp
space, intended cache policy and reachable VAE/LLM endpoints. `/dev/shm` consumes
RAM and is erased on reboot. Reproduce or explicitly record storage and cache
changes when comparing runtime objectives. Keep one database worker allocation
per intended oracle isolation policy.

Do not copy machine SSH/API credentials. Configure new SSH routing and fresh
service credentials. Captured schedules often refer to `LLMOptimizer2/3/4`,
private localhost tunnels and fixed ports. Review `resources.json`, policies,
service configs and all interpreter references before starting controllers.
Monitor jobs and scheduler sessions must be registered on the new hosts.

## Migration and acceptance sequence

1. Verify source hashes, restore source layout, and audit baseline coverage:
   `python migration/verify_snapshot.py` and
   `python migration/audit_baselines.py` from the repository root.
2. Copy immutable scientific inputs using the transfer preview/apply procedure
   in [MIGRATION.md](MIGRATION.md). Use rsync's dry-run statistics for the actual
   storage estimate; this guide does not invent a size for changing remote data.
3. Transfer and restore environments in dependency order. Restore editable
   sources. Check Python/Torch/GPyTorch/Optuna versions in the actual worker
   interpreters, including service subprocesses.
4. At a normal final checkpoint boundary, stop the old campaign through its
   own controller, back up SQLite consistently, dump PostgreSQL, and transfer
   the matching final model/pool/journal state. Keep one authoritative writer.
5. Restore databases, configure new host resources/endpoints/tokens and inspect
   retained `PAUSE`/`STOP` markers. Do not remove a scientific failure gate simply
   to force launch; resolve the failure and record new validation evidence.
6. Run `python migration/audit_baselines.py --restored`. This checks entrypoint
   hashes, exported environments, interpreter paths and the 25 main-input paths;
   it does not certify every artifact, remote service or checkpoint identity.
7. Run CPU regression tests in [EXPERIMENTS.md](EXPERIMENTS.md), then the selected
   campaign's real VAE encode/decode, APEX or SQL oracle, model reload, full
   source-bank, inference, transaction and interruption/resume gates. Check
   stored checkpoint/code identities and objective direction/units.
8. Run a separately identified one-task canary with actual oracle calls and
   verify counters, initialization exclusion, feasibility, censoring, duplicate
   handling, fallback labels and journal resume. Only then admit more workers.
9. Record target hardware, versions, data/model hashes and measurement boundary.
   Regenerate status/results from committed journals. Historical PIDs and health
   reports cannot validate new-server processes.

Read-only progress entrypoints after layout/environment restoration:

```bash
cd /root/data/cys/baseline/campaigns/peptide_bora_test200_20261118
/usr/bin/python3 code/scheduler.py --status
cd /root/data/cys/baseline/campaigns/peptide_rgpe_screen200_test200_20261119
/usr/bin/python3 status.py
cd /root/data/cys/baseline/campaigns/query_bolt_lolbo_test200_20261117
/root/data/cys/envs/bolt_clean_venv/bin/python run_lolbo.py status
```

Launch commands remain in the versioned campaign READMEs and
[EXPERIMENTS.md](EXPERIMENTS.md). They must be used with the intended campaign
working directory and validated new-host topology. No experiment was launched,
stopped or transferred as part of this repository audit.
