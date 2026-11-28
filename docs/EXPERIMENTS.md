# Experiment map

Paths below refer to the restored server layout. Their Git copies are under
`snapshot/` with the same hierarchy. Original READMEs remain beside each
campaign. This inventory preserves experiment versions rather than treating
their different methods/results as interchangeable.

## Peptide

| Version | Original directory | Purpose |
| --- | --- | --- |
| Feasible-search continuation | `/root/data/logs/peptide_fsbo_feasible_search_20261122` | Later exact-feasibility screening and adaptive decode-yield policy; retains the loss gate and two-attempt cap |
| 20-round/max-two gate | `/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119` | Fixed target/source loss gate, 20 stagnant acquisition rounds before an attempt, at most two attempts per task |
| First target-loss gate | `/root/data/logs/peptide_target_loss_gate_test10_20261119` | Parent gate implementation and real-checkpoint validation |
| APEXGo recipe + FSBO | `/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118` | Parent offline VAE training, full source800 FSBO rebuild and online evaluation |
| Source replay/adaptation | `/root/data/logs/peptide_vae_fsbo_source_replay_test10_20261118` and `/root/data/logs/peptide_vae_fsbo_adaptation_test10_20261117` | Earlier adaptation implementations and shared dependencies |
| ORPO/RL training | `/root/data/cys/baseline/campaigns/peptide_our_orpo_rl800_20261111` | Pair building/audits, ORPO, retrieval, rollout collection and RL training |
| Warm-start training | `/root/data/cys/baseline/campaigns/peptide_warmstart_full800_20261111` and `peptide_warmstart_lora800_20261111` | Full and LoRA warm-start variants |
| ORPO/FSBO held-out run | `/root/data/cys/baseline/campaigns/peptide_our_fsbo_test200_20261114` | Held-out evaluation and service code |
| Baselines | `.../semanticopt_peptide_test10_5000_20261117`, `.../peptide_bora_test200_20261118`, `.../peptide_rgpe_screen200_test200_20261119` | Separate SemanticOPT, BORA and RGPE campaigns; use each campaign's own recipe and caps |

The current matched peptide cohort is `pep_117`, `pep_219`, `pep_221`,
`pep_575`, `pep_673`, `pep_921`, `pep_925`, `pep_967`, `pep_980`, `pep_999`.
It uses seed 1, 100 initialization measurements, batch size 50, 4,000 proposal
and fresh-evaluation caps, similarity threshold 0.75 and source800 FSBO
retraining for 100,000 steps after accepted VAE updates. The fixed target
reconstruction+KL gate must pass against both the pre-update model and fixed
initial baseline, together with the source gate. Accepted, rejected and
interrupted attempts count against the two-attempt limit.

Essential non-Git inputs include the released author peptide VAE, APEX MIC
oracle models, source800 corpus/payloads and pretrained FSBO/VAE assets. Their
paths and existing hashes remain in the original configs/manifests and the
migration artifact inventory. `code/common.py` documents the shared input
paths. The `corpus`, `fsbo`, `vae`, `source_snapshot`, `reference` and `deps`
symlinks are dependencies, not empty optional directories.

After restoring dependencies, these CPU tests make no oracle calls:

```bash
cd /root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119/code
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/data/cys/envs/bolt_clean_venv/bin/python -m unittest test_target_gate test_retrain_schedule
```

Single-task GPU evaluator entry point:

```bash
cd /root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119
CUDA_VISIBLE_DEVICES=0 /root/data/cys/envs/bolt_clean_venv/bin/python code/evaluate.py --task pep_999 --arm adapted_online
```

The evaluator may resume an existing saved run or recognize completion. For
a fresh replicate, create a separately identified run/configuration rather
than overwriting saved scientific results. `code/pipeline.py` is the complete
controller, but its recorded GPU indices/CPU affinities and manifest pins
must be validated for the new host before using it. Historical validation
reports do not replace a new-host live VAE/oracle/checkpoint test.

## Query plans

| Version | Original directory | Purpose |
| --- | --- | --- |
| Current VAE80 rerun | `/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117` | Fine-tuned VAE, rebuilt FSBO, separate retrieval encoder and durable held-out evaluation |
| Parent frozen encoder | `/root/data/cys/baseline/campaigns/query_our_fsbo_test200_20261116` | Original frozen-representation comparison, shared code and recovery implementation |
| ORPO/RL training | `/root/data/cys/baseline/campaigns/query_our_orpo_rl800_20261112` | Preference data, ORPO/RL preparation/training and retrieval assets |
| Warm start | `/root/data/cys/baseline/campaigns/query_warmstart_full800_20261112` | Full SFT warm-start checkpoint |
| BOLT/LOLBO | `/root/data/cys/baseline/campaigns/query_bolt_lolbo_test200_20261117` | Held-out baseline comparison |
| Author native query stack | `/root/data/cys/logs/query_train800_author_fixed_20261004/native` | Native codec, SQL workloads, VAE and oracle integration |

The split is 800 training / 200 held-out queries. The VAE80 variant selects
80 training-side tasks (64 gradient, 16 validation), then rebuilds FSBO in
the new coordinates for 100,000 updates. It imports 30 saved BOLT warm
measurements per held-out task and performs 77 further batches of ten:
800 total measurements per task, including the initialization. SQL timeout
is 50 seconds. Retrieval continues to use the original encoder and source
GP coordinate space; the FSBO encoder must not replace it.

Important paths within the current campaign:

- `training/config.json`, `train.py`, `pipeline.py`, `gpu_reembed.py`, `promote.py`:
  training selection, durable training sequence and promotion gates.
- `training/vae/best.ckpt`, `training/fsbo/weights`: selected models, transferred
  as artifacts rather than Git files.
- `batch_recipe.json`, `batch_policy.json`, `fsbo_recipe.json`: algorithm and
  model identities. `pools/batch10_frozen_pool.json`: saved warm-start pools.
- `batch_code/run.py`: current `controller`, `worker`, `status`, `freeze`
  commands. The earlier `code/run.py` uses a different stage naming scheme.
- `resources.json`: hostnames, CPU/GPU assignments, PostgreSQL data paths and
  ports. The deployed allocation expanded after the original campaign README;
  use this captured file for the latest recorded topology.
- `orpo_batch10/runs/<task>/runtime.sqlite`, completion/checkpoint files:
  durable scientific state. Back these up consistently before a final move.

CPU regression checks after dependencies are restored:

```bash
cd /root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/batch_code
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /root/data/cys/envs/bolt_clean_venv/bin/python -m unittest test_retraining test_fsbo_batch test_batch
```

Training entry point on a prepared host:

```bash
/root/data/cys/envs/bolt_clean_venv/bin/python /root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/training/pipeline.py
```

The historical training pipeline includes promotion/launch steps. Read its
entry point before invoking it on a migrated topology. After configuring
new-host database/service endpoints, validating model pins and re-running
the live gates, the evaluation controller entry is:

```bash
/root/data/cys/envs/bolt_clean_venv/bin/python /root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/batch_code/run.py controller
```

The controller's host names must resolve to the intended new machines.
Provision the matching PostgreSQL/IMDb data and indexes, `pg_hint_plan`, LLM
replicas, both VAE serving paths and fresh service-token files before launch.
Data directories under `/dev/shm` are temporary RAM storage, not portable
database backups. A changed storage layout also changes runtime-comparison
provenance and must be recorded.

## Later query and peptide versions captured on 28 November

These versions supplement the historical entry points above. Hardware allocations
and recorded pause states describe the source deployment and need new-host validation.

| Campaign | Recorded behavior |
| --- | --- |
| `C/query_bolt30_lolbo_total800_20261123` | Saved BOLT30 starts, 77 further batches of ten, 800 total measurements per task; train80 VAE frozen during evaluation, online DKL/GP and Thompson sampling. Includes the canonicalization repair after the older failed canary. |
| `C/query_bolt_focus_migration_20261127` | Later query deployment; Nov28 operations add Optimizer5, with 30 SQL workers on CPU hosts4/5 and 16 VAE services on A16 GPUs. SemanticOPT intentionally paused during this allocation. |
| `C/query_continuation_operations_20261124` | Operations and continuation/recovery procedures. |
| `C/semanticopt_query_400_pilot800_20261123` | Imports the original200 proposals plus49Bao starts; first ten tasks extend to800, other190 to400, with independent journals/leases. No BOLT measurements are substituted. |
| `C/peptide_rgpe_top80_bootstrap100_test4_20261123` | Separate RGPE screening/bootstrap experiment; use its own policy and saved source-bank identities. |
| `/root/data/logs/peptide_fsbo_feasible_search_20261122` | Exact global-constraint filtering over5,000 Sobol decodes per acquisition. Decode yield below10% halves length; above50% doubles it, bounded by[0.0078125,1.6], up to three adjustments. Reuses Sobol candidates and feasibility masks. Recenter after an accepted VAE update; source refresh uses recent50+top200. |

Here `C=/root/data/cys/baseline/campaigns`. The feasible-search evaluator retains
native trust-region length updates but removes restarts. It uses feasible native
Thompson-sampling candidates first, filling shortages from infeasible candidates;
do not interpret screening as a guarantee that every proposal is feasible. Its
`code/pipeline.py` manages the ten-task rerun. Preserve its private evaluator/policy
alongside the earlier target-loss-gated implementation.

## Shared source and analyses (continued)

`/root/data/cys/llmoptimizer_benchmark/tools` contains the FSBO subprocess,
peptide local BO, transfer, RGPE, query encoder services, training builders and
analysis helpers. Native BOLT, APEXGo, BayesQO, FSBO, LOLBO and BORA source trees
are preserved with upstream licenses. LLaMA-Factory and shared training helpers
are under `/root/data/cys/LLMOptimizer`.

Peptide/query report scripts and small tabular summaries under `/root/reports`
are captured in the snapshot. Large result arrays, plots and complete histories
remain in the artifact transfer plan. A report's historical status is separate
from the live state of its campaign.
