# Domain transfer (protein, query opt, DL HPO)

Port of the LLM+TPE + experience loop to three new domains. **Do not write into** `benchmark_data/real_hpo_v2_mltest20_algo40/`.

| Domain | Search space | Oracle (first) | Histories |
|--------|----------------|----------------|-----------|
| protein | sequence VAE `z` (d=16) | BOLT/APEXGo scored libraries | fake BO via encode→pairs |
| query | join-order VAE `z` (d=16) | dummy EXPLAIN cost; Postgres later | CEB SQL + synthetic TPE |
| dl | named train HPs | zaxie `tune_report.json` lookup | real Optuna TPE logs |

## Layout

- `mindware/domains/` — `DomainAdapter`, sequence VAE, search loop
- `history_data/domain_{protein,query,dl}/` — OpenBox JSON + experience banks
- `benchmark_data/domain_{protein,query,dl}/` — eval / RL runs

## Bootstrap (fake BO → bank → split)

```bash
cd /root/data/cys/blockdata_archive/llmoptimizer_benchmark

python examples/benchmark/domains/convert_zaxie_histories.py --src /root/data/zaxie --min-scored 10
python examples/benchmark/domains/convert_taskset_histories.py --max-tasks 80

python examples/benchmark/domains/convert_protein_libraries.py
python examples/benchmark/domains/convert_query_histories.py --limit 24

python examples/benchmark/domains/build_domain_experience.py --domain dl
python examples/benchmark/domains/build_domain_experience.py --domain protein
python examples/benchmark/domains/build_domain_experience.py --domain query

python examples/benchmark/domains/make_tune_lock_split.py --domain dl
python examples/benchmark/domains/make_tune_lock_split.py --domain protein
python examples/benchmark/domains/make_tune_lock_split.py --domain query
```

## Smoke search

```bash
python examples/benchmark/domains/run_domain_optimizer.py --domain protein --optimizer cmaes --n-trials 12 --dummy
python examples/benchmark/domains/run_domain_optimizer.py --domain query --optimizer cmaes --n-trials 12 --dummy
python examples/benchmark/domains/run_domain_optimizer.py --domain dl --optimizer tpe --n-trials 12
python examples/benchmark/domains/run_domain_optimizer.py --domain protein --optimizer ptllm_cmaes --n-trials 12 --dummy
```

## RL

TUNE-only rollouts, then the existing REINFORCE trainer with a **per-domain LoRA** (`lora_adapters.json`):

```bash
python examples/benchmark/domains/run_domain_rl_rollout.py --domain protein --task bolt_peptides --n-trials 12
python examples/benchmark/train_p5_reinforce.py --buffer-root benchmark_data/domain_protein/rl_rollouts
```

LOCK 80-trial eval must be real search runs, not fake library order.

## BOLT VAE states + 80-trial BO (protein & query)

The numpy `SequenceVAE` (d=16) is only for bootstrap / dummy tests. Suite search uses the **BOLT** checkpoints:

- Protein: UniRef InfoTransformer, `z ∈ R^256` (`optimization/peptides/uniref_vae/saved_models/...epoch_118.pkl`), APEX mean-MIC oracle.
- Query: `VAEModule` `CEB_64.ckpt`, `z ∈ R^64` (`bn_size=2 × d_neck=32`, same dim as BOLT LOLBO). Postgres EXPLAIN is optional; dummy plan cost is the default until IMDB is up.

```bash
# hpo_benchmark env (CUDA torch + lightning)
export PYTHONPATH=/root/data/cys/blockdata_archive/llmoptimizer_benchmark

# Query ckpt is a placeholder until this trains it
python examples/benchmark/domains/train_bolt_query_vae.py --max-epochs 12

# 100 datasets × 80 TPE trials each (resume with --skip-done)
python examples/benchmark/domains/run_bolt_bo_suite.py --domain protein --n-datasets 100 --n-trials 80 --optimizer tpe
python examples/benchmark/domains/run_bolt_bo_suite.py --domain query --n-datasets 100 --n-trials 80 --optimizer tpe

python examples/benchmark/domains/build_domain_experience.py --domain protein
python examples/benchmark/domains/build_domain_experience.py --domain query
python examples/benchmark/domains/make_tune_lock_split.py --domain protein
python examples/benchmark/domains/make_tune_lock_split.py --domain query
```

Outputs: `history_data/domain_{protein,query}/{fitness,cost}/<task>/tpe_bolt.json` and `benchmark_data/domain_{protein,query}/bo80_bolt/`.
# Domain transfer (protein, query opt, DL HPO)

Port of the LLM+TPE + experience loop to three new domains. **Do not write into** `benchmark_data/real_hpo_v2_mltest20_algo40/`.

| Domain | Search space | Oracle (first) | Histories |
|--------|----------------|----------------|-----------|
| protein | sequence VAE `z` (d=16) | BOLT/APEXGo scored libraries | fake BO via encode→pairs |
| query | join-order VAE `z` (d=16) | dummy EXPLAIN cost; Postgres later | CEB SQL + synthetic TPE |
| dl | named train HPs | zaxie `tune_report.json` lookup | real Optuna TPE logs |

## Layout

- `mindware/domains/` — `DomainAdapter`, sequence VAE, search loop
- `history_data/domain_{protein,query,dl}/` — OpenBox JSON + experience banks
- `benchmark_data/domain_{protein,query,dl}/` — eval / RL runs

## Bootstrap (fake BO → bank → split)

```bash
cd /root/data/cys/blockdata_archive/llmoptimizer_benchmark

python examples/benchmark/domains/convert_zaxie_histories.py --src /root/data/zaxie --min-scored 10
python examples/benchmark/domains/convert_taskset_histories.py --max-tasks 80

python examples/benchmark/domains/convert_protein_libraries.py
python examples/benchmark/domains/convert_query_histories.py --limit 24

python examples/benchmark/domains/build_domain_experience.py --domain dl
python examples/benchmark/domains/build_domain_experience.py --domain protein
python examples/benchmark/domains/build_domain_experience.py --domain query

python examples/benchmark/domains/make_tune_lock_split.py --domain dl
python examples/benchmark/domains/make_tune_lock_split.py --domain protein
python examples/benchmark/domains/make_tune_lock_split.py --domain query
```

## Smoke search

```bash
python examples/benchmark/domains/run_domain_optimizer.py --domain protein --optimizer cmaes --n-trials 12 --dummy
python examples/benchmark/domains/run_domain_optimizer.py --domain query --optimizer cmaes --n-trials 12 --dummy
python examples/benchmark/domains/run_domain_optimizer.py --domain dl --optimizer tpe --n-trials 12
python examples/benchmark/domains/run_domain_optimizer.py --domain protein --optimizer ptllm_cmaes --n-trials 12 --dummy
```

## RL

TUNE-only rollouts, then the existing REINFORCE trainer with a **per-domain LoRA** (`lora_adapters.json`):

```bash
python examples/benchmark/domains/run_domain_rl_rollout.py --domain protein --task bolt_peptides --n-trials 12
python examples/benchmark/train_p5_reinforce.py --buffer-root benchmark_data/domain_protein/rl_rollouts
```

LOCK 80-trial eval must be real search runs, not fake library order.

## BOLT VAE states + 80-trial BO (protein & query)

The numpy `SequenceVAE` (d=16) is only for bootstrap / dummy tests. Suite search uses the **BOLT** checkpoints:

- Protein: UniRef InfoTransformer, `z ∈ R^256` (`optimization/peptides/uniref_vae/saved_models/...epoch_118.pkl`), APEX mean-MIC oracle.
- Query: `VAEModule` `CEB_64.ckpt`, `z ∈ R^64` (`bn_size=2 × d_neck=32`, same dim as BOLT LOLBO). Postgres EXPLAIN is optional; dummy plan cost is the default until IMDB is up.

```bash
# hpo_benchmark env (CUDA torch + lightning)
export PYTHONPATH=/root/data/cys/blockdata_archive/llmoptimizer_benchmark

# Query ckpt is a placeholder until this trains it
python examples/benchmark/domains/train_bolt_query_vae.py --max-epochs 12

# 100 datasets × 80 TPE trials each (resume with --skip-done)
python examples/benchmark/domains/run_bolt_bo_suite.py --domain protein --n-datasets 100 --n-trials 80 --optimizer tpe
python examples/benchmark/domains/run_bolt_bo_suite.py --domain query --n-datasets 100 --n-trials 80 --optimizer tpe

python examples/benchmark/domains/build_domain_experience.py --domain protein
python examples/benchmark/domains/build_domain_experience.py --domain query
python examples/benchmark/domains/make_tune_lock_split.py --domain protein
python examples/benchmark/domains/make_tune_lock_split.py --domain query
```

Outputs: `history_data/domain_{protein,query}/{fitness,cost}/<task>/tpe_bolt.json` and `benchmark_data/domain_{protein,query}/bo80_bolt/`.
