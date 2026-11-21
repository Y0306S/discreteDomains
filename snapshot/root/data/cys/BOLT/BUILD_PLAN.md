# BOLT query-plan BO — build plan

**Server:** `cs-0b2cd-18deb-server`  
**Data root (use this for transfers):** `/root/data/cys/`  
**Do not put large files on `/root` overlay** (30G, ephemeral). Use lustre.

---

## Phase 0 — Transfer your data here (you do this before/during build)

Copy anything you need from your other machine into these paths:

| What you're bringing | Destination on this server |
|----------------------|----------------------------|
| **VAE / model weights** (e.g. `CEB_64.ckpt`) | `/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/` |
| **BAO init CSV** (if replacing) | `/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/initialization_data/` |
| **LLM checkpoints / HF weights** | `/root/data/cys/downloads/weights/` (create as needed) |
| **Benchmark outputs** (optional) | `/root/data/cys/llmoptimizer_benchmark/benchmark_data/domain_query/` |
| **Custom conda env exports** | `/root/data/cys/envs/` |

Example from your laptop (replace `<ssh-host>` with your platform jump/SSH target):

```bash
# New weights / checkpoints
rsync -avP ./CEB_64.ckpt <ssh-host>:/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/

# Or a whole weights folder
rsync -avP ./my_weights/ <ssh-host>:/root/data/cys/downloads/weights/
```

After transfer, link VAE if needed:

```bash
cd /root/data/cys/BOLT
bash scripts/link_query_vae_ckpt.sh /root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/vae/CEB_64.ckpt
```

**Already on disk (no transfer needed unless updating):**
- `CEB_64.ckpt` (~30 MB) — present
- `ceb_3k_bao_initialization.csv` — repaired in-repo
- IMDB PG clones: `pg_imdb`, `pg_imdb_b`, `pg_imdb_c3`, `pg_imdb_d`
- BOLT + BayesQO code under `/root/data/cys/`

---

## Phase 1 — Python env on lustre (`bolt_qp`)

```bash
conda create -p /root/data/cys/envs/bolt_qp python=3.12 pip -y
/root/data/cys/envs/bolt_qp/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
/root/data/cys/envs/bolt_qp/bin/pip install \
  numpy botorch gpytorch lightning fire wandb psycopg psycopg2-binary \
  "sqlglot==18.15.0" loguru pydot python-dotenv peewee tqdm matplotlib

# Postgres 16 binaries for oracle cluster
conda create -p /root/data/cys/envs/hpo_benchmark postgresql=16 -c conda-forge -y
```

Verify:

```bash
/root/data/cys/envs/bolt_qp/bin/python -c "import torch, botorch, gpytorch; print(torch.__version__, torch.cuda.device_count())"
```

---

## Phase 2 — IMDB Postgres cluster (4 ports for 4 BO workers)

```bash
export PG_BIN_DIR=/root/data/cys/envs/hpo_benchmark/bin
bash /root/data/cys/BOLT/scripts/start_imdb_postgres_cluster.sh
```

Expected ports:

| Worker slot | GPU | DB_PORT | Data dir |
|-------------|-----|---------|----------|
| gpu0 runner0 | 0 | 5432 | `pg_imdb` |
| gpu0 runner1 | 0 | 5433 | `pg_imdb_c3` |
| gpu1 runner0 | 1 | 5434 | `pg_imdb_b` |
| gpu1 runner1 | 1 | 5435 | `pg_imdb_d` |

---

## Phase 3 — BOLT workloads + verify

```bash
cd /root/data/cys/BOLT
bash scripts/setup_eval_pipeline.sh
python3 scripts/verify_bolt_setup.py --db-host 127.0.0.1 --db-port 5432
```

Fix any FAIL/WARN (VAE, CEB SQL extract, Postgres).

---

## Phase 4 — Launch 4 parallel BO workers (2 per GPU)

Uses BayesQO-style `parallel_launch.py` (per-worker `DB_PORT`), BOLT entrypoint, BAO init.

```bash
cd /root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/lolbo_scripts

export WANDB_MODE=disabled
export DB_HOST=127.0.0.1
export DB_USER=imdb
export DB_PASSWORD=imdb
export DB_NAME=imdb

mkdir -p /root/data/cys/logs

nohup /root/data/cys/envs/bolt_qp/bin/python parallel_launch.py \
  --num_per_gpu 2 \
  --task_file ../../tasks/ceb_tasks.txt \
  --init_w_bao True \
  --allow_cross_joins False \
  --track_with_wandb False \
  --max_n_oracle_calls 80 \
  >> /root/data/cys/logs/bolt_bo_parallel.log 2>&1 &

echo "BO launcher pid=$! log=/root/data/cys/logs/bolt_bo_parallel.log"
```

**Algorithm (exact BOLT path):** TuRBO + Thompson Sampling + censored DKL-GP in VAE latent space, `bsz=1`, BAO init (~49 plans/query), Postgres execution oracle.

**Not required:** `/root/data/cys/blockdata_archive/llmoptimizer_benchmark` (benchmark histories only).

---

## Phase 5 — Smoke test (optional, before full fleet)

Single task on GPU 0 / port 5432:

```bash
cd /root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/lolbo_scripts
export DB_HOST=127.0.0.1 DB_PORT=5432 DB_USER=imdb DB_PASSWORD=imdb WANDB_MODE=disabled
/root/data/cys/envs/bolt_qp/bin/python info_transformer_vae_optimization.py \
  --workload_name CEB_1A10 \
  --allow_cross_joins False \
  --init_w_bao True \
  --track_with_wandb False \
  --max_n_oracle_calls 5 \
  run_lolbo done
```

---

## Success criteria

- [ ] Weights landed under `/root/data/cys/...` (VAE at least 1 MB, not `.placeholder`)
- [ ] `bolt_qp` imports torch + botorch on 2 GPUs
- [ ] All 4 Postgres ports accept `psql -h 127.0.0.1 -p PORT -U imdb -d imdb`
- [ ] `verify_bolt_setup.py` passes query-plan checks
- [ ] `parallel_launch.py` running; log shows workers on distinct `DB_PORT`s
- [ ] First workload completes without CSV parse errors

---

## Logs

- BO parallel: `/root/data/cys/logs/bolt_bo_parallel.log`
- Postgres: `/root/data/cys/pg_imdb*/pg.log`
