#!/usr/bin/env bash
# One-shot local prep for BOLT evaluation (peptides + query plans).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

bash scripts/prepare_query_workloads.sh
bash scripts/prepare_peptide_assets.sh

# Python env for query plans (optional if already installed)
QP="$ROOT/optimization/query_plans"
if command -v uv >/dev/null 2>&1; then
  echo "[setup] uv sync in optimization/query_plans"
  (cd "$QP" && uv sync -U)
elif command -v poetry >/dev/null 2>&1; then
  echo "[setup] poetry install in optimization/query_plans"
  (cd "$QP" && poetry install)
else
  echo "[setup] WARN: install uv or poetry for query_plan deps (see optimization/query_plans/pyproject.toml)"
fi

python3 scripts/verify_bolt_setup.py || true

cat <<'EOF'

Next manual steps (not automated):
  1. Query-plan VAE: obtain CEB_64.ckpt (not in public BOLT repo) and run:
       bash scripts/link_query_vae_ckpt.sh /path/to/CEB_64.ckpt
     Contact BOLT authors or train from planner-generated plans (see scripts/SETUP_PIPELINE.md).

  2. IMDB Postgres for query oracle:
       See optimization/query_plans/query_plan_optimization/imdb_postgres/README.md
       Set DB_HOST, DB_USER, DB_PASSWORD (default imdb/imdb).

  3. Smoke tests:
       Peptides: cd optimization/peptides/lolbo_scripts && bash run_single_seed_0.sh
       Query:    cd optimization/query_plans/query_plan_optimization/lolbo_scripts && \\
         python info_transformer_vae_optimization.py --workload_name CEB_1A10 ...

EOF
