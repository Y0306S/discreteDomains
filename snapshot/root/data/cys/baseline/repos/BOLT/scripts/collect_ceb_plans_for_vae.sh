#!/usr/bin/env bash
# Populate workload.db with CEB alias plans (needs IMDB Postgres) and export encoded CSVs for VAE training.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QP="$ROOT/optimization/query_plans"
TD="$QP/query_plan_optimization/training_data"
ENC="$TD/encoded_ceb"

: "${DB_HOST:=127.0.0.1}"
: "${DB_PORT:=5432}"
: "${DB_USER:=imdb}"
: "${DB_PASSWORD:=imdb}"
: "${DB_NAME:=imdb}"
: "${PLANNER_WORKERS:=16}"

export DB_HOST DB_PORT DB_USER DB_PASSWORD DB_NAME PLANNER_WORKERS

cd "$QP/query_plan_optimization"
export PYTHONPATH="$QP/query_plan_optimization:${PYTHONPATH:-}"
if [[ -x /root/data/cys/envs/bolt_qp/bin/python ]]; then
  RUN=(/root/data/cys/envs/bolt_qp/bin/python)
elif [[ -x /root/data/cys/envs/hpo_benchmark/bin/python ]]; then
  RUN=(/root/data/cys/envs/hpo_benchmark/bin/python)
elif [[ -x "$QP/.venv/bin/python" ]]; then
  RUN=("$QP/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  RUN=(uv run --directory "$QP" python)
else
  RUN=(poetry -C "$QP" run python)
fi

echo "[collect_ceb] DB_HOST=$DB_HOST workers=$PLANNER_WORKERS python=${RUN[*]}"
echo "[collect_ceb] 1) Generate alias join queries + EXPLAIN plans into workload.db (long; needs PG)"
"${RUN[@]}" - <<'PY'
from workload.workloads import get_workload_set
from workload.schema import build_alias_join_graph
from training_data.gen_alias_workload import gen_alias_join_queries, generate_plans
from training_data.storage import PlanType

load = get_workload_set("CEB_3K")
g = build_alias_join_graph(load.join_graph, 3)
gen_alias_join_queries(g, "CEB")
for pt in PlanType:
    generate_plans(pt, "CEB")
print("plan collection done")
PY

echo "[collect_ceb] 2) Export encoded plans -> $ENC"
mkdir -p "$ENC"
"${RUN[@]}" - <<'PY'
from codec.codec import AliasesCodec
from workload.schema import build_table_order
from workload.workloads import IMDB_SCHEMA_PATH
from training_data.codec import encode_alias_plans

codec = AliasesCodec(build_table_order(IMDB_SCHEMA_PATH))
encode_alias_plans(codec, schema="CEB", directory="training_data/encoded_ceb")
PY

echo "[collect_ceb] Done. CSV count: $(ls -1 "$ENC"/*.csv 2>/dev/null | wc -l)"
