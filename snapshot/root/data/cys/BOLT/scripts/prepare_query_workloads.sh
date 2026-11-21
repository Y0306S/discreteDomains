#!/usr/bin/env bash
# Extract CEB-3k SQL workloads for query-plan optimization + fine-tuning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$ROOT/optimization/query_plans/query_plan_optimization/workload/ceb-3k/ceb_3k.zip"
DEST="$ROOT/optimization/query_plans/query_plan_optimization/workload/ceb-3k"

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP"
  exit 1
fi

# Quick sanity: already extracted if many .sql files exist
SQL_COUNT=$(find "$DEST" -name '*.sql' 2>/dev/null | wc -l)
if [[ "$SQL_COUNT" -gt 100 ]]; then
  echo "[prepare_query_workloads] Already extracted ($SQL_COUNT .sql files under $DEST)"
else
  echo "[prepare_query_workloads] Extracting $ZIP -> $DEST"
  unzip -o -q "$ZIP" -d "$DEST"
  SQL_COUNT=$(find "$DEST" -name '*.sql' | wc -l)
  echo "[prepare_query_workloads] Extracted $SQL_COUNT .sql files"
fi

# Fine-tuning scripts expect ./workload/ceb-3k/...
FT_DIR="$ROOT/fine-tuning/query_plans"
mkdir -p "$FT_DIR"
if [[ ! -e "$FT_DIR/workload" ]]; then
  ln -sfn "$ROOT/optimization/query_plans/query_plan_optimization/workload" "$FT_DIR/workload"
  echo "[prepare_query_workloads] Linked $FT_DIR/workload"
fi

# Symlink tasks list for convenience
if [[ ! -e "$FT_DIR/data/ceb_tasks.txt" ]]; then
  ln -sfn "$ROOT/optimization/query_plans/tasks/ceb_tasks.txt" "$FT_DIR/data/ceb_tasks.txt"
fi

echo "[prepare_query_workloads] Done."
#!/usr/bin/env bash
# Extract CEB-3k SQL workloads for query-plan optimization + fine-tuning.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP="$ROOT/optimization/query_plans/query_plan_optimization/workload/ceb-3k/ceb_3k.zip"
DEST="$ROOT/optimization/query_plans/query_plan_optimization/workload/ceb-3k"

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP"
  exit 1
fi

# Quick sanity: already extracted if many .sql files exist
SQL_COUNT=$(find "$DEST" -name '*.sql' 2>/dev/null | wc -l)
if [[ "$SQL_COUNT" -gt 100 ]]; then
  echo "[prepare_query_workloads] Already extracted ($SQL_COUNT .sql files under $DEST)"
else
  echo "[prepare_query_workloads] Extracting $ZIP -> $DEST"
  unzip -o -q "$ZIP" -d "$DEST"
  SQL_COUNT=$(find "$DEST" -name '*.sql' | wc -l)
  echo "[prepare_query_workloads] Extracted $SQL_COUNT .sql files"
fi

# Fine-tuning scripts expect ./workload/ceb-3k/...
FT_DIR="$ROOT/fine-tuning/query_plans"
mkdir -p "$FT_DIR"
if [[ ! -e "$FT_DIR/workload" ]]; then
  ln -sfn "$ROOT/optimization/query_plans/query_plan_optimization/workload" "$FT_DIR/workload"
  echo "[prepare_query_workloads] Linked $FT_DIR/workload"
fi

# Symlink tasks list for convenience
if [[ ! -e "$FT_DIR/data/ceb_tasks.txt" ]]; then
  ln -sfn "$ROOT/optimization/query_plans/tasks/ceb_tasks.txt" "$FT_DIR/data/ceb_tasks.txt"
fi

echo "[prepare_query_workloads] Done."
