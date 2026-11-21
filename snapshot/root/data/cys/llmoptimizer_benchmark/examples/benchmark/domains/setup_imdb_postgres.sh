#!/usr/bin/env bash
# Restore Harvard Dataverse imdb_pg11 into a local Postgres (no Docker required).
# Data dir lives on /root/data (lustre) because the rootfs overlay is too small.
set -euo pipefail

DUMP="${DUMP:-/root/data/cys/BOLT/optimization/query_plans/query_plan_optimization/imdb_postgres/imdb_pg11}"
PGDATA="${PGDATA:-/root/data/cys/pg_imdb}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
LOG="$PGDATA/pg.log"

expect_bytes=1277543282
expect_md5=df3e976b235288005cb410cea09a115f

if [[ ! -f "$DUMP" ]]; then
  echo "Missing dump: $DUMP"
  exit 1
fi
sz=$(stat -c%s "$DUMP")
if [[ "$sz" -lt 1000000000 ]]; then
  echo "Dump still downloading? size=$sz expected=$expect_bytes"
  exit 1
fi
echo "[setup_imdb] dump size=$sz"
if command -v md5sum >/dev/null; then
  got=$(md5sum "$DUMP" | awk '{print $1}')
  echo "[setup_imdb] md5=$got (expected $expect_md5)"
  if [[ "$got" != "$expect_md5" ]]; then
    echo "WARNING: md5 mismatch; continuing anyway"
  fi
fi

export PGDATA PGPORT
if ! command -v initdb >/dev/null || ! command -v pg_restore >/dev/null; then
  echo "[setup_imdb] installing postgresql..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq postgresql postgresql-contrib postgresql-client >/dev/null
fi
BINDIR=$(pg_config --bindir 2>/dev/null || echo /usr/lib/postgresql/*/bin | awk '{print $1}')
# fall back to PATH
INITDB=$(command -v initdb || ls /usr/lib/postgresql/*/bin/initdb | tail -1)
PG_CTL=$(command -v pg_ctl || ls /usr/lib/postgresql/*/bin/pg_ctl | tail -1)
PSQL=$(command -v psql)
PG_RESTORE=$(command -v pg_restore)

mkdir -p "$PGDATA"
if [[ ! -f "$PGDATA/PG_VERSION" ]]; then
  echo "[setup_imdb] initdb $PGDATA"
  "$INITDB" -D "$PGDATA" --auth=trust --username="$PGUSER"
  cat >> "$PGDATA/postgresql.conf" <<EOF
listen_addresses = '127.0.0.1'
port = $PGPORT
shared_buffers = 2GB
work_mem = 64MB
maintenance_work_mem = 1GB
max_parallel_workers_per_gather = 4
EOF
  echo "host all all 127.0.0.1/32 trust" >> "$PGDATA/pg_hba.conf"
fi

if ! "$PG_CTL" -D "$PGDATA" status >/dev/null 2>&1; then
  echo "[setup_imdb] starting postgres on :$PGPORT"
  "$PG_CTL" -D "$PGDATA" -l "$LOG" -o "-p $PGPORT" start
  sleep 2
fi

$PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='imdb'" | grep -q 1 || \
  $PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d postgres -c "CREATE DATABASE imdb;"
$PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d postgres -tc "SELECT 1 FROM pg_roles WHERE rolname='imdb'" | grep -q 1 || \
  $PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d postgres -c "CREATE ROLE imdb LOGIN PASSWORD 'imdb' SUPERUSER;"

if ! $PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d imdb -tc "SELECT to_regclass('public.title')" | grep -q title; then
  echo "[setup_imdb] pg_restore (this takes several minutes)..."
  $PG_RESTORE -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d imdb --no-owner --role=imdb -j 4 "$DUMP" || true
  $PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d imdb -c "ANALYZE;"
fi

# optional pg_hint_plan
if $PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d imdb -c "CREATE EXTENSION IF NOT EXISTS pg_hint_plan;" 2>/dev/null; then
  echo "[setup_imdb] pg_hint_plan enabled"
else
  echo "[setup_imdb] pg_hint_plan not available (Leading hints will be ignored; EXPLAIN still works)"
fi

echo "[setup_imdb] tables:"
$PSQL -h 127.0.0.1 -p "$PGPORT" -U "$PGUSER" -d imdb -c "\dt" | head -30
echo "DSN=postgresql://imdb:imdb@127.0.0.1:${PGPORT}/imdb"
export QUERYOPT_PG_DSN="postgresql://imdb:imdb@127.0.0.1:${PGPORT}/imdb"
echo "$QUERYOPT_PG_DSN"
