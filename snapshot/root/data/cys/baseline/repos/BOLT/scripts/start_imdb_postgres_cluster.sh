#!/usr/bin/env bash
# Start 8 isolated IMDB Postgres instances for 4 GPUs × 2 BO workers.
set -euo pipefail

mkdir -p /usr/local/pgsql/bin
# Prefer postgres client/server from lustre env; fall back to system install.
PG_BIN_DIR="${PG_BIN_DIR:-}"
if [[ -z "$PG_BIN_DIR" && -x /root/data/cys/envs/pg16/bin/postgres ]]; then
  PG_BIN_DIR=/root/data/cys/envs/pg16/bin
elif [[ -z "$PG_BIN_DIR" && -x /root/data/cys/envs/hpo_benchmark/bin/postgres ]]; then
  PG_BIN_DIR=/root/data/cys/envs/hpo_benchmark/bin
elif [[ -z "$PG_BIN_DIR" && -x /usr/lib/postgresql/16/bin/postgres ]]; then
  PG_BIN_DIR=/usr/lib/postgresql/16/bin
fi
if [[ -z "$PG_BIN_DIR" || ! -x "$PG_BIN_DIR/postgres" ]]; then
  echo "Install PostgreSQL 16 first, or set PG_BIN_DIR to a bin dir with postgres/pg_ctl/psql."
  echo "  conda create -p /root/data/cys/envs/pg16 postgresql=16 -c conda-forge -y"
  exit 1
fi
# Use conda binaries in place so RPATH finds libxml2.so.16. Copy only as a convenience PATH.
for bin in postgres pg_ctl psql pg_isready; do
  ln -sfn "$PG_BIN_DIR/$bin" "/usr/local/pgsql/bin/$bin"
done
chmod -R a+rx /usr/local/pgsql
PG_LIB_DIR="$(cd "$PG_BIN_DIR/../lib" && pwd)"
chmod o+rx /root /root/data /root/data/cys /root/data/cys/envs "$(dirname "$PG_BIN_DIR")" "$PG_BIN_DIR" "$PG_LIB_DIR" 2>/dev/null || true

groupadd -g 10000 pgimdb 2>/dev/null || true
useradd -u 10000 -g 10000 -M -s /bin/bash pgimdb 2>/dev/null || true

PGCTL="$PG_BIN_DIR/pg_ctl"
PSQL="$PG_BIN_DIR/psql"
PGENV="PATH=/usr/local/pgsql/bin:\$PATH LD_LIBRARY_PATH=${PG_LIB_DIR}:\${LD_LIBRARY_PATH:-}"

start_one() {
  local dat=$1 port=$2
  echo "=== $dat port=$port ==="
  su -s /bin/bash pgimdb -c "rm -f '$dat/postmaster.pid'" || true
  if su -s /bin/bash pgimdb -c "$PGENV $PGCTL -D '$dat' status" >/dev/null 2>&1; then
    echo "already running"
  else
    su -s /bin/bash pgimdb -c "$PGENV $PGCTL -D '$dat' -l '$dat/pg.log' start"
    sleep 2
  fi
  su -s /bin/bash pgimdb -c "$PGENV $PGCTL -D '$dat' status"
  env PATH=/usr/local/pgsql/bin:$PATH LD_LIBRARY_PATH=${PG_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
    $PSQL -h 127.0.0.1 -p "$port" -U imdb -d imdb -c "SELECT 1 AS ok"
}

start_one /root/data/cys/pg_imdb 5432
start_one /root/data/cys/pg_imdb_c3 5433
start_one /root/data/cys/pg_imdb_b 5434
start_one /root/data/cys/pg_imdb_d 5435
start_one /root/data/cys/pg_imdb_e 5436
start_one /root/data/cys/pg_imdb_f 5437
start_one /root/data/cys/pg_imdb_g 5438
start_one /root/data/cys/pg_imdb_h 5439

echo "All IMDB Postgres instances ready."
