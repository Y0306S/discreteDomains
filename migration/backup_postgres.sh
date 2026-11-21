#!/usr/bin/env bash
# Uses libpq's PGHOST/PGPORT/PGUSER/PGDATABASE/PGPASSFILE environment.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: PGHOST=... PGPORT=... PGUSER=... PGDATABASE=imdb bash migration/backup_postgres.sh /outside/repo/imdb.dump' >&2
  exit 2
fi
TASK_DUMP=$1
TASK_PG_BIN=${DISCRETE_PG_BIN:-/root/data/cys/envs/pg16/bin}
if [[ -e "$TASK_DUMP" ]]; then echo 'Refusing to overwrite a dump.' >&2; exit 1; fi
umask 077
"$TASK_PG_BIN/pg_dump" --format=custom --file="$TASK_DUMP"
"$TASK_PG_BIN/pg_restore" --list "$TASK_DUMP" > "$TASK_DUMP.contents.txt"
sha256sum "$TASK_DUMP" > "$TASK_DUMP.sha256"
