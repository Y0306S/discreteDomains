#!/usr/bin/env bash
# Gracefully stop bsz10 r100 fleet (SIGTERM → saves VAE checkpoint mid-task).
set -euo pipefail

LOGDIR="${LOGDIR:-/root/data/logs}"
PIDFILE="$LOGDIR/bolt_bsz10_r100_fleet.pids"
WAIT_SEC="${WAIT_SEC:-600}"

if [[ ! -f "$PIDFILE" ]]; then
  echo "[stop-bsz10-r100] no pidfile at $PIDFILE"
  exit 0
fi

mapfile -t PIDS <"$PIDFILE"
alive=()
for pid in "${PIDS[@]}"; do
  if kill -0 "$pid" 2>/dev/null; then
    alive+=("$pid")
  fi
done

if ((${#alive[@]} == 0)); then
  echo "[stop-bsz10-r100] no running workers"
  exit 0
fi

echo "[stop-bsz10-r100] sending SIGTERM to: ${alive[*]} (wait up to ${WAIT_SEC}s for VAE checkpoint)"
kill -TERM "${alive[@]}" 2>/dev/null || true

deadline=$(( $(date +%s) + WAIT_SEC ))
while (( $(date +%s) < deadline )); do
  running=0
  for pid in "${alive[@]}"; do
    kill -0 "$pid" 2>/dev/null && running=$((running + 1)) || true
  done
  if (( running == 0 )); then
    echo "[stop-bsz10-r100] all workers exited cleanly"
    exit 0
  fi
  sleep 5
done

echo "[stop-bsz10-r100] timeout; still running:"
ps -p "$(IFS=,; echo "${alive[*]}")" -o pid,etime,cmd 2>/dev/null || true
exit 1
