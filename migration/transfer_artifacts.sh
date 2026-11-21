#!/usr/bin/env bash
# Run on the OLD server. The default previews; --apply performs the transfer.
set -euo pipefail
TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 USER@NEW-SERVER:/ABSOLUTE/STAGING/DIR/ [--apply]" >&2
  exit 2
fi
TASK_DEST=$1
TASK_MODE=${2:---dry-run}
if [[ "$TASK_DEST" != *:* || "$TASK_DEST" == -* || ( "$TASK_MODE" != --apply && "$TASK_MODE" != --dry-run ) ]]; then
  echo 'Expected an SSH rsync destination and optional --apply.' >&2
  exit 2
fi
TASK_PREVIEW=(--dry-run)
if [[ "$TASK_MODE" == --apply ]]; then TASK_PREVIEW=(); fi
# PostgreSQL data, sockets, machine credentials and runtime service tokens are
# restored separately. The inventory intentionally keeps scientific checkpoints
# out of Git. Scientific symlink targets are included explicitly in the plan.
# Preserve links instead of following staged-directory links back into a parent.
rsync -arR --partial --info=progress2,stats2 \
  "${TASK_PREVIEW[@]}" \
  --files-from="$TASK_REPO/migration/inventory/artifact-roots.txt" \
  --exclude-from="$TASK_REPO/migration/inventory/source-excludes.txt" \
  --exclude='.git/' --exclude='__pycache__/' --exclude='wandb/' \
  --exclude='*.token' --exclude='*.secret' --exclude='.env' \
  --exclude='*token.txt' --exclude='*token.json' \
  --exclude='*.lock' --exclude='*.sock' --exclude='*.pid' \
  --exclude='pgdata/' --exclude='pg_data/' --exclude='base/' \
  --exclude='global/' --exclude='pg_wal/' --exclude='pg_tblspc/' \
  --exclude='postmaster.*' --exclude='.s.PGSQL.*' \
  / "$TASK_DEST"
