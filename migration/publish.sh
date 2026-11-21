#!/usr/bin/env bash
set -euo pipefail
TASK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_REPO"
TASK_ACCOUNT=$(gh api user --jq .login)
if [[ "$TASK_ACCOUNT" != Y0306S ]]; then
  echo "Authenticate as Y0306S before publishing (current account: $TASK_ACCOUNT)." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo 'Commit reviewed changes before publishing.' >&2
  exit 1
fi
if ! gh repo view Y0306S/discreteDomains --json name >/dev/null 2>&1; then
  gh repo create Y0306S/discreteDomains --private \
    --description 'Peptide and query-domain experiment code, environments, and server migration guide'
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin https://github.com/Y0306S/discreteDomains.git
fi
# The source server has a global GitHub mirror rewrite. Disable it only for
# this push so authentication reaches GitHub directly.
gh auth setup-git --hostname github.com
git -c remote.origin.pushurl=https://Y0306S@github.com/Y0306S/discreteDomains.git push -u origin main
