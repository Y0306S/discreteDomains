#!/usr/bin/env bash
# Deploy conda envs packed from the GPU box onto a server that mounts /root/data
# but NOT /root/blockdata.
#
# Usage:
#   ./deploy_envs.sh                      # unpack all envs to $DEST
#   ./deploy_envs.sh openbox_qwen_eval    # unpack just one (or several)
#
# DEST defaults to /root/envs (LOCAL disk - strongly preferred: a conda env is
# ~100k small files, and running python out of Lustre makes every import slow).
# Set DEST=/root/data/envs only if the target server has no local scratch.

set -euo pipefail

PACKS="${PACKS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/packs}"
DEST="${DEST:-/root/envs}"

ALL=(taskset_eval openbox_eval hpo_benchmark llamafactory_qwen35 openbox_qwen_eval)
WANT=("${@:-${ALL[@]}}")
[ $# -gt 0 ] && WANT=("$@")

echo "packs dir : $PACKS"
echo "dest dir  : $DEST"
echo "envs      : ${WANT[*]}"
echo

mkdir -p "$DEST"

for e in "${WANT[@]}"; do
  tarball="$PACKS/$e.tar.gz"
  if [ ! -f "$tarball" ]; then
    echo "!! SKIP $e - no tarball at $tarball" >&2
    continue
  fi

  target="$DEST/$e"
  if [ -d "$target" ]; then
    echo "== $e already exists at $target - skipping (rm -rf it to redeploy)"
    continue
  fi

  echo "== unpacking $e -> $target"
  mkdir -p "$target"
  tar -xzf "$tarball" -C "$target"

  # Rewrites every baked-in prefix (shebangs, conda-meta, .pc files, RPATHs)
  # to wherever the env actually landed. Must run before first use.
  echo "== conda-unpack $e"
  "$target/bin/conda-unpack"

  # conda-unpack only rewrites the prefix conda-pack recorded
  # (/root/data/cys/blockdata_archive/miniconda3/envs/$e). These envs were built through
  # symlinks, so ~80 bin/ scripts have the *symlink* prefix
  # /root/miniconda3/envs/$e baked in instead, which conda-unpack leaves
  # alone -> `pip`, `wandb`, etc. fail with "required file not found".
  # Rewrite those stale prefixes to the real target here.
  echo "== fixing stale symlink-prefix shebangs in $e"
  fixed=0
  for stale in "/root/miniconda3/envs/$e" "/root/data/cys/blockdata_archive/miniconda3/envs/$e"; do
    while IFS= read -r f; do
      [ -n "$f" ] || continue
      sed -i "s|$stale|$target|g" "$f"
      fixed=$((fixed+1))
    done < <(grep -rIl -- "$stale" "$target/bin" 2>/dev/null || true)
  done
  echo "   rewrote $fixed file(s)"

  echo "== verifying $e"
  "$target/bin/python" -c "import sys; print('   python', sys.version.split()[0], '->', sys.executable)"
  if [ -x "$target/bin/pip" ]; then
    "$target/bin/pip" --version >/dev/null 2>&1 \
      && echo "   pip OK" \
      || echo "   !! pip still broken - inspect $target/bin/pip" >&2
  fi
  if grep -rIlq -- "/root/data/cys/blockdata_archive" "$target/bin" 2>/dev/null; then
    echo "   !! WARNING: leftover /root/data/cys/blockdata_archive refs in $target/bin:" >&2
    grep -rIl -- "/root/data/cys/blockdata_archive" "$target/bin" 2>/dev/null | sed 's/^/      /' >&2
  else
    echo "   no leftover /root/data/cys/blockdata_archive refs"
  fi
  echo
done

cat <<EOF

Done. To use an env without needing 'conda activate':

    $DEST/<env>/bin/python your_script.py

Or activate it properly:

    source $DEST/<env>/bin/activate
    ...
    source $DEST/<env>/bin/deactivate

To make them visible to an existing conda on this server:

    conda config --append envs_dirs $DEST

EOF
