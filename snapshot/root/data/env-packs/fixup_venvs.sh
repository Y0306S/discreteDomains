#!/usr/bin/env bash
# Relocate the two overlay venvs (bolt_clean_venv, fanova) after their parent
# conda env has been deployed by deploy_envs.sh.
#
# A stdlib venv hardcodes its parent interpreter in pyvenv.cfg and in every
# bin/ shebang, so it must be repointed at the parent's NEW location.
#
# Usage:
#   ENVS=/root/envs DEST=/root/venvs ./fixup_venvs.sh
#
#   ENVS = where deploy_envs.sh put the conda envs (default /root/envs)
#   DEST = where to unpack the venvs           (default /root/venvs)

set -euo pipefail

PACKS="${PACKS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/packs}"
ENVS="${ENVS:-/root/envs}"
DEST="${DEST:-/root/venvs}"

# venv name -> parent conda env name
declare -A PARENT=(
  [bolt_clean_venv]=openbox_qwen_eval
  [fanova]=hpo_benchmark
)

mkdir -p "$DEST"

for v in "${!PARENT[@]}"; do
  tarball="$PACKS/$v.venv.tar.gz"
  parent="$ENVS/${PARENT[$v]}"
  target="$DEST/$v"

  if [ ! -f "$tarball" ]; then
    echo "!! SKIP $v - no tarball at $tarball" >&2; continue
  fi
  if [ ! -x "$parent/bin/python" ]; then
    echo "!! SKIP $v - parent env not deployed at $parent (run deploy_envs.sh ${PARENT[$v]} first)" >&2
    continue
  fi
  if [ -d "$target" ]; then
    echo "== $v already at $target - skipping (rm -rf it to redeploy)"; continue
  fi

  echo "== unpacking $v -> $target"
  mkdir -p "$(dirname "$target")"
  tar -xzf "$tarball" -C "$(dirname "$target")"

  pyver="$("$parent/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"

  echo "== rewriting pyvenv.cfg for $v (parent: $parent, python $pyver)"
  cat > "$target/pyvenv.cfg" <<EOF
home = $parent/bin
include-system-site-packages = $([ "$v" = bolt_clean_venv ] && echo true || echo false)
version = $("$parent/bin/python" -c 'import platform;print(platform.python_version())')
executable = $parent/bin/python$pyver
command = $parent/bin/python -m venv $([ "$v" = bolt_clean_venv ] && echo '--system-site-packages ')$target
EOF

  echo "== rewriting bin/ shebangs for $v"
  n=0
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    sed -i -E "1s|^#!.*/(python[0-9.]*)$|#!$target/bin/\1|" "$f"
    sed -i "s|/root/data/cys/blockdata_archive/miniconda3/envs/${PARENT[$v]}|$parent|g; s|/root/miniconda3/envs/${PARENT[$v]}|$parent|g" "$f"
    n=$((n+1))
  done < <(grep -rIl -E '/root/(blockdata/)?miniconda3|^#!' "$target/bin" 2>/dev/null || true)
  echo "   rewrote $n file(s)"

  # the venv's python/python3 are symlinks to the parent interpreter
  for link in python python3 "python$pyver"; do
    [ -e "$target/bin/$link" ] || continue
    if [ -L "$target/bin/$link" ]; then
      ln -sfn "$parent/bin/python$pyver" "$target/bin/$link"
    fi
  done

  echo "== verifying $v"
  "$target/bin/python" -c "import sys; print('   python', sys.version.split()[0]); print('   prefix', sys.prefix); print('   base  ', sys.base_prefix)"
  if [ "$v" = bolt_clean_venv ]; then
    "$target/bin/python" -c "import sitecustomize, gpytorch.lazy as g; print('   gpytorch LazyTensor shim:', hasattr(g,'LazyTensor'))" 2>&1 | tail -2
  else
    "$target/bin/python" -c "import fanova, ConfigSpace; print('   fanova + ConfigSpace import OK')" 2>&1 | tail -2
  fi
  echo
done

echo "Done."
