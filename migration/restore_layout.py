#!/usr/bin/env python3
"""Restore the original path layout. Defaults to a preview; never overwrites."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True,
                        help='Filesystem root, e.g. / on the NEW server, or a staging directory')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    destination = args.destination.resolve()
    sources = json.loads((REPO/'migration/inventory/sources.json').read_text())
    pending = []
    conflicts = []
    for absolute, record in sources.items():
        source = REPO/record['snapshot']
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise SystemExit('Snapshot hash mismatch: '+str(source))
        target = destination/absolute.lstrip('/')
        # Do not follow pre-existing symlinks out of a staging destination.
        if any(p.is_symlink() for p in [target, *target.parents] if p != destination.parent):
            conflicts.append(str(target)+' (symlink in destination path)')
        elif target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                conflicts.append(str(target))
        else:
            pending.append((source, target))
    print(f'{len(pending)} files to restore, {len(conflicts)} conflicts')
    if conflicts:
        print('\n'.join(conflicts[:20]))
        raise SystemExit('Refusing to overwrite existing paths. Use a fresh destination.')
    if not args.apply:
        print('Preview only. Add --apply to copy files.')
        return
    for source, target in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print('Source layout restored. Restore artifacts/environments before running experiments.')


if __name__ == '__main__':
    main()
