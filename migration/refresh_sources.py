#!/usr/bin/env python3
"""Refresh the migration copy, preserving explicitly added scientific inputs."""
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location('migration_collector', REPO/'migration/collect.py')
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    inventory = REPO/'migration/inventory'
    previous = json.loads((inventory/'sources.json').read_text())
    # Main collector discovers current campaign roots and sanitizes credentials.
    collector.main()
    for path, item in previous.items():
        if path not in collector.COPIED:
            # Preserve deliberate large-input additions and previous versions.
            collector.COPIED[path] = item
    diagnostics = Path('/root/data/cys/tmp/peptide_train800_trust_region_20261121')
    if diagnostics.exists():
        for source in sorted(diagnostics.iterdir()):
            if source.is_file() and source.suffix in {'.py', '.sh', '.md'}:
                collector.copy_file(source)
    (inventory/'sources.json').write_text(json.dumps(collector.COPIED, indent=2, sort_keys=True)+'\n')
    summary = json.loads((inventory/'summary.json').read_text())
    summary.update(files=len(collector.COPIED), bytes=sum(r['bytes'] for r in collector.COPIED.values()),
                   refreshed_at='2026-11-28', sanitized_files=[p for p,r in collector.COPIED.items() if r['sanitized']])
    (inventory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Final inventory: {len(collector.COPIED)} files; preserved explicit input additions.')


if __name__ == '__main__':
    main()
