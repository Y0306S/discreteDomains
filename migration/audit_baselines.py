#!/usr/bin/env python3
"""Read-only baseline source/environment/input audit; never launches experiments."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restored', action='store_true', help='Also check original absolute paths on this host')
    args = parser.parse_args()
    catalog = json.loads((REPO/'migration/inventory/baselines.json').read_text())
    sources = json.loads((REPO/'migration/inventory/sources.json').read_text())
    environments = {r['name']: r for r in json.loads((REPO/'environments/index.json').read_text())}
    failures = []
    for method in catalog:
        for name in method['environments']:
            env = environments.get(name)
            if not env or env.get('error'):
                failures.append(f"{method['name']}: missing environment export {name}")
            elif args.restored and not (Path(env['prefix'])/'bin/python').is_file():
                failures.append(f"{method['name']}: missing interpreter {env['prefix']}")
        for path in method['entrypoints']:
            record = sources.get(path)
            if not record:
                failures.append('Missing source inventory: '+path)
            else:
                local = REPO/record['snapshot']
                if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != record['sha256']:
                    failures.append('Missing/changed snapshot: '+path)
            if args.restored and not Path(path).is_file():
                failures.append('Missing restored source: '+path)
        print(f"{method['name']}: {method['status']}; environments: {', '.join(method['environments'])}")
    if args.restored:
        # This is a presence check, not checkpoint identity or database readiness.
        for item in json.loads((REPO/'migration/inventory/primary-inputs.json').read_text()):
            if not Path(item['path']).exists():
                failures.append('Missing primary input: '+item['path'])
    if failures:
        raise SystemExit('\n'.join(sorted(set(failures))))
    print('PASS: baseline entrypoint hashes and environment exports' + ('; primary input paths present' if args.restored else ''))
    print('GPU execution, remote assets, PostgreSQL, service endpoints and scientific resume gates require new-host validation.')


if __name__ == '__main__':
    main()
