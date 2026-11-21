"""Pack completed ML runs, or verify and restore one immutable run archive."""

import argparse
import fcntl
import json
from pathlib import Path
import signal

from ml_run_archive import ArchiveError, archive_path, load_json, manifest, pack_run, restore, task_key, verify


def pack_campaign(root, *, cleanup=False, method=None, task=None, limit=None, host=None):
    protocol = load_json(root / 'protocol.json')
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    reports = []
    try:
        for name in protocol['methods']:
            if method and name != method:
                continue
            for item in load_json(root / 'queues' / (name + '.json')):
                if interrupted or (limit is not None and len(reports) >= limit):
                    return reports
                key = task_key(item)
                if task and key != task:
                    continue
                state = load_json(root / 'state' / name / (key + '.json'), {})
                if host and state.get('host') != host:
                    continue
                progress = load_json(root / 'sessions' / name / key / 'progress.json', {})
                if state.get('status') not in ('complete', 'complete_with_failures') or not progress.get('budget_exhausted'):
                    continue
                try:
                    packed = archive_path(root, name, item)
                    data = manifest(packed) if packed.exists() else None
                    if cleanup and data and not any((root / p).exists() or (root / p).is_symlink()
                                                    for p in data['source_roots']):
                        result = dict(status='already_compacted', method=name, task=key, archive=str(packed))
                    else:
                        result = pack_run(root, name, item, cleanup=cleanup)
                except (ArchiveError, OSError, ValueError) as exc:
                    result = dict(status='error', method=name, task=key, error=str(exc))
                reports.append(result)
                print(json.dumps(result), flush=True)
    finally:
        for signum, handler in old.items():
            signal.signal(signum, handler)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    pack = sub.add_parser('pack')
    pack.add_argument('--campaign', type=Path, required=True)
    pack.add_argument('--method')
    pack.add_argument('--host')
    pack.add_argument('--task')
    pack.add_argument('--limit', type=int)
    pack.add_argument('--cleanup', action='store_true')
    for action in ('verify', 'restore'):
        command = sub.add_parser(action)
        command.add_argument('--archive', type=Path, required=True)
        if action == 'restore':
            command.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'pack':
        path = args.campaign / 'locks/archive_campaign.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reports = pack_campaign(args.campaign, cleanup=args.cleanup, method=args.method,
                                    task=args.task, limit=args.limit, host=args.host)
        if any(r['status'] == 'error' for r in reports):
            raise SystemExit(1)
    else:
        data = verify(args.archive) if args.action == 'verify' else restore(args.archive, args.destination)
        print(json.dumps(dict(status='verified' if args.action == 'verify' else 'restored',
                              run_id=data['run_id'], progress=data['progress'])))


if __name__ == '__main__':
    main()
