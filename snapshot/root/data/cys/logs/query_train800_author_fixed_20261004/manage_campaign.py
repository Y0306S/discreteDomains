"""Finalize the repaired snapshot and operate its dedicated database services."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
OLD = Path('/root/data/cys/logs/query_train800_author_bo_20261002')
sys.path.insert(0, str(ROOT / 'code'))
import query_train800_fleet as fleet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('finalize', 'databases', 'cleanup'))
    parser.add_argument('--host')
    args = parser.parse_args()
    plan = fleet.read(ROOT / 'plan.json')
    if args.action == 'finalize':
        if list((ROOT / 'outputs').iterdir()):
            raise RuntimeError('Cannot revise a running campaign')
        plan['cache_gp_policy'] = 'retain_existing_plan_duplicate_filter'
        plan['cached_batch_counters'] = 'update_progress_and_trust_region_once_per_acquisition_batch'
        plan['scope_note'] = 'Apply the explicitly requested cached-batch counter fix. Restoring cached GP observations is not included.'
        plan['database_storage'] = 'Cold copies on local ext4 (controller) and tmpfs (remote); original sources preserved'
        plan['files_sha256'] = {str(p.relative_to(ROOT)): fleet.digest(p)
                               for directory in ('code', 'native') for p in sorted((ROOT / directory).rglob('*'))
                               if p.is_file() and '__pycache__' not in p.parts}
        fleet.write(ROOT / 'plan.json', plan)
        print(json.dumps(dict(status='finalized', hashed_files=len(plan['files_sha256']), policy=plan['cache_gp_policy'])))
    elif args.action == 'databases':
        for worker in plan['workers']:
            if worker['host'] == args.host:
                fleet.db_start(ROOT, worker['name'])
    else:
        # Only the retired campaign's result trees are eligible for deletion.
        targets = [OLD / 'outputs', OLD / 'smoke']
        entries = []
        for target in targets:
            if target.is_symlink() or target.resolve().parent != OLD.resolve():
                raise RuntimeError('Unexpected deletion target')
            files = [p for p in target.rglob('*') if p.is_file()]
            entries.append(dict(path=str(target), files=len(files), bytes=sum(p.stat().st_size for p in files)))
        if not (ROOT / 'provenance/previous_task_statuses.json').exists():
            raise RuntimeError('Old provenance was not preserved')
        record = dict(started_at=time.time(), targets=entries, status='deleting',
                      retained=['author checkpoint', 'Bao initializations', 'splits', 'code', 'logs', 'task status audit', 'original PostgreSQL copies'])
        fleet.write(ROOT / 'provenance/cleanup.json', record)
        for target in targets:
            shutil.rmtree(target)
            target.mkdir()
        record.update(status='complete', finished_at=time.time())
        fleet.write(ROOT / 'provenance/cleanup.json', record)
        print(json.dumps(record))


if __name__ == '__main__':
    main()
