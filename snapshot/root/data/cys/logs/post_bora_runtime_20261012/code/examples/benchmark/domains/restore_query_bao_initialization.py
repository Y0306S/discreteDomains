#!/usr/bin/env python3
"""Restore both active CEB Bao CSVs from the verified BayesQO release."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from itertools import product
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from audit_query_vae_recovery import BAO_SHA, BAYESQO, RELEASE
from query_transfer_common import QP_ROOT, VAE_FILE, HISTORICAL_VAE_FILE, atomic_json, digest, read_split
from run_query_ceb_baselines import load_initializations


PREVIOUS_SHA = 'f8a2026a7f71d1a4c642c6bcf1265858656b47ea13afb81a5d39d2a5a069b011'


def read_rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {(r['query_name'], r['plan_id']): r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate query/plan IDs: ' + str(path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New backup/report directory')
    args = parser.parse_args()
    original = BAYESQO / '.git/lfs/objects' / BAO_SHA[:2] / BAO_SHA[2:4] / BAO_SHA
    pointer = subprocess.check_output([
        'git', '-C', str(BAYESQO), 'show', RELEASE +
        ':bayes_lqo/initialization_data/ceb_3k_bao_initialization.csv',
    ], text=True)
    if 'oid sha256:' + BAO_SHA not in pointer.splitlines() or digest(original) != BAO_SHA:
        raise ValueError('Original CSV does not match the pinned release')
    expected = read_rows(original)
    queries = defaultdict(list)
    for row in expected.values():
        plan = [int(value) for value in row['encoded_plan'].split(',')]
        runtime = float(row['runtime_secs'])
        if (not plan or len(plan) % 5 or row['timed_out'] not in {'0', '1'}
                or not math.isfinite(runtime) or runtime <= 0):
            raise ValueError('Invalid released observation: ' + row['plan_id'])
        queries[row['query_name']].append(row)
    join = {row['join_hint'] for row in expected.values()}
    scan = {row['scan_hint'] for row in expected.values()}
    if len(join) != 7 or len(scan) != 7:
        raise ValueError('Expected seven join settings and seven scan settings')
    for task, rows in queries.items():
        pairs = Counter((row['join_hint'], row['scan_hint']) for row in rows)
        if pairs != Counter(product(join, scan)):
            raise ValueError('Incomplete 49-setting initialization: ' + task)
    train, test = read_split()
    load_initializations(original, train + test)

    targets = {
        'BOLT': QP_ROOT / 'initialization_data/ceb_3k_bao_initialization.csv',
        'BayesQO': BAYESQO / 'bayes_lqo/initialization_data/ceb_3k_bao_initialization.csv',
    }
    changes = {}
    for name, path in targets.items():
        current_sha = digest(path)
        if current_sha not in {PREVIOUS_SHA, BAO_SHA}:
            raise ValueError('Unreviewed change to active Bao input: ' + str(path))
        current = read_rows(path)
        changes[name] = dict(
            path=str(path), before_sha256=current_sha,
            changed_records=[dict(before=current[key], original=expected[key])
                             for key in sorted(current.keys() & expected.keys())
                             if current[key] != expected[key]],
            missing_records=[expected[key] for key in sorted(expected.keys() - current.keys())],
            extra_records=[current[key] for key in sorted(current.keys() - expected.keys())])

    protected = [VAE_FILE, HISTORICAL_VAE_FILE, BAYESQO / 'bayes_lqo/vae/CEB_64.ckpt']
    protected_hashes = {str(path): digest(path) for path in protected}
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Back up every active copy before replacing either one.
    for name, path in targets.items():
        backup = args.output / (name + '.ceb_3k_bao_initialization.before.csv')
        shutil.copy2(path, backup)
        if digest(backup) != changes[name]['before_sha256']:
            raise ValueError('Active input changed during backup: ' + str(path))
        changes[name]['backup'] = str(backup)

    report = dict(
        status='prepared', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        release=RELEASE, original_path=str(original), original_sha256=BAO_SHA,
        restored_queries=len(queries), restored_rows=len(expected), rows_per_query=49,
        train_queries=len(train), test_queries=len(test),
        train_rows=49 * len(train), test_rows=49 * len(test),
        changes=changes, checkpoints_before=protected_hashes,
        historical_results_modified=False, training_started=False, database_calls=0,
        script_sha256=digest(Path(__file__)))
    report_path = args.output / 'restore_report.json'
    atomic_json(report_path, report)
    for name, path in targets.items():
        fd, tmp_name = tempfile.mkstemp(prefix='.bao_restore_', suffix='.tmp', dir=path.parent)
        os.close(fd)
        temporary = Path(tmp_name)
        try:
            shutil.copyfile(original, temporary)
            os.chmod(temporary, path.stat().st_mode & 0o777)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            if digest(temporary) != BAO_SHA or digest(path) != changes[name]['before_sha256']:
                raise ValueError('Input changed during restore: ' + str(path))
            os.replace(temporary, path)
            parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        finally:
            temporary.unlink(missing_ok=True)
        changes[name]['after_sha256'] = digest(path)
        if changes[name]['after_sha256'] != BAO_SHA:
            raise ValueError('Restored input checksum mismatch')
        load_initializations(path, train + test)
        changes[name]['split_loader_check'] = 'passed'
        atomic_json(report_path, report)

    report['checkpoints_after'] = {str(path): digest(path) for path in protected}
    if report['checkpoints_after'] != protected_hashes:
        raise ValueError('A VAE checkpoint changed during restore')
    report['status'] = 'complete'
    atomic_json(report_path, report)
    print('Restored both active Bao files; report: ' + str(report_path), flush=True)


if __name__ == '__main__':
    main()
