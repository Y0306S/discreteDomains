"""Remove audited local-VAE train800 run files after a verified source archive."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

from query_transfer_common import HISTORY_ROOT, TRAIN_FILE, digest, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    audit_path = Path('/root/blockdata/llmoptimizer_benchmark/benchmark_data/'
                      'query_train800_author_v1/source_audit.json')
    audit = json.loads(audit_path.read_text())
    train = set(TRAIN_FILE.read_text().split())
    if set(audit['training_tasks']) != train or len(train) != 800:
        raise ValueError('Audit is not the exact train800 split')
    old_sha = '70ffaa02f88be77f73c808396dfc254a8ddda76f0945c5327b84c1f7bdc79669'
    if audit['historical_checkpoint_sha256'] != old_sha:
        raise ValueError('Audit does not identify the retired local checkpoint')
    files = {}
    for row in audit['tasks']:
        path = HISTORY_ROOT / f'optimimze-db_nowandb_{row["task_id"]}_all-data-collected.csv'
        if Path(row['source_path']) != path or digest(path) != row['source_sha256']:
            raise ValueError('Source changed since the historical audit: ' + str(path))
        marker = HISTORY_ROOT / f'{row["task_id"]}.done'
        for source in (path, marker):
            files[source.name] = dict(sha256=digest(source), bytes=source.stat().st_size,
                                      task_id=row['task_id'])
    archive_path = HISTORY_ROOT / 'retired_train800_local_vae.zip'
    index_path = HISTORY_ROOT / 'retired_train800_local_vae.json'
    if archive_path.exists() or index_path.exists():
        raise ValueError('Retirement archive already exists; inspect it before continuing')
    index = dict(status='planned', files=files, tasks=800, old_checkpoint_sha256=old_sha,
                 archive=archive_path.name, source_audit=str(audit_path),
                 source_audit_sha256=digest(audit_path),
                 bytes_before=sum(v['bytes'] for v in files.values()),
                 reason='Superseded by fresh train800 BO using the author VAE; provenance retained')
    print(json.dumps({k: v for k, v in index.items() if k != 'files'}), flush=True)
    if not args.execute:
        return
    temporary = archive_path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in files:
            archive.write(HISTORY_ROOT / name, name)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(files):
            raise ValueError('Archive membership mismatch')
        for name, metadata in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != metadata['sha256']:
                raise ValueError('Archive bytes do not match audited source: ' + name)
    temporary.replace(archive_path)
    index.update(status='archive_verified', archive_sha256=digest(archive_path),
                 archive_bytes=archive_path.stat().st_size)
    atomic_json(index_path, index)
    removed = 0
    for name, metadata in files.items():
        path = HISTORY_ROOT / name
        if digest(path) != metadata['sha256']:
            raise ValueError('Source changed immediately before deletion: ' + name)
        path.unlink()
        removed += 1
    index.update(status='retired', removed_files=removed,
                 retired_at=datetime.now(timezone.utc).isoformat())
    atomic_json(index_path, index)
    print(json.dumps({k: v for k, v in index.items() if k != 'files'}))


if __name__ == '__main__':
    main()
