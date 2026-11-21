"""Cold-copy QueryOpt databases off Lustre and verify their copied contents."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import psycopg

OLD = Path('/root/data/cys/logs/query_train800_author_bo_20261002')
REPORT = OLD / 'restart_20261004'
PG = Path('/root/data/cys/envs/pg16/bin')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(obj, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def target(worker):
    base = Path('/root/blockdata/queryopt_pg_20261004') if worker['host'] == 'LLMOptimizer2' else Path('/dev/shm/queryopt_pg_20261004')
    return base / Path(worker['pgdata']).name


def cold_copy(worker):
    source, destination = Path(worker['pgdata']), target(worker)
    record = REPORT / (worker['name'] + '.storage.json')
    if record.exists() and json.loads(record.read_text()).get('status') == 'verified_copy':
        if not destination.is_dir():
            raise RuntimeError('Verified copy is missing; explicit restoration required')
        print(json.dumps({'worker': worker['name'], 'status': 'already_copied'}), flush=True)
        return
    if destination.exists():
        raise RuntimeError(f'Unverified destination exists: {destination}')
    cpus = ','.join(map(str, worker['db_cpus']))
    os.sched_setaffinity(0, set(worker['db_cpus']))
    with psycopg.connect(host='127.0.0.1', port=worker['port'], dbname='imdb', user='imdb', autocommit=True) as conn:
        clients = conn.execute("SELECT pid,usename,state FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()").fetchall()
        if clients:
            raise RuntimeError(f'Other database clients: {clients}')
    pid = int((source / 'postmaster.pid').read_text().splitlines()[0])
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if str(source).encode() not in argv or b'postgres' not in argv[0]:
        raise RuntimeError('Database PID identity mismatch')
    started = time.time()
    subprocess.run(['taskset', '-c', cpus, 'setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
                    str(PG / 'pg_ctl'), '-D', str(source), '-m', 'fast', '-w', '-t', '90', 'stop'], cwd='/tmp', check=True)
    control = subprocess.check_output([str(PG / 'pg_controldata'), str(source)], text=True)
    if 'shut down' not in next(line for line in control.splitlines() if 'Database cluster state:' in line):
        raise RuntimeError('Database was not cleanly shut down')
    if any(source.joinpath('pg_tblspc').iterdir()) or (source / 'pg_wal').is_symlink():
        raise RuntimeError('External database storage needs explicit handling')
    size = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < size + 5 * 1024**3:
        raise RuntimeError('Insufficient local disk headroom')
    if str(destination).startswith('/dev/shm/'):
        cg = Path('/sys/fs/cgroup')
        headroom = int((cg / 'memory.max').read_text()) - int((cg / 'memory.current').read_text())
        if headroom < size + 16 * 1024**3:
            raise RuntimeError('Insufficient container RAM headroom for local database')
    state = dict(worker=worker, source=str(source), destination=str(destination), bytes=size,
                 started_at=started, volatile=str(destination).startswith('/dev/shm/'),
                 status='copying', source_preserved=True, clean_shutdown=True)
    write(record, state)
    subprocess.run(['cp', '-a', '--reflink=auto', str(source), str(destination)], check=True)
    files = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            if os.readlink(path) != os.readlink(destination / path.relative_to(source)):
                raise RuntimeError('Symlink mismatch')
        elif path.is_file():
            relative = path.relative_to(source)
            expected = fingerprint(path)
            if fingerprint(destination / relative) != expected:
                raise RuntimeError(f'Copied file differs: {relative}')
            files[str(relative)] = expected
    write(REPORT / (worker['name'] + '.files.json'), files)
    state.update(status='verified_copy', verified_files=len(files), finished_at=time.time())
    write(record, state)
    print(json.dumps(state), flush=True)


def verify_sql(worker):
    os.sched_setaffinity(0, set(worker['db_cpus']))
    started = time.time()
    with psycopg.connect(host='127.0.0.1', port=worker['port'], dbname='imdb', user='imdb', autocommit=True) as conn:
        directory = conn.execute('SHOW data_directory').fetchone()[0]
        if directory != str(target(worker)):
            raise RuntimeError('Database is still on the old storage')
        conn.execute("SET statement_timeout='10min'")
        conn.execute('SET enable_indexscan=off')
        conn.execute('SET enable_indexonlyscan=off')
        conn.execute('SET enable_bitmapscan=off')
        counts = {}
        for name in ('name', 'movie_info', 'title'):
            counts[name] = conn.execute(psycopg.sql.SQL('SELECT count(*) FROM {}').format(psycopg.sql.Identifier(name))).fetchone()[0]
        # amcheck verifies the index implicated in two of the failed workers.
        conn.execute('CREATE EXTENSION IF NOT EXISTS amcheck')
        conn.execute("SELECT bt_index_check('title_pkey'::regclass, true)")
        version = conn.execute('SELECT version()').fetchone()[0]
    result = dict(worker=worker['name'], status='verified', directory=directory, row_counts=counts,
                  title_pkey_heapallindexed=True, postgres=version, elapsed_seconds=time.time()-started)
    write(REPORT / (worker['name'] + '.sql_verified.json'), result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('copy', 'verify'))
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    plan = json.loads((OLD / 'plan.json').read_text())
    if socket.gethostname() != plan['hosts'][args.host]:
        raise RuntimeError('Wrong host')
    for worker in plan['workers']:
        if worker['host'] == args.host:
            (cold_copy if args.action == 'copy' else verify_sql)(worker)


if __name__ == '__main__':
    main()
