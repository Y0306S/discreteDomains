"""Provision a private, verified Query PostgreSQL clone on LLMOptimizer4.

This script only prepares database instances. It never launches optimizers or
changes campaign resource allocations.
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[2]
PG = Path('/root/data/cys/envs/pg16/bin')
SOURCE = Path('/dev/shm/query_fsbo_pilot_20261106_v1/w18')
SOURCE_PORT = 19718
OLD_TABLESPACES = {
    '/root/data/logs/query_fsbo_test200_50s_20261106_v1/tablespaces/w18': 'legacy',
    '/root/data/cys/baseline/campaigns/query_our_fsbo_test200_20261116/scratch/LLMOptimizer4/worker20': 'scratch',
}


def paths(index):
    if not 22 <= index <= 28:
        raise ValueError('Only planned worker indices 22 through 28 are allowed')
    base = ROOT / 'databases/LLMOptimizer4'
    return {
        'index': index,
        'port': 19720 + index - 22,
        'pgdata': base / f'w{index}',
        'legacy': base / f'w{index}_legacy_tablespace',
        'scratch': ROOT / 'scratch/LLMOptimizer4' / f'worker{index}',
        'socket': Path('/tmp/queryfsbo_pg') / f'w{index}',
        'postgres_cpus': list(range(68 + 4 * (index - 22), 71 + 4 * (index - 22))),
        'optimizer_cpu': 71 + 4 * (index - 22),
    }


def run(command, log=None, timeout=None):
    if log:
        with log.open('ab') as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    else:
        subprocess.run(command, check=True, timeout=timeout)


def owned_empty(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f'Unsafe nonempty target: {path}')
    os.chown(path, 10000, 10000)
    path.chmod(0o700)


def preflight(p):
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert int(quota) / int(period) >= 64
    allowed = os.sched_getaffinity(0)
    assert set(p['postgres_cpus'] + [p['optimizer_cpu']]) <= allowed
    assert not (p['pgdata'] / 'PG_VERSION').exists()
    with socket.create_connection(('127.0.0.1', SOURCE_PORT), timeout=3):
        pass
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', p['port'])) != 0, 'Target port occupied'
    with psycopg.connect(host='127.0.0.1', port=SOURCE_PORT, dbname='imdb', user='imdb', connect_timeout=5) as db:
        assert db.execute('SHOW data_directory').fetchone()[0] == str(SOURCE)
        assert db.execute('SHOW wal_level').fetchone()[0] == 'replica'
    stat = os.statvfs(str(ROOT))
    assert stat.f_bavail * stat.f_frsize > 300 * 1024**3, 'Insufficient shared-disk reserve'
    used = int(Path('/sys/fs/cgroup/memory.current').read_text())
    maximum = int(Path('/sys/fs/cgroup/memory.max').read_text())
    assert maximum - used > 40 * 1024**3, 'Insufficient memory reserve'


def start_verified(p):
    assert (p['pgdata'] / 'backup_manifest').exists()
    assert 'backup successfully verified' in (ROOT / 'operations/capacity' / f"w{p['index']}.verify.log").read_text()
    with socket.socket() as sock:
        running = sock.connect_ex(('127.0.0.1', p['port'])) == 0
    if not running:
        owned_empty(p['socket'])
        source_options = shlex.split((SOURCE / 'postmaster.opts').read_text())
        assert source_options[0] == str(PG / 'postgres')
        options = source_options[1:]
        for flag, value in [('-D', str(p['pgdata'])), ('-p', str(p['port'])), ('-k', str(p['socket']))]:
            options[options.index(flag) + 1] = value
        command = ['setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
                   'taskset', '-c', ','.join(map(str, p['postgres_cpus'])),
                   str(PG / 'pg_ctl'), '-D', str(p['pgdata']),
                   '-l', str(p['pgdata'] / 'semanticopt.log'), '-o', shlex.join(options),
                   '-w', '-t', '180', 'start']
        print(f"START w{p['index']}", flush=True)
        run(command, timeout=200)
    with psycopg.connect(host='127.0.0.1', port=p['port'], dbname='imdb', user='imdb', connect_timeout=10, autocommit=True) as db:
        assert db.execute('SHOW data_directory').fetchone()[0] == str(p['pgdata'])
        db.execute("LOAD 'pg_hint_plan'")
        assert db.execute('SHOW pg_hint_plan.enable_hint').fetchone()[0] == 'on'
        assert db.execute("SELECT reltuples::bigint FROM pg_class WHERE oid='title'::regclass").fetchone()[0] > 1000000
        assert db.execute('SELECT count(*) FROM movie_info').fetchone()[0] == 14835720
        assert db.execute("SELECT count(*) FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND state='active'").fetchone()[0] == 0
        assert db.execute('SHOW temp_tablespaces').fetchone()[0] == 'query_hybrid_temp'
    postmaster = int((p['pgdata'] / 'postmaster.pid').read_text().splitlines()[0])
    assert set(os.sched_getaffinity(postmaster)) == set(p['postgres_cpus'])
    result = {'ready': True, 'index': p['index'], 'port': p['port'],
              'pgdata': str(p['pgdata']), 'scratch': str(p['scratch']),
              'postgres_cpus': p['postgres_cpus'], 'optimizer_cpu': p['optimizer_cpu'],
              'postmaster_pid': postmaster, 'at': time.time()}
    (ROOT / 'operations/capacity' / f"w{p['index']}.ready.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def provision(index):
    p = paths(index)
    preflight(p)
    for name in ['pgdata', 'legacy', 'scratch']:
        owned_empty(p[name])
    log = ROOT / 'operations/capacity' / f'w{index}.backup.log'
    command = [
        'setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
        'taskset', '-c', ','.join(map(str, p['postgres_cpus'])),
        str(PG / 'pg_basebackup'), '-h', '127.0.0.1', '-p', str(SOURCE_PORT),
        '-U', 'imdb', '-w', '-D', str(p['pgdata']), '-F', 'p', '-X', 'stream',
        '-c', 'spread', '-r', '64M', '--manifest-checksums=SHA256', '-P', '-v',
    ]
    for old, name in OLD_TABLESPACES.items():
        command.extend(['-T', f'{old}={p[name]}'])
    print(f'BACKUP w{index} started {time.strftime("%Y-%m-%dT%H:%M:%S%z")}', flush=True)
    run(command, log=log, timeout=1800)
    assert (p['pgdata'] / 'backup_manifest').exists()
    print(f'BACKUP w{index} complete; verifying SHA256 manifest', flush=True)
    run(['setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
         str(PG / 'pg_verifybackup'), '-n', str(p['pgdata'])],
        log=ROOT / 'operations/capacity' / f'w{index}.verify.log', timeout=1800)
    start_verified(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)
    parser.add_argument('--start-existing', action='store_true')
    args = parser.parse_args()
    if args.start_existing:
        start_verified(paths(args.index))
    else:
        provision(args.index)
