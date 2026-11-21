#!/usr/bin/env python3
"""Stop the requested host with durable QueryOpt results and a cold DB archive."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time


ROOT = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
OUT = ROOT / 'maintenance/LLMOptimizer3_gpu_change_20261005'
NAME = 'LLMOptimizer3.w00'
REASON = 'User requested stopping LLMOptimizer3 for conversion to four A800 GPUs.'
PGDATA = Path('/dev/shm/queryopt_pg_20261004/pg_imdb_d')
PGBIN = Path('/root/data/cys/envs/pg16/bin')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0],
                'argv': (path / 'cmdline').read_bytes().split(b'\0')}
    except FileNotFoundError:
        return None


def signal_checked(original, signum):
    current = identity(original['pid'])
    if current is not None and current['start_ticks'] == original['start_ticks'] and current['state'] != 'Z':
        os.kill(original['pid'], signum)


def active(original):
    current = identity(original['pid'])
    return current is not None and current['start_ticks'] == original['start_ticks'] and current['state'] != 'Z'


def freeze(original):
    signal_checked(original, signal.SIGSTOP)
    deadline = time.monotonic() + 10
    while active(original):
        states = []
        for thread in (Path('/proc') / str(original['pid']) / 'task').iterdir():
            try:
                fields = (thread / 'stat').read_text().rsplit(')', 1)[1].split()
                states.append(fields[0])
            except FileNotFoundError:
                pass
        if states and all(state in ('T', 't') for state in states):
            return
        if time.monotonic() > deadline:
            raise RuntimeError('Process did not stop at the inspection boundary')
        time.sleep(.05)


def pause_worker():
    state_path = ROOT / 'workers' / (NAME + '.json')
    state = read(state_path)
    if state.get('status') == 'paused' and (OUT / 'worker_paused.json').exists():
        print('Worker already paused.', flush=True)
        return
    supervisor, child = identity(state['pid']), identity(state['child_pid'])
    if supervisor is None or child is None:
        raise RuntimeError('Expected active worker processes were not found')
    assert str(ROOT / 'code/query_train800_fleet.py').encode() in supervisor['argv']
    assert NAME.encode() in supervisor['argv']
    assert str(ROOT / 'code/run_query_ceb_baselines.py').encode() in child['argv']
    assert state['task'].encode() in child['argv']
    directory = ROOT / 'outputs' / state['task'] / 'lolbo/seed0'
    assert (directory / 'checkpoint.pt').exists()
    write(OUT / 'worker_before.json', state)
    write(OUT / 'claim_before.json', read(ROOT / 'claims' / (state['task'] + '.json')))
    freeze(supervisor)
    try:
        deadline = time.monotonic() + 180
        while True:
            freeze(child)
            unresolved = [path for path in directory.glob('proposal_*.pending.json')
                          if not path.with_name(path.name.replace('.pending.json', '.json')).exists()]
            if not unresolved:
                break
            signal_checked(child, signal.SIGCONT)
            if time.monotonic() > deadline:
                raise RuntimeError('An SQL evaluation is still in flight; nothing was terminated')
            time.sleep(.25)
        for item in ('checkpoint.pt', 'history.json', 'status.json'):
            if (directory / item).exists():
                shutil.copy2(directory / item, OUT / ('before_stop_' + item))
        signal_checked(child, signal.SIGTERM)
        signal_checked(child, signal.SIGCONT)
        signal_checked(supervisor, signal.SIGTERM)
        signal_checked(supervisor, signal.SIGCONT)
        deadline = time.monotonic() + 30
        while active(supervisor) or active(child):
            if time.monotonic() > deadline:
                raise RuntimeError('Processes did not exit after SIGTERM')
            time.sleep(.1)
    except BaseException:
        signal_checked(child, signal.SIGCONT)
        signal_checked(supervisor, signal.SIGCONT)
        raise
    results = sorted(directory.glob('proposal_[0-9][0-9][0-9][0-9].json'))
    assert [int(path.name[9:13]) for path in results] == list(range(len(results)))
    assert not any(not path.with_name(path.name.replace('.pending.json', '.json')).exists()
                   for path in directory.glob('proposal_*.pending.json'))
    artifacts = {str(path.relative_to(ROOT)): digest(path) for path in
                 [directory / 'checkpoint.pt', *results]}
    record = {'status': 'paused', 'reason': REASON, 'paused_at': now(), 'worker': NAME,
              'task': state['task'], 'saved_proposals': len(results),
              'unresolved_database_calls': 0, 'artifacts_sha256': artifacts,
              'checkpoint': str(directory / 'checkpoint.pt'),
              'supervisor_pid': supervisor['pid'], 'optimizer_pid': child['pid']}
    state.update(status='paused', updated_at=now(), paused_at=record['paused_at'],
                 pause_reason=REASON, error=None, saved_proposals=len(results),
                 maintenance_record=str(OUT / 'worker_paused.json'))
    write(state_path, state)
    claim_path = ROOT / 'claims' / (state['task'] + '.json')
    claim = read(claim_path)
    claim.update(status='paused', updated_at=now(), pause_reason=REASON)
    write(claim_path, claim)
    write(OUT / 'worker_paused.json', record)
    print(json.dumps({key: value for key, value in record.items() if key != 'artifacts_sha256'}), flush=True)


def stop_scan():
    scan = identity(41275)
    if scan is not None and scan['state'] != 'Z':
        assert scan['argv'][0] == b'du' and scan['argv'][1] == b'-xsh'
        assert os.getpgid(scan['pid']) == 41274
        os.killpg(41274, signal.SIGTERM)
        print('Stopped the old disk-usage scan and its pipeline.', flush=True)


def stop_database():
    assert (OUT / 'worker_paused.json').exists()
    if (PGDATA / 'postmaster.pid').exists():
        pid = int((PGDATA / 'postmaster.pid').read_text().splitlines()[0])
        proc = identity(pid)
        assert proc is not None and str(PGDATA).encode() in proc['argv']
        subprocess.run(['setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
                        str(PGBIN / 'pg_ctl'), '-D', str(PGDATA), '-m', 'fast', '-w', '-t', '60', 'stop'],
                       check=True, cwd='/tmp')
    control = subprocess.check_output([str(PGBIN / 'pg_controldata'), str(PGDATA)], text=True,
                                      env={**os.environ, 'LC_ALL': 'C'})
    state = next(line.split(':', 1)[1].strip() for line in control.splitlines()
                 if line.startswith('Database cluster state:'))
    assert state == 'shut down', state
    path = ROOT / 'databases' / (NAME + '.json')
    db = read(path)
    write(OUT / 'database_before.json', db)
    db.update(status='stopped_for_maintenance', stopped_at=now(), stop_reason=REASON,
              cluster_state=state, maintenance_record=str(OUT))
    write(path, db)
    write(OUT / 'database_stopped.json', {'stopped_at': now(), 'cluster_state': state,
                                        'pgdata': str(PGDATA), 'control_data': control})
    print('PostgreSQL is cleanly shut down.', flush=True)


def archive_database():
    assert (OUT / 'database_stopped.json').exists() and not (PGDATA / 'postmaster.pid').exists()
    archive = OUT / 'pg_imdb_d.clean_shutdown.tar'
    if (OUT / 'database_archive.json').exists():
        raise RuntimeError('Verified database backup already exists')
    partial = archive.with_suffix('.tar.partial')
    if partial.exists() or archive.exists():
        raise RuntimeError('Archive destination already exists; inspect it before proceeding')
    subprocess.run(['tar', '--sparse', '-cpf', str(partial), '-C', str(PGDATA.parent), PGDATA.name], check=True)
    print('Archive written; comparing it against the stopped database.', flush=True)
    subprocess.run(['tar', '--compare', '-f', str(partial), '-C', str(PGDATA.parent)], check=True)
    checksum = digest(partial)
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    partial.replace(archive)
    write(OUT / 'database_archive.json', {'verified_at': now(), 'archive': str(archive),
          'bytes': archive.stat().st_size, 'sha256': checksum,
          'verification': 'GNU tar compare against the cleanly shut-down database',
          'restore_parent': str(PGDATA.parent), 'original_source_preserved': '/root/data/cys/pg_imdb_d'})
    print(json.dumps(read(OUT / 'database_archive.json')), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stop', 'archive'))
    args = parser.parse_args()
    assert socket.gethostname() == 'cs-0b2cd-18deb-server', 'Wrong host'
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'stop':
        pause_worker()
        stop_scan()
        stop_database()
    else:
        archive_database()


if __name__ == '__main__':
    main()
