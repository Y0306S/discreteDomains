"""Preserve committed peptide batches before replacing target-fit code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/peptide_transfer_test200_20261028_v1')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        words = (root / 'cmdline').read_bytes().split(b'\0')
        return stat[0], stat[19], words
    except FileNotFoundError:
        return None


def stop(side):
    path = CAMPAIGN / 'workers' / ('queue_' + side + '.json')
    queue = json.loads(path.read_text())
    pid = queue['pid']
    current = process(pid)
    if current and any(b'queue_peptide_transfer.py' in w for w in current[2]):
        os.kill(pid, signal.SIGTERM)
        for _ in range(250):
            current = process(pid)
            if current is None or current[0] == 'Z':
                break
            time.sleep(0.2)
        else:
            raise RuntimeError('Supervisor did not stop')
    queue = json.loads(path.read_text())
    write(ROOT / ('queue_' + side + '_before.json'), queue)
    retained, stopped = [], []
    resume = queue.get('reroute_resume', {})
    for worker in queue['active']:
        if worker['method'] != 'fsbo':
            retained.append(worker)
            continue
        pid = worker['pid']
        current = process(pid)
        if not current or current[0] == 'Z':
            resume[str(worker['cpu'])] = dict(method='fsbo', task=worker['task'])
            continue
        if current[1] != worker['process_start'] or os.getpgid(pid) != pid:
            raise RuntimeError('Worker process identity/group changed')
        os.killpg(pid, signal.SIGSTOP)
        terminated = False
        try:
            # A frozen writer may hold a transaction. In that case resume and retry
            # maintenance later rather than interrupting an uncommitted batch.
            db_path = CAMPAIGN / 'fsbo/runs' / worker['task'] / 'runtime.sqlite'
            with sqlite3.connect('file:' + str(db_path) + '?mode=ro', uri=True, timeout=2) as db:
                pending = db.execute("SELECT value FROM meta WHERE key='pending'").fetchone()
                if pending and pickle.loads(pending[0]) is not None:
                    raise RuntimeError('Worker has a pending evaluation batch; retry at next checkpoint')
                values = dict(db.execute('SELECT key,value FROM meta'))
                destination = ROOT / (side + '_' + worker['task'] + '.sqlite')
                with sqlite3.connect(str(destination)) as backup:
                    # Copy committed records under the same read transaction.
                    backup.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)')
                    backup.executemany('INSERT OR REPLACE INTO meta VALUES (?,?)', values.items())
                    if backup.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                        raise RuntimeError('Backup integrity check failed')
                evidence = dict(worker=worker, backup=str(destination), pending=None,
                    state_sha256=hashlib.sha256(values['state']).hexdigest(), stopped_at=time.time())
                write(ROOT / (side + '_' + worker['task'] + '.json'), evidence)
            os.killpg(pid, signal.SIGKILL)
            terminated = True
            resume[str(worker['cpu'])] = dict(method='fsbo', task=worker['task'])
            stopped.append(evidence)
        finally:
            if not terminated:
                os.killpg(pid, signal.SIGCONT)
    queue.update(active=retained, reroute_resume=resume, status='maintenance', updated=time.time())
    write(path, queue)
    write(ROOT / (side + '_stopped.json'), dict(stopped=stopped, at=time.time()))
    print(json.dumps(dict(side=side, stopped=len(stopped), retained=len(retained))))


def resume(side):
    path = CAMPAIGN / 'workers' / ('queue_' + side + '.json')
    queue = json.loads(path.read_text())
    before = json.loads((ROOT / ('queue_' + side + '_before.json')).read_text())
    for worker in before['active']:
        if worker['method'] != 'fsbo':
            continue
        path = ROOT / (side + '_' + worker['task'] + '.json')
        if not path.exists():
            raise RuntimeError('Missing maintenance evidence')
        evidence = json.loads(path.read_text())
        db_path = CAMPAIGN / 'fsbo/runs' / worker['task'] / 'runtime.sqlite'
        with sqlite3.connect('file:' + str(db_path) + '?mode=ro', uri=True) as db:
            value = db.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0]
        if hashlib.sha256(value).hexdigest() != evidence['state_sha256']:
            raise RuntimeError('Saved evaluation state changed during maintenance')
    launch = json.loads((CAMPAIGN / 'workers' / ('queue_' + side + '_launch.json')).read_text())
    command = launch['command'] + ['--launch']
    subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'), check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stop', 'resume'))
    parser.add_argument('side', choices=('local', 'remote'))
    args = parser.parse_args()
    (stop if args.action == 'stop' else resume)(args.side)
