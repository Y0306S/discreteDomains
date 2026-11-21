"""Pause QueryOpt only between durably recorded database calls."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parent
DEPLOY = Path('/root/data/cys/logs/query_hosts24_20261026')
QUERY = Path('/root/data/cys/logs/query_test200_baselines_20261012')
sys.path.insert(0, str(DEPLOY))
import manage as manager

write = manager.write


def prepare():
    manifest = manager.load(DEPLOY / 'manifest.json')
    before = ROOT / 'query_manifest.before.json'
    if not before.exists():
        write(before, manifest)
    source = DEPLOY / 'manage.py'
    manifest['sources'][str(source)] = manager.fleet.digest(source)
    manifest['operational_revision'] = dict(report=str(ROOT), scope='Explicit Random Search launch hold')
    write(DEPLOY / 'launch_hold.json', dict(time=time.time(), reason='User reassigned QueryOpt Random Search resources to peptide BO', report=str(ROOT)))
    write(DEPLOY / 'manifest.json', manifest)
    manager.verify()
    print(json.dumps(dict(held=True, report=str(ROOT))), flush=True)


def identity(pid):
    return manager.d.process(pid)


def send(record, signum):
    if record is not None and identity(record['pid']) == record:
        os.kill(record['pid'], signum)
        return True
    return False


def state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[0]
    except FileNotFoundError:
        return None


def gone(record):
    return identity(record['pid']) != record or state(record['pid']) in (None, 'Z', 'X')


def journal_boundary(directory):
    path = directory / 'artifacts.sqlite3'
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=0.05) as connection:
        pending = [r[0] for r in connection.execute("SELECT name FROM artifacts WHERE name GLOB 'proposal_*.pending.json'")]
        pending += [p.name for p in directory.glob('proposal_*.pending.json')]
        if pending:
            return None
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('Corrupt QueryOpt journal')
        evidence = dict(connection.execute("SELECT name,sha256 FROM artifacts WHERE name GLOB 'proposal_[0-9][0-9][0-9][0-9].json' ORDER BY name"))
    checkpoint = directory / 'checkpoint.pt'
    if not checkpoint.is_file():
        return None
    return dict(completed_sha256=evidence, checkpoint_sha256=manager.fleet.digest(checkpoint), pending=[])


def database_clients(worker):
    if manager.fleet.postgres_pid(worker) is None:
        return []
    with manager.connect(worker) as connection:
        actual = connection.execute('SHOW data_directory').fetchone()[0]
        if actual != worker['pgdata']:
            raise RuntimeError('Wrong PostgreSQL data directory')
        return connection.execute("SELECT pid,state FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()").fetchall()


def pause():
    if not (DEPLOY / 'launch_hold.json').exists():
        raise RuntimeError('Install the launch hold first')
    host = manager.HOST
    if host not in manager.ALLOWED:
        raise RuntimeError('Reserved host')
    workers = [w for w in manager.verify()['workers'] if w['host'] == host]
    record_path = ROOT / (host + '.paused.json')
    saved = manager.load(record_path, dict(host=host, workers=[]))
    finished = {r['name'] for r in saved['workers']}
    active = []
    try:
        for worker in workers:
            name = worker['name']
            if name in finished:
                continue
            process = manager.load(DEPLOY / 'processes' / (name + '.json'))
            previous = manager.load(QUERY / 'workers' / (name + '.json'), {})
            entry = dict(name=name, allocation=worker, manager=process, previous_state=previous, requested_at=time.time())
            if process and identity(process['pid']) == process:
                args = manager.args_for(process['pid'])
                if str(DEPLOY / 'manage.py') not in args or 'worker' not in args or name not in args:
                    raise RuntimeError('Manager identity mismatch')
                send(process, signal.SIGSTOP)
                children = Path(f'/proc/{process["pid"]}/task/{process["pid"]}/children').read_text().split()
                candidates = []
                for pid in children:
                    args = manager.args_for(int(pid))
                    if str(QUERY / 'code/run_query_ceb_baselines_v3.py') in args:
                        candidates.append(identity(int(pid)))
                if len(candidates) != 1:
                    send(process, signal.SIGCONT)
                    raise RuntimeError('Expected one live QueryOpt child: ' + name)
                entry['child'] = candidates[0]
                entry['task'] = previous['task']
                active.append(entry)
            else:
                if database_clients(worker):
                    raise RuntimeError('Unowned database clients: ' + name)
                manager.h.stop_database(worker)
                entry.update(status='already_stopped', stopped_at=time.time())
                saved['workers'].append(entry)
                write(record_path, saved)
        deadline = time.monotonic() + 300
        while active:
            if time.monotonic() >= deadline:
                raise TimeoutError('No durable query boundary within five minutes')
            for entry in list(active):
                child = entry['child']
                directory = QUERY / 'outputs' / entry['task'] / 'random/seed0'
                if gone(child):
                    boundary = journal_boundary(directory)
                else:
                    send(child, signal.SIGSTOP)
                    until = time.monotonic() + 1
                    while state(child['pid']) not in ('T', 't', 'Z', None) and time.monotonic() < until:
                        time.sleep(0.01)
                    if state(child['pid']) not in ('T', 't', 'Z', None):
                        send(child, signal.SIGCONT)
                        continue
                    try:
                        boundary = journal_boundary(directory)
                        if boundary is not None and any(s == 'active' for _, s in database_clients(entry['allocation'])):
                            boundary = None
                    except sqlite3.OperationalError:
                        boundary = None
                    if boundary is None:
                        send(child, signal.SIGCONT)
                        continue
                if boundary is None:
                    raise RuntimeError('Query child exited with an unresolved database call')
                entry['boundary'] = boundary
                write(ROOT / (entry['name'] + '.boundary.json'), entry)
                send(child, signal.SIGTERM)
                send(child, signal.SIGCONT)
                send(entry['manager'], signal.SIGTERM)
                send(entry['manager'], signal.SIGCONT)
                until = time.monotonic() + 30
                while (not gone(child) or not gone(entry['manager'])) and time.monotonic() < until:
                    time.sleep(0.2)
                if not gone(child) or not gone(entry['manager']):
                    raise RuntimeError('Query process did not exit')
                if database_clients(entry['allocation']):
                    raise RuntimeError('Database still has a client after worker exit')
                with manager.fleet.lock(QUERY / 'workers' / (entry['name'] + '.lock')):
                    after = journal_boundary(directory)
                    if after != boundary:
                        raise RuntimeError('Query evidence changed after stop')
                    manager.h.stop_database(entry['allocation'])
                paused = dict(entry['previous_state'], status='paused', paused_at=manager.fleet.stamp(),
                    saved_proposals=len(boundary['completed_sha256']), child_pid=None,
                    reason='User resource reassignment to peptide BO', pause_report=str(record_path))
                write(QUERY / 'workers' / (entry['name'] + '.json'), paused)
                write(DEPLOY / 'state' / (entry['name'] + '.json'), dict(status='paused', time=manager.fleet.stamp(), report=str(record_path)))
                claim_path = QUERY / 'claims' / ('random.' + entry['task'] + '.json')
                with manager.fleet.lock(QUERY / 'claims/.queue.lock', blocking=True):
                    claim = manager.load(claim_path)
                    if claim['status'] != 'complete':
                        write(claim_path, dict(claim, status='paused', updated_at=manager.fleet.stamp(), pause_report=str(record_path)))
                entry.update(status='paused', stopped_at=time.time(), database_stopped=True)
                saved['workers'].append(entry)
                active.remove(entry)
                write(record_path, saved)
                print(json.dumps(dict(paused=entry['name'], task=entry['task'], proposals=len(boundary['completed_sha256']))), flush=True)
            if active:
                time.sleep(0.25)
        print(json.dumps(dict(host=host, stopped=len(saved['workers']), report=str(record_path))), flush=True)
    finally:
        for entry in active:
            send(entry.get('child'), signal.SIGCONT)
            send(entry.get('manager'), signal.SIGCONT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'pause'])
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else pause)()
