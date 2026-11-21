"""Move random-search SQL spill files off tmpfs and resume reviewed failures."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parent
HANDOFF = Path('/root/data/cys/logs/query_random_optformer_handoff_20261014')
TEMP_ROOT = Path('/root/data/cys/queryopt_temp/LLMOptimizer4_random_20261015')
TABLESPACE = 'query_random_disk_temp'
FAILED = {'LLMOptimizer4.random01': ('CEB_9A136', 207),
          'LLMOptimizer4.random03': ('CEB_8A492', 111)}
sys.path.insert(0, str(HANDOFF))
import handoff as h
from run_artifact_store import JsonJournal, read_bytes


def save(name, value):
    h.write(ROOT / name, value)


def connect(worker):
    return psycopg.connect(host='127.0.0.1', port=worker['port'], dbname='imdb',
                           user='imdb', autocommit=True, connect_timeout=10)


def headroom():
    info = os.statvfs(TEMP_ROOT if TEMP_ROOT.exists() else '/root/data')
    result = {'free_gib': info.f_bavail * info.f_frsize / 2**30,
              'free_files': info.f_favail, 'time': time.time()}
    if result['free_gib'] < 256 or result['free_files'] < 300000:
        raise RuntimeError('Insufficient shared-storage headroom for SQL spill files')
    return result


def configure():
    h.verify()
    baseline = ROOT / 'before.json'
    if not baseline.exists():
        save('before.json', {'time': time.time(), 'headroom': headroom(),
                            'handover': h.load(HANDOFF / 'status.json'),
                            'controller': h.load(HANDOFF / 'controller_process.json')})
    for worker in h.allocations():
        name = worker['name']
        marker = ROOT / (name + '.storage.json')
        location = TEMP_ROOT / name
        record = h.load(marker, {'worker': name, 'port': worker['port'],
                                'pgdata': worker['pgdata'], 'tablespace': TABLESPACE,
                                'location': str(location), 'started_at': time.time(),
                                'headroom_before': headroom()})
        with connect(worker) as connection:
            actual = connection.execute('SHOW data_directory').fetchone()[0]
            if actual != worker['pgdata']:
                raise RuntimeError('Unexpected PostgreSQL instance')
            record.setdefault('previous_temp_tablespaces', connection.execute(
                'SHOW temp_tablespaces').fetchone()[0])
            record.setdefault('worker_at_change', {
                k: h.load(h.QUERY / 'workers' / (name + '.json'), {}).get(k)
                for k in ('task', 'saved_proposals', 'updated_at')})
            save(marker.name, record)
            row = connection.execute(
                'SELECT oid,pg_tablespace_location(oid) FROM pg_tablespace WHERE spcname=%s',
                (TABLESPACE,)).fetchone()
            if row is None:
                location.mkdir(parents=True, mode=0o700, exist_ok=True)
                if any(location.iterdir()):
                    raise RuntimeError('Unverified existing tablespace directory')
                os.chown(location, 10000, 10000)
                os.chmod(location, 0o700)
                connection.execute(sql.SQL('CREATE TABLESPACE {} OWNER imdb LOCATION {}').format(
                    sql.Identifier(TABLESPACE), sql.Literal(str(location))))
            elif row[1] != str(location):
                raise RuntimeError('Tablespace has an unexpected location')
            connection.execute(sql.SQL('ALTER DATABASE imdb SET temp_tablespaces TO {}').format(
                sql.Literal(TABLESPACE)))
        with connect(worker) as connection:
            effective = connection.execute('SHOW temp_tablespaces').fetchone()[0]
            if effective != TABLESPACE:
                raise RuntimeError('New SQL sessions did not inherit the storage policy')
            connection.execute("SET statement_timeout = '30s'")
            connection.execute("SET work_mem = '64kB'")
            connection.execute('SET max_parallel_workers_per_gather = 0')
            connection.execute('SET jit = off')
            connection.execute('CREATE TEMP TABLE storage_probe (value integer)')
            relation = connection.execute(
                "SELECT pg_relation_filepath('storage_probe')").fetchone()[0]
            oid = connection.execute('SELECT oid FROM pg_tablespace WHERE spcname=%s',
                                     (TABLESPACE,)).fetchone()[0]
            if not relation.startswith('pg_tblspc/' + str(oid) + '/'):
                raise RuntimeError('Temporary relation was placed outside the new tablespace')
            plan = connection.execute(
                'EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) '
                'SELECT repeat(md5(g::text),32) FROM generate_series(1,20000) AS g '
                'ORDER BY md5(g::text)').fetchone()[0][0]['Plan']
            if plan.get('Sort Space Type') != 'Disk' or plan.get('Temp Written Blocks', 0) <= 0:
                raise RuntimeError('Storage probe did not exercise actual disk spill')
            record.update(effective_temp_tablespaces=effective, temporary_relation=relation,
                          spill_probe=plan, verified_at=time.time())
        save(marker.name, record)
        print(json.dumps({'worker': name, 'storage_verified': True,
                          'spill_kib': plan.get('Sort Space Used')}), flush=True)
    save('storage_policy.json', {
        'time': time.time(), 'workers': [w['name'] for w in h.allocations()],
        'temp_tablespaces': TABLESPACE, 'root': str(TEMP_ROOT), 'headroom': headroom(),
        'scope': 'New SQL connections on the eight handover databases',
        'timing_note': 'Temporary SQL spill uses shared NVMe instead of RAM-backed tmpfs; '
                       'this infrastructure change can affect spill-heavy query runtimes.',
        'benchmark_policy': 'Unchanged 100-second SQL timeout, proposal budget, VAE, seed and plans'})


def prefix_hashes(journal, count):
    result = {}
    for index in range(count):
        name = 'proposal_%04d.json' % index
        result[name] = hashlib.sha256(read_bytes(journal.root, name)).hexdigest()
    return result


def review(worker, task, count):
    name = worker['name']
    directory = h.QUERY / 'outputs' / task / 'random/seed0'
    marker = ROOT / (name + '.recovery.json')
    with h.fleet.lock(HANDOFF / 'locks' / (name + '.lock')):
        with h.fleet.lock(h.QUERY / 'workers' / (name + '.lock')):
            with h.fleet.lock(directory / '.lock'):
                process = h.load(HANDOFF / 'processes' / (name + '.json'), {})
                if process and h.d.process(process['pid']) == process:
                    raise RuntimeError('Failed worker is unexpectedly alive')
                claim = h.load(h.QUERY / 'claims' / ('random.' + task + '.json'))
                if claim['worker'] != name or claim['status'] == 'complete':
                    raise RuntimeError('Task ownership changed')
                journal = JsonJournal(directory)
                pending_name = 'proposal_%04d.pending.json' % count
                attempts_name = 'proposal_%04d.database_attempts.json' % count
                if journal.names('proposal_*.pending.json') != [pending_name]:
                    raise RuntimeError('Unexpected pending database calls')
                if len(journal.names('proposal_[0-9][0-9][0-9][0-9].json')) != count:
                    raise RuntimeError('Completed proposal count changed')
                proposal = journal.read(pending_name)
                attempts = journal.read(attempts_name)
                last = attempts[-1]
                if (last.get('status') != 'error' or last.get('sqlstate') != '53100'
                        or not last.get('finished_at')
                        or any(a.get('proposal') != proposal for a in attempts)):
                    raise RuntimeError('Unrecognized failed SQL attempt')
                with connect(worker) as connection:
                    clients = connection.execute(
                        "SELECT count(*) FROM pg_stat_activity WHERE "
                        "backend_type='client backend' AND pid<>pg_backend_pid()"
                    ).fetchone()[0]
                    connection.execute('SELECT 1 FROM title LIMIT 1').fetchone()
                    effective = connection.execute('SHOW temp_tablespaces').fetchone()[0]
                if clients or effective != TABLESPACE:
                    raise RuntimeError('SQL activity or unverified temporary storage prevents replay')
                record = {'worker': name, 'task': task, 'saved_proposals': count,
                          'pending': pending_name, 'original_attempts': attempts,
                          'completed_sha256': prefix_hashes(journal, count),
                          'original_worker_state': h.load(HANDOFF / 'state' / (name + '.json')),
                          'original_claim': claim, 'verified_at': time.time()}
                if not marker.exists():
                    save(marker.name, record)
                recovery = {'replay_authorized': True, 'active_database_calls': 0,
                            'original_result_unavailable': True, 'evidence_report': str(marker),
                            'verified_at': time.time(), 'temp_tablespaces': TABLESPACE,
                            'reason': 'SQL returned DiskFull (53100) before a result was committed; '
                                      'disk spill storage is repaired and no query remains active'}
                attempts[-1]['storage_recovery'] = recovery
                journal.write(attempts_name, attempts)


def restart():
    h.verify()
    if not (ROOT / 'storage_policy.json').exists():
        raise RuntimeError('Configure and validate storage before resuming')
    for name, (task, count) in FAILED.items():
        review(h.worker_config(name), task, count)
    controller = h.load(HANDOFF / 'controller_process.json')
    if controller and h.d.process(controller['pid']) == controller:
        h.d.stop(controller)
    try:
        for name in FAILED:
            path = HANDOFF / 'state' / (name + '.json')
            old = h.load(path)
            h.write(path, dict(old, status='smoke_ready', error=None,
                              recovery_report=str(ROOT / (name + '.recovery.json'))))
            process = h.spawn_worker(h.worker_config(name))
            print(json.dumps({'worker': name, 'started': process,
                              'waiting_for_memory': process is None}), flush=True)
    finally:
        h.start()


def verify():
    h.verify()
    report = {'time': time.time(), 'workers': h.status(), 'recovered': []}
    for name, (task, count) in FAILED.items():
        baseline = h.load(ROOT / (name + '.recovery.json'))
        journal = JsonJournal(h.QUERY / 'outputs' / task / 'random/seed0')
        if prefix_hashes(journal, count) != baseline['completed_sha256']:
            raise RuntimeError('A previously completed evaluation changed')
        status = journal.read('status.json')
        result_name = 'proposal_%04d.json' % count
        row = journal.read(result_name) if journal.exists(result_name) else None
        report['recovered'].append({'worker': name, 'task': task,
            'saved_proposals': status['proposals'], 'previous_results_unchanged': True,
            'unresolved_database_calls': status.get('unresolved_database_calls'),
            'replayed_result': None if row is None else {
                k: row.get(k) for k in ('status', 'runtime', 'database_attempts')}})
    save('verification.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['configure', 'restart', 'verify'])
    args = parser.parse_args()
    globals()[args.action]()
