"""Preserve failed QueryOpt calls and park their dedicated database capacity."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/root/data/cys/logs/query_train800_author_bo_20261002')
OUT = ROOT / 'recovery_20261002'
CASES = [('LLMOptimizer3.w01', 'CEB_2A870', 41, 219638),
         ('LLMOptimizer3.w02', 'CEB_8A92', 35, 211399)]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage():
    code = OUT / 'code'
    code.mkdir(exist_ok=True)
    source = Path('/root/blockdata/llmoptimizer_benchmark/examples/benchmark/domains/run_query_ceb_baselines.py')
    shutil.copy2(source, code / source.name)
    shutil.copy2(ROOT / 'code/run_query_ceb_baselines.py', code / 'run_query_ceb_baselines.original.py')
    shutil.copy2('/root/blockdata/llmoptimizer_benchmark/tests/test_query_ceb_baselines.py', code / 'test_query_ceb_baselines.py')
    evidence = []
    for name, task, number, block in CASES:
        directory = ROOT / 'outputs' / task / 'lolbo/seed0'
        pending = directory / f'proposal_{number:04d}.pending.json'
        log = ROOT / 'logs' / f'{name}.{task}.run.log'
        assert 'psycopg.errors.IoError:' in log.read_text()
        assert f'could not read block {block}' in log.read_text()
        assert not directory.joinpath(f'proposal_{number:04d}.json').exists()
        assert len(list(directory.glob('proposal_[0-9][0-9][0-9][0-9].json'))) == number
        archive = OUT / f'{name}.{task}.before_recovery.tar.gz'
        if not archive.exists():
            with tarfile.open(archive, 'w:gz') as stream:
                for path in (directory, log, ROOT / 'workers' / f'{name}.json',
                             ROOT / 'claims' / f'{task}.json'):
                    stream.add(path, arcname=str(path.relative_to(ROOT)))
        record = dict(worker=name, task=task, proposal_index=number, error_block=block,
                      archive=str(archive), archive_sha256=digest(archive),
                      log=str(log), log_sha256=digest(log), pending_sha256=digest(pending))
        evidence.append(record)
        ledger = directory / f'proposal_{number:04d}.database_attempts.json'
        if not ledger.exists():
            write(ledger, [dict(proposal=read(pending), status='io_error', sqlstate='58030',
                                error=f'could not read block {block} in file base/16385/16475.1: Input/output error',
                                elapsed_seconds=None, started_at=None, finished_at=None,
                                historical_recovery=True, evidence=record)])
    write(OUT / 'evidence.json', evidence)
    write(OUT / 'release.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
          original_sha256=digest(ROOT / 'code/run_query_ceb_baselines.py'),
          fixed_sha256=digest(code / source.name), tests='16 focused unittest cases passed',
          status='staged; QueryOpt workers remain stopped',
          future_resume_argument=['--resume-runner-sha256', digest(ROOT / 'code/run_query_ceb_baselines.py')],
          recovery='Confirmed SQLSTATE 58030 can replay the exact saved proposal on explicit resume; unknown outcomes block.'))
    print('Recovery staged; original campaign code preserved.')


def verify_and_release():
    import psycopg
    results = []
    for name, task, number, block in CASES:
        worker_path = ROOT / 'workers' / f'{name}.json'
        worker = read(worker_path)
        proc = Path('/proc') / str(worker['pid'])
        assert not proc.exists(), f'Worker is still alive: {name}'
        database_path = ROOT / 'databases' / f'{name}.json'
        database = read(database_path)
        pgdata = Path(worker['pgdata'])
        if not (pgdata / 'postmaster.pid').exists():
            assert database.get('status') == 'parked_for_ckpt32'
            continue
        pid = int((pgdata / 'postmaster.pid').read_text().splitlines()[0])
        assert pid == database['pid']
        assert os.sched_getaffinity(pid) == set(worker['db_cpus'])
        with psycopg.connect(host='127.0.0.1', port=worker['port'], dbname='imdb',
                             user='imdb', connect_timeout=5, autocommit=True) as conn:
            others = conn.execute("SELECT pid FROM pg_stat_activity WHERE backend_type='client backend' AND pid<>pg_backend_pid()").fetchall()
            assert not others, f'Other database clients exist: {others}'
            relation = conn.execute("SELECT pg_relation_filepath('cast_info'), pg_relation_size('cast_info')").fetchone()
            conn.execute("SET statement_timeout='30s'")
            rows = conn.execute('SELECT count(*) FROM cast_info').fetchone()[0]
            assert rows == 36244344, rows
            fd = os.open(pgdata / 'base/16385/16475.1', os.O_RDONLY)
            try:
                page_bytes = len(os.pread(fd, 8192, (block - 131072) * 8192))
            finally:
                os.close(fd)
            assert page_bytes == 8192
        result = dict(worker=name, task=task, checked_at=datetime.now(timezone.utc).isoformat(),
                      relation=relation, full_scan_rows=rows, failed_page_bytes=page_bytes,
                      read_error_reproduced=False, underlying_storage_fault_resolved='unproven')
        results.append(result)
        write(OUT / f'{name}.read_validation.json', result)
        subprocess.run(['taskset', '-c', ','.join(map(str, worker['db_cpus'])),
                        'setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
                        '/root/data/cys/envs/pg16/bin/pg_ctl', '-D', str(pgdata),
                        '-m', 'fast', '-w', '-t', '30', 'stop'], check=True)
        assert not (pgdata / 'postmaster.pid').exists()
        worker.update(previous_status=worker['status'], status='parked_for_ckpt32',
                      resource_replacement='ckpt32_rgpe_llmoptimizer3_20261002',
                      recovery_release=str(OUT), updated_at=datetime.now(timezone.utc).isoformat())
        database.update(status='parked_for_ckpt32', stopped_at=datetime.now(timezone.utc).isoformat())
        write(worker_path, worker)
        write(database_path, database)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'verify-release'])
    args = parser.parse_args()
    {'stage': stage, 'verify-release': verify_and_release}[args.action]()
