#!/usr/bin/env python3
"""Bounded, isolated read-only query timing under 1/2/3 PostgreSQL CPU limits."""
import json
import os
from pathlib import Path
import subprocess
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/dev/shm/query_cpu_capacity_20261106_v1')
SOURCE = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
PG = Path('/root/data/cys/envs/pg16/bin')
NATIVE = SOURCE / 'native'
ENV = dict(os.environ, LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib:' + os.environ.get('LD_LIBRARY_PATH', ''),
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
PREFIX = ['setpriv', '--reuid=10000', '--regid=10000', '--clear-groups']


def save(value):
    temp = ROOT / 'cpu_benchmark.tmp'
    temp.write_text(json.dumps(value, indent=2))
    os.replace(str(temp), str(ROOT / 'cpu_benchmark.json'))


def control(action, cpus=None):
    command = PREFIX + [str(PG / 'pg_ctl'), '-D', str(DATA), '-w', '-t', '60']
    if action == 'start':
        settings = json.loads((SOURCE / 'plan.json').read_text())['postgres_settings']
        options = ['-p', '19540', '-h', '127.0.0.1', '-k', str(DATA), '-c', 'max_connections=20',
                   '-c', 'dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib']
        for key, value in settings.items():
            options += ['-c', key + '=' + value]
        command += ['-l', str(DATA / 'probe.log'), '-o',
                    shlex.join(options), 'start']
        command = ['taskset', '-c', ','.join(map(str, cpus))] + command
    else:
        command += ['-m', 'fast', 'stop']
    subprocess.run(command, env=ENV, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main():
    import psycopg
    sys.path.insert(0, str(NATIVE))
    from workload.workloads import get_workload_set, WorkloadSpec, OracleCodec
    from oracle.oracle import _decode_query
    import re
    workload = get_workload_set('CEB_3K')
    train = (SOURCE / 'train800.txt').read_text().split()
    chosen = []
    for family in ['1A', '2A', '5A', '8A', '9A', '11A']:
        for task in train:
            if not task.startswith('CEB_' + family) or not re.fullmatch('CEB_' + family + r'\d+', task):
                continue
            history = json.loads((SOURCE / 'outputs' / task / 'lolbo/seed0/history.json').read_text())
            candidates = [r for r in history['observations'][:49]
                if not r['extra_info']['censored'] and 0.02 <= r['objectives'][0] <= 3.0]
            if not candidates:
                continue
            row = candidates[0]
            match = re.fullmatch(r'CEB_(\d+[A-Z])(\d+)', task)
            sql_path = NATIVE / 'workload/ceb-3k' / match[1].lower() / ((match[1] + match[2]).lower() + '.sql')
            definition = workload.extract_query(sql_path.read_text(), task)
            spec = WorkloadSpec.from_definition(definition, OracleCodec.Aliases)
            chosen.append(dict(task=task, sql=_decode_query(spec, row['extra_info']['plan']),
                               historical_runtime=row['objectives'][0]))
            break
    if len(chosen) != 6:
        raise ValueError('Insufficient representative train-only benchmark queries')
    report = dict(status='provisioning', purpose='cpu_capacity_probe_only', actual_optimization_runs=0,
        selected_queries=[{k:v for k,v in q.items() if k!='sql'} for q in chosen], results=[],
        settings={}, started=time.time(), database=str(DATA))
    save(report)
    if not DATA.exists():
        DATA.mkdir(mode=0o700)
        subprocess.run(['tar', '-xf', '/root/data/cys/logs/semanticopt_recovery_20261007/pg_imdb_d.clean_shutdown.tar',
                        '-C', str(DATA), '--strip-components=1'], check=True)
        subprocess.run(['chown', '-R', '10000:10000', str(DATA)], check=True)
    elif not (DATA / 'PG_VERSION').is_file():
        raise ValueError('Existing benchmark directory is not a PostgreSQL cluster')
    if (DATA / 'postmaster.pid').exists():
        raise ValueError('Restored database unexpectedly has a postmaster PID')
    report['status'] = 'benchmarking'
    save(report)
    for cores in [3, 1, 2]:
        running = False
        try:
            control('start', list(range(cores)))
            running = True
            with psycopg.connect(host='127.0.0.1', port=19540, dbname='imdb', user='imdb', password='imdb', autocommit=True) as conn:
                conn.execute("LOAD 'pg_hint_plan'")
                conn.execute('SET pg_hint_plan.enable_hint=on')
                conn.execute('SET join_collapse_limit=1')
                conn.execute('SET default_transaction_read_only=on')
                conn.execute("SET statement_timeout='5s'")
                for key in ['max_parallel_workers_per_gather', 'max_parallel_workers', 'max_worker_processes', 'shared_buffers', 'work_mem']:
                    report['settings'][key] = conn.execute('SHOW ' + key).fetchone()[0]
                for query in chosen:
                    for repeat in [-1, 0, 1]:
                        started = time.perf_counter()
                        try:
                            result = conn.execute('EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) ' + query['sql']).fetchone()[0][0]
                            nodes = []
                            def walk(node):
                                nodes.append(node)
                                for child in node.get('Plans', []):
                                    walk(child)
                            walk(result['Plan'])
                            value = dict(task=query['task'], postgres_cores=cores, repeat=repeat, status='complete',
                                wall_seconds=time.perf_counter()-started,
                                execution_seconds=result['Execution Time']/1000,
                                parallel_workers_planned=max(n.get('Workers Planned', 0) for n in nodes),
                                parallel_workers_launched=max(n.get('Workers Launched', 0) for n in nodes),
                                root_type=result['Plan']['Node Type'])
                        except psycopg.errors.QueryCanceled:
                            value = dict(task=query['task'], postgres_cores=cores, repeat=repeat, status='timeout',
                                         wall_seconds=time.perf_counter()-started)
                        report['results'].append(value)
                        save(report)
                        print(json.dumps(value), flush=True)
        finally:
            if running:
                control('stop')
    report.update(status='complete', finished=time.time(), database_stopped=True)
    save(report)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        (ROOT / 'cpu_benchmark_failure.json').write_text(json.dumps(dict(error=repr(exc))))
        raise
