"""Stop owned QueryOpt workers and release reproducible private tmpfs database clones."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

DEPLOY = Path('/root/data/cys/logs/query_hosts24_20261026')
OUT = DEPLOY / 'stopped_for_peptide_20261027'
HOSTS = {'LLMOptimizer2': 'cs-94bac-0693e-server', 'LLMOptimizer4': 'cs-fa498-8793f-server'}

def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)

def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, start_ticks=stat[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except OSError:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=HOSTS, required=True)
    parser.add_argument('--retire-backed-databases', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == HOSTS[args.host]
    OUT.mkdir(exist_ok=True)
    workers = [w for w in json.loads((DEPLOY / 'manifest.json').read_text())['workers']
               if w['host'] == args.host]
    stopped = []
    for worker in workers:
        record = json.loads((DEPLOY / 'processes' / (worker['name'] + '.json')).read_text())
        if identity(record['pid']) == record:
            command = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().decode().split('\0')
            assert str(DEPLOY / 'manage.py') in command and worker['name'] in command
            os.kill(record['pid'], signal.SIGTERM)
            stopped.append(record)
    deadline = time.monotonic() + 60
    while any(identity(row['pid']) == row for row in stopped):
        if time.monotonic() >= deadline:
            raise RuntimeError('A QueryOpt manager did not stop; no database will be removed')
        time.sleep(1)
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                command = (proc / 'cmdline').read_bytes().decode().split('\0')
            except OSError:
                continue
            if '--method' in command and command[command.index('--method') + 1] == 'random' and any(
                    token.endswith('run_query_ceb_baselines_v3.py') for token in command):
                raise RuntimeError('A random-search evaluator is still alive')
    report = dict(host=args.host, stopped_at=time.time(), stopped_processes=stopped,
        reason='User stopped Random Search and reassigned its CPUs to peptide BO', databases=[])
    for worker in workers:
        target = Path(worker['pgdata'])
        if not target.exists():
            continue
        pidfile = target / 'postmaster.pid'
        if pidfile.exists():
            pid = int(pidfile.read_text().splitlines()[0])
            command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            assert str(target) in command and '/root/data/cys/envs/pg16/bin/postgres' in command
            subprocess.run(['taskset', '-c', ','.join(map(str, worker['db_cpus'])),
                'setpriv', '--reuid=10000', '--regid=10000', '--clear-groups',
                '/root/data/cys/envs/pg16/bin/pg_ctl', '-D', str(target), '-m', 'fast',
                '-w', '-t', '60', 'stop'], cwd='/tmp', check=True)
        control = subprocess.check_output(['/root/data/cys/envs/pg16/bin/pg_controldata', str(target)],
            text=True, env=dict(os.environ, LC_ALL='C'))
        state = next(line.split(':', 1)[1].strip() for line in control.splitlines()
                     if line.startswith('Database cluster state:'))
        assert state == 'shut down' and not pidfile.exists()
        row = dict(worker=worker['name'], path=str(target), state=state, removed=False)
        report['databases'].append(row)
    write(OUT / (args.host + '.json'), report)
    if args.retire_backed_databases:
        backup = json.loads(Path('/root/data/cys/logs/semanticopt_recovery_20261007/database_backup.json').read_text())
        digest = hashlib.sha256()
        with Path(backup['archive']).open('rb') as stream:
            for chunk in iter(lambda: stream.read(2**20), b''):
                digest.update(chunk)
        assert digest.hexdigest() == backup['sha256'] and backup['tar_compare'] == 'passed'
        for row in report['databases']:
            worker = next(w for w in workers if w['name'] == row['worker'])
            target = Path(row['path'])
            if worker['source_pgdata'] != backup['archive']:
                continue
            assert target.resolve().is_relative_to('/dev/shm') and target.stat().st_uid == 10000
            assert not (target / 'postmaster.pid').exists() and (target / 'PG_VERSION').read_text().strip() == '16'
            for log in target.glob('*.log'):
                shutil.copy2(log, OUT / (worker['name'] + '.' + log.name))
            row.update(restore_archive=backup['archive'], restore_sha256=backup['sha256'])
            write(OUT / (args.host + '.json'), report)
            shutil.rmtree(target)
            row['removed'] = True
            write(OUT / (args.host + '.json'), report)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
