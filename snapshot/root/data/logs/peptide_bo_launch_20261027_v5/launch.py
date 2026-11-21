"""Launch audited one-core peptide workers without overlapping physical reservations."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from resources import snapshot

ROOT = Path('/root/data/logs/peptide_bo_release_20261027_v5')
CAMPAIGN = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
LOCAL = [4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 25, 26,
         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
         3, 5, 6, 7, 17, 18, 19]
REMOTE = [22, 23, 27, 39] + [cpu for cpu in range(48) if cpu not in (22, 23, 27, 39)]

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=['LLMOptimizer2', 'LLMOptimizer4'], required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    cores = LOCAL if args.host == 'LLMOptimizer2' else REMOTE
    if args.start < 0 or args.count < 1 or args.start + args.count > len(cores):
        parser.error('Worker range exceeds the audited physical cores')
    sys.path.insert(0, str(ROOT / 'tools'))
    import peptide_bo_v5 as runner
    runner.verify(CAMPAIGN)
    runner.preflight(CAMPAIGN)
    directory = CAMPAIGN / 'workers'
    directory.mkdir(exist_ok=True)
    with (directory / (args.host + '.launch.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record_path = directory / (args.host + '.json')
        records = json.loads(record_path.read_text()) if record_path.exists() else []
        for index in range(args.start, args.start + args.count):
            if any(row['local_index'] == index for row in records):
                print(json.dumps(dict(skipped=index, reason='already recorded')), flush=True)
                continue
            resources = snapshot()
            core = cores[index]
            if core not in resources['free_physical_cpus']:
                raise RuntimeError(f'CPU {core} is already reserved on {args.host}')
            minimum = 48 if args.host == 'LLMOptimizer2' else 32
            if resources['available_gib'] < minimum:
                print(json.dumps(dict(stopped='RAM headroom', **resources)), flush=True)
                break
            if args.host == 'LLMOptimizer2':
                global_index = index if index < 27 else index + 4
            else:
                global_index = 27 + index if index < 4 else 34 + index
            command = ['taskset', '-c', str(core), PYTHON, '-u', str(ROOT / 'tools/peptide_bo_v5.py'),
                       'worker', '--campaign', str(CAMPAIGN), '--cpu', str(core),
                       '--index', str(global_index), '--workers', '82']
            if args.host == 'LLMOptimizer2':
                command += ['--gpu', str([0, 2, 3][index % 3])]
            else:
                port = [18871, 18872, 18873][index % 3]
                command += ['--server', f'http://127.0.0.1:{port}']
            log = directory / f'{args.host}.w{index:02d}.log'
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            with log.open('ab') as output:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, env=env, cwd=ROOT, start_new_session=True)
            row = dict(host=args.host, local_index=index, index=global_index, cpu=core,
                pid=child.pid, command=command, log=str(log), started=time.time(),
                available_gib_before=resources['available_gib'])
            records.append(row)
            save(record_path, records)
            print(json.dumps(row), flush=True)

if __name__ == '__main__':
    main()
