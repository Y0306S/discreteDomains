#!/usr/bin/env python3
"""Run the authorized full export, then validated CPU transfer training on machine 4."""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import peptide_transfer as transfer

BASE = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
SCRIPT = Path(transfer.__file__).resolve()
QUEUE = Path(__file__).resolve()
EXPORT_PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
FSBO_PYTHON = '/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python'
RGPE_PYTHON = '/root/data/cys/baseline/envs/bl_rgpe/bin/python'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def launch(command, log, cpus=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(ENV)
    if cpus and len(cpus.split(',')) == 2:
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with log.open('ab') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    return process


def run_local():
    BASE.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {40})
    with transfer.lock(BASE / 'queue_local.lock'):
        export_server = os.environ.get('PEPTIDE_EXPORT_SERVER', 'http://127.0.0.1:18872')
        command = [EXPORT_PYTHON, str(SCRIPT), 'export', '--output', str(BASE / 'corpus'),
                   '--cpus', '18', '--server', export_server]
        child = launch(command, BASE / 'export.log')
        transfer.atomic_json(BASE / 'queue.json', dict(status='exporting', export_pid=child.pid,
            source_tasks=800, fsbo_updates=100000, requested_rgpe_workers=8,
            script_sha256=transfer.digest(SCRIPT), export_server=export_server, machine3_used=False))
        if child.wait() != 0:
            raise RuntimeError('Export failed; transfer training was not launched')
        command = ['ssh', '-o', 'BatchMode=yes', 'LLMOptimizer4', EXPORT_PYTHON, str(QUEUE), 'launch-remote']
        result = subprocess.run(command, env=ENV, check=True, text=True, capture_output=True)
        transfer.atomic_json(BASE / 'queue.json', dict(status='training_remote', remote=result.stdout.strip(),
            source_tasks=800, fsbo_updates=100000, requested_rgpe_workers=8, machine3_used=False))


def bank_tasks():
    path = BASE / 'rgpe' / 'models.sqlite'
    if not path.exists():
        return set()
    with sqlite3.connect(str(path), timeout=120) as db:
        return {row[0] for row in db.execute('SELECT task FROM models')}


def run_remote():
    os.sched_setaffinity(0, {46, 47})
    with transfer.lock(BASE / 'queue_remote.lock'):
        manifest, db = transfer.load_corpus(BASE / 'corpus')
        db.close()
        check = launch([FSBO_PYTHON, str(SCRIPT), 'fsbo', '--corpus', str(BASE / 'corpus'),
            '--output', str(BASE / 'fsbo_component'), '--cpus', '8,9', '--steps', '100'],
            BASE / 'fsbo_component.log', '8,9')
        if check.wait() != 0:
            raise RuntimeError('FSBO component update/reload check failed')
        fsbo = launch([FSBO_PYTHON, str(SCRIPT), 'fsbo', '--corpus', str(BASE / 'corpus'),
            '--output', str(BASE / 'fsbo'), '--cpus', '8,9', '--steps', '100000'], BASE / 'fsbo.log', '8,9')
        pending = [t for t in manifest['source_tasks'] if t not in bank_tasks()]
        active = {}
        worker_limit = 1
        reference_peak = None
        while pending or active:
            for cpu, (task, child) in list(active.items()):
                result = child.poll()
                if result is None:
                    continue
                if result != 0:
                    raise RuntimeError('Full RGPE source fit failed: ' + task)
                del active[cpu]
                with sqlite3.connect(str(BASE / 'rgpe' / 'models.sqlite'), timeout=120) as bank:
                    meta = json.loads(bank.execute('SELECT metadata FROM models WHERE task=?', (task,)).fetchone()[0])
                reference_peak = max(reference_peak or 0, meta['peak_rss_gib'])
                # ARD exact GPs can materialize N*N*256 gradients. Scale concurrency
                # only after a complete fit provides a real memory measurement.
                available = (int(Path('/sys/fs/cgroup/memory.max').read_text()) -
                             int(Path('/sys/fs/cgroup/memory.current').read_text())) / 2**30
                worker_limit = min(8, max(1, int(available / max(16, reference_peak * 1.5))))
            for cpu in range(worker_limit):
                if not pending or cpu in active:
                    continue
                task = pending.pop(0)
                child = launch([RGPE_PYTHON, str(SCRIPT), 'rgpe-task', '--corpus', str(BASE / 'corpus'),
                    '--output', str(BASE / 'rgpe'), '--cpus', str(cpu), '--task', task], BASE / ('rgpe_cpu%d.log' % cpu))
                active[cpu] = (task, child)
            transfer.atomic_json(BASE / 'remote_queue.json', dict(status='training', fsbo_pid=fsbo.pid,
                fsbo_returncode=fsbo.poll(), rgpe_completed=len(bank_tasks()), rgpe_total=800,
                rgpe_active={str(c): dict(task=t, pid=p.pid) for c, (t, p) in active.items()},
                memory_guard_worker_limit=worker_limit, first_fit_peak_gib=reference_peak))
            if fsbo.poll() not in (None, 0):
                raise RuntimeError('Full FSBO training failed')
            time.sleep(10)
        if fsbo.wait() != 0:
            raise RuntimeError('Full FSBO training failed')
        transfer.atomic_json(BASE / 'remote_queue.json', dict(status='complete', rgpe_completed=800,
            fsbo_updates=100000, source_tasks=800))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('local', 'remote', 'launch-local', 'launch-remote'))
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.action.startswith('launch-'):
        side = args.action.split('-')[1]
        process = launch([EXPORT_PYTHON, str(QUEUE), side], BASE / ('queue_%s.log' % side))
        print(json.dumps(dict(pid=process.pid, side=side)), flush=True)
        return
    try:
        (run_local if args.action == 'local' else run_remote)()
    except BaseException as exc:
        transfer.atomic_json(BASE / ('queue_%s_failure.json' % args.action), dict(error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
