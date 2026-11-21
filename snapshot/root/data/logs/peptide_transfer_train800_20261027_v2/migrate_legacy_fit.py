#!/usr/bin/env python3
"""Replace only the legacy source fit, preserving other live fits and saved GPs."""

import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

BASE = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
WORKERS = Path('/root/data/logs/peptide_test200_baselines_20261027_v2/workers')
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
REPORT = BASE / 'legacy_migration.json'
PYTHON = '/root/data/cys/baseline/envs/bl_rgpe/bin/python'


def write(path, value):
    temp = path.with_name(path.name + '.migration.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        args = [v.decode() for v in (root / 'cmdline').read_bytes().split(b'\0') if v]
        if not args:
            return None
        status = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines()
                      if ':' in line)
        return dict(pid=pid, args=args, start_ticks=(root / 'stat').read_text().split()[21],
                    cpus=sorted(os.sched_getaffinity(pid)),
                    rss_gib=int(status.get('VmRSS', '0').split()[0]) / 2**20)
    except (OSError, ValueError, KeyError):
        return None


def stop(row):
    current = process(row['pid'])
    if current is None:
        return
    if current['start_ticks'] != row['start_ticks'] or current['args'] != row['args']:
        raise RuntimeError('Process identity changed before termination')
    os.kill(row['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while process(row['pid']) is not None:
        if time.monotonic() > deadline:
            raise RuntimeError('Process did not exit after SIGTERM: %d' % row['pid'])
        time.sleep(0.2)


def records():
    with sqlite3.connect('file:' + str(BASE / 'rgpe/models.sqlite') + '?mode=ro',
                         uri=True, timeout=60) as db:
        return {task: dict(metadata=json.loads(meta), sha256=sha)
                for task, meta, sha in db.execute('SELECT task,metadata,sha256 FROM models')}


def launch_queue(queue):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (BASE / 'queue_remote_v3.log').open('ab') as log:
        child = subprocess.Popen([PYTHON, str(TOOLS / 'peptide_rgpe_queue.py')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=env)
    updated = dict(queue, remote=json.dumps(dict(pid=child.pid, side='remote')),
                   requested_rgpe_workers=11, legacy_fits_preserved=False,
                   legacy_fit_replaced_with_exact=True)
    write(BASE / 'queue.json', updated)
    return child.pid


def main():
    os.sched_setaffinity(0, {46, 47})
    settings_path = WORKERS / 'rgpe_runtime.json'
    settings = json.loads(settings_path.read_text())
    validation_path = Path(settings['validation_report'])
    validation = json.loads(validation_path.read_text())
    if not (validation['passed'] and digest(validation_path) == settings['validation_sha256']
            and digest(TOOLS / 'peptide_rgpe_fit.py') == settings['runner_sha256']
            and digest(TOOLS / 'peptide_rgpe_exact.py') == settings['objective_sha256']):
        raise RuntimeError('Validated exact-GP implementation changed')
    if settings['fit_cpus'] != list(range(36, 46)) or settings['requested_workers'] != 10:
        raise RuntimeError('Unexpected allocation; migration requires the existing ten-fit layout')
    queue = json.loads((BASE / 'queue.json').read_text())
    supervisor = process(json.loads(queue['remote'])['pid'])
    if supervisor is None or str(TOOLS / 'peptide_rgpe_queue.py') not in supervisor['args']:
        raise RuntimeError('Expected source-fitting supervisor is not running')
    fits = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        row = process(int(entry.name))
        if row is None:
            continue
        args = row['args']
        if not all(flag in args for flag in ('--task', '--corpus', '--output')):
            continue
        if (args[args.index('--corpus') + 1] != str(BASE / 'corpus')
                or args[args.index('--output') + 1] != str(BASE / 'rgpe')):
            continue
        row['task'] = args[args.index('--task') + 1]
        row['compact'] = str(TOOLS / 'peptide_rgpe_fit.py') in args
        fits.append(row)
    legacy = [row for row in fits if not row['compact']]
    if len(legacy) != 1 or legacy[0]['task'] != 'pep_001' or legacy[0]['cpus'] != [0]:
        raise RuntimeError('Expected only the legacy pep_001 fit on CPU 0')
    if str(TOOLS / 'peptide_transfer.py') not in legacy[0]['args']:
        raise RuntimeError('Unexpected legacy fitting command')
    before = records()
    state = dict(status='prepared', time=time.time(), old_supervisor=supervisor,
                 legacy_fit=legacy[0], preserved_fits=[r for r in fits if r['compact']],
                 completed_before=len(before), previous_settings=settings,
                 runner_sha256=settings['runner_sha256'], objective_sha256=settings['objective_sha256'],
                 interrupted_task_restart_from_beginning=True)
    write(REPORT, state)
    # Stop dispatch first; existing detached fitting processes continue running.
    stop(supervisor)
    new_pid = None
    try:
        state['status'] = 'supervisor_stopped'
        write(REPORT, state)
        stop(legacy[0])
        state['status'] = 'legacy_fit_stopped'
        write(REPORT, state)
        updated_settings = dict(settings, fit_cpus=[0] + list(range(36, 46)),
                                requested_workers=11, legacy_migration_report=str(REPORT))
        write(settings_path, updated_settings)
        launched_at = time.time()
        new_pid = launch_queue(queue)
        state.update(status='verifying', new_supervisor_pid=new_pid,
                     new_fit_cpus=updated_settings['fit_cpus'], launched_at=launched_at)
        write(REPORT, state)
        deadline = time.monotonic() + 60
        while True:
            if process(new_pid) is None:
                raise RuntimeError('Replacement supervisor exited; inspect queue_remote_v3.log')
            progress_path = BASE / 'remote_queue.json'
            progress = json.loads(progress_path.read_text())
            if progress_path.stat().st_mtime >= launched_at:
                if progress.get('adopted_legacy_workers') == 0 and progress.get('compact_workers') == 11:
                    replacement = progress['rgpe_active']['0']
                    if replacement['compact'] and (replacement['task'] == 'pep_001'
                            or 'pep_001' in records()):
                        break
            if time.monotonic() > deadline:
                raise RuntimeError('Replacement queue has not confirmed all eleven exact fits')
            time.sleep(1)
        after = records()
        if any(after.get(task) != record for task, record in before.items()):
            raise RuntimeError('A previously committed GP record changed')
        current_pids = {row['pid'] for row in progress['rgpe_active'].values()}
        if any(row['pid'] not in current_pids and row['task'] not in after
               for row in state['preserved_fits']):
            raise RuntimeError('A preserved fit is missing without a completed model')
        state.update(status='complete', completed_after=len(after),
                     completed_records_unchanged=True, preserved_fits_verified=True,
                     replacement_fit=replacement, compact_workers=11, legacy_workers=0,
                     time_completed=time.time())
        write(REPORT, state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status='verification_failed', error=repr(exc), time_failed=time.time())
        # If migration fails before launching, restore supervision of any surviving fits.
        if new_pid is None:
            state['recovery_supervisor_pid'] = launch_queue(queue)
        write(REPORT, state)
        raise


if __name__ == '__main__':
    main()
