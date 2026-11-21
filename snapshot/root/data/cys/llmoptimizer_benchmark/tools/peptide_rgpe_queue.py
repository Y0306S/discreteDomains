#!/usr/bin/env python3
"""Adopt existing source fits and dispatch memory-guarded exact RGPE workers."""

import argparse
import json
import os
from pathlib import Path
import resource
import sqlite3
import subprocess
import socket
import time

import peptide_transfer as transfer

BASE = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
WORKERS = Path('/root/data/logs/peptide_test200_baselines_20261027_v2/workers')
SCRIPT = Path(__file__).with_name('peptide_rgpe_fit.py')
PYTHON = '/root/data/cys/baseline/envs/bl_rgpe/bin/python'


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        words = [word.decode() for word in (root/'cmdline').read_bytes().split(b'\0') if word]
        if ('--task' not in words or '--corpus' not in words or '--output' not in words
                or words[words.index('--corpus')+1] != str(BASE/'corpus')
                or words[words.index('--output')+1] != str(BASE/'rgpe')):
            return None
        status = dict(line.split(':',1) for line in (root/'status').read_text().splitlines() if ':' in line)
        return dict(pid=pid, task=words[words.index('--task')+1],
            cpu=int(words[words.index('--cpu')+1]) if '--cpu' in words else int(words[words.index('--cpus')+1]),
            compact=str(SCRIPT) in words, rss_gib=int(status['VmRSS'].split()[0])/2**20,
            peak_gib=int(status['VmHWM'].split()[0])/2**20)
    except (OSError, ValueError, KeyError):
        return None


def bank_tasks(manifest, corpus_digest):
    path = BASE/'rgpe/models.sqlite'
    if not path.exists():
        return set()
    with sqlite3.connect(str(path), timeout=120) as db:
        rows = db.execute('SELECT task,metadata FROM models').fetchall()
    complete = set()
    for task, value in rows:
        meta = json.loads(value)
        if (task not in manifest['source_tasks'] or meta['source_payload_sha256'] != manifest['task_hashes'][task]
                or meta['corpus_sha256'] != corpus_digest
                or not meta['reload_verified']):
            raise ValueError('Incompatible source GP bank record: '+task)
        complete.add(task)
    return complete


def assigned_tasks(tasks, assignments, name):
    """A fixed partition prevents hosts racing to fit the same source task."""
    if assignments is None:
        return set(tasks)
    if assignments['default_owner'] != 'remote':
        raise ValueError('Unsupported default assignment owner')
    owners = assignments['owners']
    if not set(owners).issubset(set(tasks)) or any(v not in ('remote', 'machine3') for v in owners.values()):
        raise ValueError('Invalid source task assignment')
    return {task for task in tasks if owners.get(task, 'remote') == name}


def run(args):
    settings = json.loads(args.settings.read_text())
    os.sched_setaffinity(0, set(settings.get('queue_cpus', [46,47])))
    validation = Path(settings['validation_report'])
    if (transfer.digest(validation) != settings['validation_sha256']
            or not json.loads(validation.read_text())['passed']
            or transfer.digest(SCRIPT) != settings['runner_sha256']
            or transfer.digest(SCRIPT.with_name('peptide_rgpe_exact.py')) != settings['objective_sha256']):
        raise ValueError('RGPE exact-gradient validation changed')
    with transfer.lock(BASE/('queue_'+args.name+'.lock')):
        manifest, db = transfer.load_corpus(BASE/'corpus')
        db.close()
        corpus_digest = transfer.digest(BASE/'corpus/manifest.json')
        assignments = json.loads(args.assignments.read_text()) if args.assignments else None
        if assignments is not None and assignments['corpus_sha256'] != corpus_digest:
            raise ValueError('Assignment corpus changed')
        owned = assigned_tasks(manifest['source_tasks'], assignments, args.name)
        active, children = {}, {}
        for entry in Path('/proc').iterdir():
            if entry.name.isdigit():
                row = process(int(entry.name))
                if row:
                    if row['task'] not in owned:
                        raise ValueError('Live source fit belongs to another queue: '+row['task'])
                    if row['task'] in {r['task'] for r in active.values()}:
                        raise ValueError('Duplicate live source fits')
                    active[row['pid']] = row
        known_peak = settings['initial_compact_peak_gib']
        while True:
            complete = bank_tasks(manifest, corpus_digest)
            for pid, row in list(active.items()):
                current = process(pid)
                if current is None:
                    if pid in children:
                        result = children.pop(pid).wait()
                        if result != 0:
                            raise RuntimeError('RGPE worker failed: '+row['task'])
                    if row['task'] not in complete:
                        raise RuntimeError('Source fit exited without a committed model: '+row['task'])
                    if row['compact']:
                        with sqlite3.connect(str(BASE/'rgpe/models.sqlite'), timeout=120) as bank:
                            meta = json.loads(bank.execute('SELECT metadata FROM models WHERE task=?',
                                                           (row['task'],)).fetchone()[0])
                        known_peak = max(known_peak, row['peak_gib'], meta['peak_rss_gib'])
                    del active[pid]
                else:
                    active[pid] = current
                    if current['compact']:
                        known_peak = max(known_peak, current['peak_gib'])
            pending = [task for task in manifest['source_tasks']
                       if task in owned and task not in complete and task not in {r['task'] for r in active.values()}]
            if len(complete) == len(manifest['source_tasks']) and not active:
                transfer.atomic_json(BASE/(args.name+'_queue.json'),dict(status='complete',rgpe_completed=800,
                    fsbo_updates=100000,source_tasks=800,execution_engine='exact_trace_contraction'))
                return
            reservation = max(4.0, known_peak*1.75)
            cgroup=Path('/sys/fs/cgroup')
            available=(int((cgroup/'memory.max').read_text())-int((cgroup/'memory.current').read_text()))/2**30
            growth=sum(max(0, (reservation if r['compact'] else settings['legacy_peak_gib'])-r['rss_gib'])
                       for r in active.values())
            admissible=available-growth-settings['memory_headroom_gib']
            used_cpus={r['cpu'] for r in active.values()}
            for cpu in settings['fit_cpus']:
                if not pending or cpu in used_cpus or admissible < reservation:
                    continue
                task=pending.pop(0)
                command=[PYTHON,str(SCRIPT),'--corpus',str(BASE/'corpus'),'--output',str(BASE/'rgpe'),
                         '--task',task,'--cpu',str(cpu)]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                log_name = 'rgpe_cpu%d.log'%cpu if args.name == 'remote' else 'rgpe_%s_cpu%d.log'%(args.name,cpu)
                with (BASE/log_name).open('ab') as output:
                    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,
                        stderr=subprocess.STDOUT,start_new_session=True,env=env)
                children[child.pid]=child
                active[child.pid]=dict(pid=child.pid,task=task,cpu=cpu,compact=True,rss_gib=0,peak_gib=0)
                admissible-=reservation
                used_cpus.add(cpu)
            transfer.atomic_json(BASE/(args.name+'_queue.json'),dict(status='training' if active or pending else 'waiting_external',fsbo_returncode=0,
                pid=os.getpid(),host=socket.gethostname(),updated=time.time(),queue_name=args.name,
                assigned_tasks=len(owned),assigned_pending=len(pending),
                rgpe_completed=len(complete),rgpe_total=800,
                rgpe_active={str(r['cpu']):r for r in active.values()},
                requested_compact_workers=settings['requested_workers'],
                compact_workers=sum(r['compact'] for r in active.values()),
                adopted_legacy_workers=sum(not r['compact'] for r in active.values()),
                memory_guard_worker_limit=len([r for r in active.values() if r['compact']]),
                compact_peak_gib=known_peak,per_worker_reservation_gib=reservation,
                memory_growth_reserved_gib=growth,available_after_reservations_gib=admissible,
                execution_engine='exact_trace_contraction'))
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=('remote', 'machine3'), default='remote')
    parser.add_argument('--settings', type=Path, default=WORKERS/'rgpe_runtime.json')
    parser.add_argument('--assignments', type=Path)
    args = parser.parse_args()
    try:
        if args.name != 'remote' and args.assignments is None:
            raise ValueError('Additional hosts require an explicit task partition')
        run(args)
    except BaseException as exc:
        transfer.atomic_json(BASE/('queue_'+args.name+'_failure.json'),dict(error=repr(exc),time=time.time()))
        raise
