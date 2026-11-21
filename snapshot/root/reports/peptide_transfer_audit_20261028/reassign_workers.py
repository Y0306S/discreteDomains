#!/usr/bin/env python3
"""Move ten remote TPE CPU slots to the validated compact exact-GP queue."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

ROOT=Path(__file__).resolve().parent
TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
BASE=Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
CAMPAIGN=Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
WORKERS=CAMPAIGN/'workers'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temp,path)


def alive(pid,script):
    try:
        return str(script).encode() in Path('/proc/%d/cmdline'%pid).read_bytes().split(b'\0')
    except OSError:
        return False


def configure():
    validation=json.loads((ROOT/'exact_validation.json').read_text())
    assert validation['passed']
    for name,file in [('objective','peptide_rgpe_exact.py'),('runner','peptide_rgpe_fit.py')]:
        assert validation[name+'_sha256']==digest(TOOLS/file)
    gate=WORKERS/'rgpe_exact_validation.json'
    write(gate,validation)
    previous=json.loads((WORKERS/'tpe_runtime.json').read_text())
    if previous['workers'] != 58 or previous['cpus']['remote'] != list(range(10,46)):
        raise ValueError('Unexpected TPE layout; preserve current allocation')
    write(WORKERS/'tpe_runtime_before_rgpe_20261028.json',previous)
    settings=dict(previous)
    settings['cpus']=dict(previous['cpus'],remote=list(range(10,36)))
    settings['workers']=48
    settings['rgpe_reserved_cpus']=list(range(36,46))
    write(WORKERS/'tpe_runtime.json',settings)
    old=json.loads((BASE/'remote_queue.json').read_text())
    fsbo=json.loads((BASE/'fsbo/status.json').read_text())
    assert fsbo['status']=='complete' and fsbo['reload_verified']
    runtime=dict(fit_cpus=list(range(36,46)),requested_workers=10,
        validation_report=str(gate),validation_sha256=digest(gate),
        runner_sha256=validation['runner_sha256'],objective_sha256=validation['objective_sha256'],
        initial_compact_peak_gib=validation['peak_rss_gib'],legacy_peak_gib=old['first_fit_peak_gib'],
        memory_headroom_gib=16,user_authorized_worker_shift=True,source_tasks=800)
    write(WORKERS/'rgpe_runtime.json',runtime)
    print(json.dumps(dict(configured=True,tpe_workers=48,rgpe_new_workers=10)),flush=True)


def stop_tpe():
    settings=json.loads((WORKERS/'rgpe_runtime.json').read_text())
    records=json.loads((WORKERS/'remote.json').read_text())
    selected=[row for row in records if row['method']=='tpe' and row['cpu'] in settings['fit_cpus']]
    if len(selected)!=10:
        raise ValueError('Ten matching TPE worker records are required')
    tasks={}
    parents={row['pid']:row['cpu'] for row in selected}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            status=dict(line.split(':',1) for line in (entry/'status').read_text().splitlines() if ':' in line)
            words=(entry/'cmdline').read_bytes().split(b'\0')
            if int(status['PPid']) in parents and b'--task' in words:
                tasks[parents[int(status['PPid'])]]=words[words.index(b'--task')+1].decode()
        except (OSError,ValueError): pass
    for row in selected:
        if alive(row['pid'],row['command'][1]): os.kill(row['pid'],signal.SIGTERM)
    deadline=time.monotonic()+180
    while any(alive(row['pid'],row['command'][1]) for row in selected):
        if time.monotonic()>deadline:
            raise RuntimeError('Workers are still draining; no forced termination performed')
        time.sleep(1)
    checks=[]
    for cpu,task in tasks.items():
        path=CAMPAIGN/'tpe/runs'/task/'runtime.sqlite'
        if path.exists():
            with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=60) as db:
                assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
            checks.append(dict(cpu=cpu,task=task,recovery_database_ok=True))
        else:
            assert (CAMPAIGN/'tpe/runs'/(task+'.tar.gz')).exists()
            checks.append(dict(cpu=cpu,task=task,completed_archive=True))
    write(WORKERS/'tpe_to_rgpe_reassignment.json',dict(time=time.time(),stopped_workers=selected,
        recovery_checks=checks,forced_kills=0,remaining_tpe_workers=48,reserved_rgpe_cpus=settings['fit_cpus']))
    print(json.dumps(dict(stopped=10,recovery_checks=len(checks),forced_kills=0)),flush=True)


def replace_queue():
    queue=json.loads((BASE/'queue.json').read_text())
    old_pid=json.loads(queue['remote'])['pid']
    new_script=TOOLS/'peptide_rgpe_queue.py'
    old_script=TOOLS/'peptide_training_queue.py'
    old_state=json.loads((BASE/'remote_queue.json').read_text())
    if alive(old_pid,new_script):
        raise ValueError('Replacement queue is already running')
    if not alive(old_pid,old_script):
        raise ValueError('Expected original queue supervisor is absent')
    write(WORKERS/'rgpe_queue_before_reassignment.json',dict(queue=queue,state=old_state))
    os.kill(old_pid,signal.SIGTERM)
    deadline=time.monotonic()+30
    while alive(old_pid,old_script):
        if time.monotonic()>deadline: raise RuntimeError('Original queue has not exited')
        time.sleep(0.2)
    for row in old_state['rgpe_active'].values():
        assert alive(row['pid'],TOOLS/'peptide_transfer.py')
    python='/root/data/cys/baseline/envs/bl_rgpe/bin/python'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    with (BASE/'queue_remote_v2.log').open('ab') as output:
        child=subprocess.Popen([python,str(new_script)],stdin=subprocess.DEVNULL,stdout=output,
            stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write(BASE/'queue.json',dict(status='training_remote',remote=json.dumps(dict(pid=child.pid,side='remote')),
        source_tasks=800,fsbo_updates=100000,requested_rgpe_workers=10,fsbo_complete=True,
        legacy_fits_preserved=True,machine3_used=False))
    print(json.dumps(dict(queue_pid=child.pid,legacy_fits_preserved=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('configure','stop-tpe','replace-queue'))
    args=parser.parse_args()
    {'configure':configure,'stop-tpe':stop_tpe,'replace-queue':replace_queue}[args.action]()
