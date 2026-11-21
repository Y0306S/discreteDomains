#!/usr/bin/env python3
"""Run TaskSet FSBO in the five released LLMOptimizer4 TPE allocations."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from taskset_transfer_common import atomic_json, digest, read_split, PROTOCOL, METRIC

SOURCE = Path('/root/data/logs/taskset_major_families_cpu_20260929')
TRAINING = Path('/root/data/logs/taskset_transfer_train838_20261002')
CAMPAIGN = Path('/root/data/logs/taskset_fsbo_test210_20261002')
PYTHON = '/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python'
ML_CODE = Path('/root/data/cys/blockdata_archive/llmoptimizer_benchmark')
SLOTS = [{'name': 'fsbo.w01', 'previous': 'LLMOptimizer4.w01', 'cpus': [8,9,10,11]},
         {'name': 'fsbo.w04', 'previous': 'LLMOptimizer4.w04', 'cpus': [20,21,22,23]},
         {'name': 'fsbo.w07', 'previous': 'LLMOptimizer4.w07', 'cpus': [32,33,34,35]},
         {'name': 'fsbo.w10', 'previous': 'LLMOptimizer4.w10', 'cpus': [40,41]},
         {'name': 'fsbo.w12', 'previous': 'LLMOptimizer4.w12', 'cpus': [44,45]}]


def read(path):
    return json.loads(Path(path).read_text())


def lock(path):
    handle = Path(path).open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def prepare():
    if CAMPAIGN.exists():
        raise ValueError('Campaign exists; do not overwrite launch provenance')
    split = read_split()
    status = read(TRAINING/'fsbo/status.json')
    corpus = read(TRAINING/'corpus/manifest.json')
    if (status['status'] != 'complete' or not status['reload_verified']
            or status['identity']['corpus_sha256'] != digest(TRAINING/'corpus/manifest.json')
            or status['weights_sha256'] != digest(TRAINING/'fsbo/weights')
            or set(corpus['source_tasks']) != set(split['train'])):
        raise ValueError('FSBO training provenance is incomplete')
    for name, expected in corpus['files'].items():
        if digest(TRAINING/'corpus'/name) != expected:
            raise ValueError('Changed corpus: '+name)
    CAMPAIGN.mkdir()
    shutil.copytree(SOURCE/'code', CAMPAIGN/'code', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    scripts = CAMPAIGN/'code/examples/benchmark/domains'
    for name in ('run_taskset_fsbo.py','launch_taskset_fsbo.py','taskset_transfer_common.py'):
        shutil.copy2(Path(__file__).with_name(name), scripts/name)
    shutil.copy2(ML_CODE/'examples/benchmark/fsbo_worker.py', CAMPAIGN/'code/examples/benchmark/fsbo_worker.py')
    transfer = CAMPAIGN/'code/mindware/components/optimizers/transfer_hpo'
    transfer.mkdir(exist_ok=True)
    for name in ('__init__.py','protocol.py'):
        shutil.copy2(ML_CODE/'mindware/components/optimizers/transfer_hpo'/name, transfer/name)
    for name in ('split.json','task_baselines.json'):
        shutil.copy2(SOURCE/name, CAMPAIGN/name)
    for name in ('runtime_template','eval_wrapper','taskset_data'):
        shutil.copytree(SOURCE/name, CAMPAIGN/name, symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('assets','locks','workers','results','job_logs','runtimes'):
        (CAMPAIGN/name).mkdir()
    shutil.copy2(TRAINING/'fsbo/weights', CAMPAIGN/'assets/weights')
    shutil.copy2(TRAINING/'fsbo/status.json', CAMPAIGN/'assets/training_status.json')
    shutil.copy2(TRAINING/'corpus/configspace.json', CAMPAIGN/'assets/configspace.json')
    import numpy as np
    source_tasks = read(TRAINING/'corpus/tasks.json')
    with np.load(TRAINING/'corpus/features.npz') as arrays:
        vector = arrays['X_fsbo'][0].tolist()
    meta = dict(method='fsbo_paper', domain='taskset_cash', status='complete', dim=corpus['fsbo_dim'],
                schema_fingerprint=digest(TRAINING/'corpus/configspace.json'),
                train_tasks=corpus['fsbo_train_tasks'], validation_tasks=corpus['fsbo_validation_tasks'],
                weights_sha256=status['weights_sha256'], config=status['identity']['config'],
                training_status_sha256=digest(TRAINING/'fsbo/status.json'),
                corpus_sha256=digest(TRAINING/'corpus/manifest.json'),
                encoding_probe={'config':source_tasks[0]['configs'][0], 'vector':vector})
    atomic_json(CAMPAIGN/'assets/meta.json', meta)
    # Preserve the interleaved ordering already used by the TaskSet baseline queue.
    source_plan = read(SOURCE/'distributed/plan.json')
    tasks = [j['task_id'] for j in source_plan['jobs'] if j['side']=='test' and j['method']=='tpe']
    assert len(tasks)==210 and set(tasks)==set(split['test'])
    jobs = [dict(id='test.fsbo.'+t, task_id=t, method='fsbo', side='test',
                 output_root=str(CAMPAIGN/'test')) for t in tasks]
    recipe = dict(method='fsbo', host='LLMOptimizer4', hostname='cs-fa498-8793f-server',
                  slots=SLOTS, tasks=tasks, jobs=jobs, seed=1, evaluations_per_task=100,
                  protocol=PROTOCOL, objective_metric=METRIC, startup_trials=3,
                  candidates=256, finetune_epochs=100, paper_ei_unchanged=True,
                  weights_sha256=status['weights_sha256'], split_sha256=digest(CAMPAIGN/'split.json'),
                  runner_sha256=digest(scripts/'run_taskset_fsbo.py'),
                  worker_sha256=digest(CAMPAIGN/'code/examples/benchmark/fsbo_worker.py'),
                  created_at=time.time())
    atomic_json(CAMPAIGN/'recipe.json', recipe)
    hashes = {str(p.relative_to(CAMPAIGN)):digest(p) for folder in ('code','assets','eval_wrapper','runtime_template','taskset_data')
              for p in (CAMPAIGN/folder).rglob('*') if p.is_file() and not p.is_symlink()}
    atomic_json(CAMPAIGN/'source_hashes.json', hashes)
    print(json.dumps({k:recipe[k] for k in ('method','slots','weights_sha256','evaluations_per_task')}, indent=2))


def environment(runtime, cpus):
    threads = str(len(cpus))
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                NUMEXPR_NUM_THREADS=threads, TASKSET_EVAL_THREADS=threads,
                TF_NUM_INTRAOP_THREADS=threads, TF_NUM_INTEROP_THREADS='1',
                TASKSET_RUNTIME_ROOT=str(runtime), TASKSET_EVAL_DIR=str(CAMPAIGN/'eval_wrapper'),
                TASKSET_ROOT=str(CAMPAIGN/'taskset_data'), TASKSET_BASELINE_CACHE=str(CAMPAIGN/'task_baselines.json'),
                TASKSET_EVAL_PYTHON='/root/data/cys/conda_envs/taskset_eval/bin/python',
                PYTHONPATH=str(CAMPAIGN/'code'))


def occupied(cpus):
    all_cpus = os.cpu_count()
    hits = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            pid=int(path.name)
            if pid in (os.getpid(), os.getppid()):
                continue
            state=path.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]
            affinity=os.sched_getaffinity(pid)
            if state not in ('Z','X') and len(affinity)<all_cpus and affinity.intersection(cpus):
                hits.append(dict(pid=pid,cpus=sorted(affinity),name=path.joinpath('comm').read_text().strip()))
        except (OSError,ValueError):
            pass
    return hits


def verify():
    recipe = read(CAMPAIGN/'recipe.json')
    if socket.gethostname()!=recipe['hostname']:
        raise ValueError('Launch requires LLMOptimizer4')
    for name, expected in read(CAMPAIGN/'source_hashes.json').items():
        if digest(CAMPAIGN/name)!=expected:
            raise ValueError('Frozen source changed: '+name)
    for slot in SLOTS:
        state=read(SOURCE/'distributed/workers'/(slot['previous']+'.json'))
        if state['status']!='complete':
            raise ValueError('Previous TPE allocation has not finished: '+slot['previous'])
        hits=occupied(slot['cpus'])
        if hits:
            raise ValueError('CPU allocation is occupied: '+str(hits))
    return recipe


def work(name, check=False, dummy=False, trials=100):
    recipe=read(CAMPAIGN/'recipe.json')
    if socket.gethostname()!=recipe['hostname']:
        raise ValueError('Wrong worker host')
    slot=next(s for s in SLOTS if s['name']==name)
    hits=occupied(slot['cpus'])
    if hits:
        raise ValueError('Worker CPUs are occupied: '+str(hits))
    os.sched_setaffinity(0,slot['cpus'])
    locks=[lock(SOURCE/'locks'/('worker.'+slot['previous']+'.lock'))]
    locks += [lock(SOURCE/'locks'/('core.LLMOptimizer4.%d.lock'%c)) for c in slot['cpus']]
    sys.path.insert(0,str(CAMPAIGN/'code'))
    from mindware.domains.history import history_exists,load_openbox_history
    from mindware.domains.taskset_archive import pack_job
    state_path=CAMPAIGN/'workers'/(name+('.check' if check else '')+'.json')
    base=dict(slot,pid=os.getpid(),host='LLMOptimizer4',started_at=time.time())
    try:
        for job in recipe['jobs'][:1] if check else recipe['jobs']:
            try:
                job_lock=lock(CAMPAIGN/'locks'/(job['id']+('.check' if check else '')+'.lock'))
            except BlockingIOError:
                continue
            with job_lock:
                run=CAMPAIGN/('preflight' if check else 'test')/job['task_id']/'fsbo/seed1'
                if history_exists(run/'history.json'):
                    h=load_openbox_history(run/'history.json')
                    if (h['task_id']!=job['task_id'] or h['meta_info']['optimizer']!='fsbo'
                            or h['meta_info']['weights_sha256']!=recipe['weights_sha256']):
                        raise ValueError('Existing result identity mismatch')
                    if len(h['observations'])>=trials:
                        continue
                runtime=CAMPAIGN/'runtimes'/(job['id']+('.check' if check else ''))
                if not runtime.exists():
                    shutil.copytree(CAMPAIGN/'runtime_template',runtime,symlinks=True)
                command=[PYTHON,str(CAMPAIGN/'code/examples/benchmark/domains/run_taskset_fsbo.py'),
                         '--campaign',str(CAMPAIGN),'--task',job['task_id'],'--trials',str(trials)]
                if check: command.append('--check')
                if dummy: command.append('--dummy')
                log=CAMPAIGN/'job_logs'/(job['id']+('.check' if check else '')+'.log')
                with log.open('a') as stream:
                    process=subprocess.Popen(command,env=environment(runtime,slot['cpus']),
                        cwd=CAMPAIGN/'code',stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                        pass_fds=tuple(f.fileno() for f in [*locks,job_lock]))
                while process.poll() is None:
                    progress=read(run/'progress.json') if (run/'progress.json').exists() else {}
                    atomic_json(state_path,dict(base,status='running',job=job,child_pid=process.pid,
                                               progress=progress,updated_at=time.time()))
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError('FSBO task exited %s: %s'%(process.returncode,log))
                history=load_openbox_history(run/'history.json')
                rows=history['observations']
                if len(rows)!=trials:
                    raise ValueError('FSBO process exited with incomplete history')
                import math
                ok=sum(r['trial_state']==0 and math.isfinite(r['objectives'][0]) for r in rows)
                result=dict(job=job,host='LLMOptimizer4',worker=name,returncode=0,dummy=bool(dummy),
                            n_observations=len(rows),n_successful=ok,finished_at=time.time(),
                            status='complete' if ok==trials else 'complete_with_failures')
                if check:
                    atomic_json(CAMPAIGN/'preflight_result.json',result)
                else:
                    atomic_json(CAMPAIGN/'results'/(job['id']+'.json'),result)
                    pack_job(CAMPAIGN,job,cleanup=True,lock=job_lock)
        atomic_json(state_path,dict(base,status='complete',updated_at=time.time()))
    except BaseException as exc:
        atomic_json(state_path,dict(base,status='failed',error=repr(exc),updated_at=time.time()))
        raise


def launch():
    verify()
    check=read(CAMPAIGN/'preflight_result.json')
    if check['returncode'] or check.get('dummy') is not False or check['n_successful']<4:
        raise ValueError('Real FSBO preflight has not passed')
    for slot in SLOTS:
        with (CAMPAIGN/'workers'/(slot['name']+'.log')).open('a') as log:
            p=subprocess.Popen([PYTHON,__file__,'worker','--name',slot['name']],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                         MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(CAMPAIGN/'workers'/(slot['name']+'.launch.json'),dict(slot,pid=p.pid,launched_at=time.time()))
        print(json.dumps(dict(name=slot['name'],pid=p.pid,cpus=slot['cpus'])),flush=True)
        time.sleep(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify','worker','check','launch'))
    parser.add_argument('--name',default='fsbo.w01')
    parser.add_argument('--dummy',action='store_true')
    parser.add_argument('--trials',type=int,default=4)
    args=parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='verify': print(json.dumps(verify()['slots']))
    elif args.action=='launch': launch()
    else: work(args.name,check=args.action=='check',dummy=args.dummy,trials=args.trials if args.action=='check' else 100)
