#!/usr/bin/env python3
"""Launch TaskSet RGPE in eight free, disjoint four-core allocations."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from taskset_transfer_common import atomic_json, digest, read_split, PROTOCOL, METRIC

SOURCE = Path('/root/data/logs/taskset_major_families_cpu_20260929')
FSBO = Path('/root/data/logs/taskset_fsbo_test210_20261002')
TRAINING = Path('/root/data/logs/taskset_transfer_train838_20261002')
CAMPAIGN = Path('/root/data/logs/taskset_rgpe_test210_20261003')
PYTHON = '/root/data/cys/baseline/envs/bl_taskset_lolbo/bin/python'
HOSTNAME = 'cs-fa498-8793f-server'
SLOTS = [
    dict(name='rgpe.w00', cpus=[0,1,2,3], previous=['w14','w15']),
    dict(name='rgpe.w01', cpus=[4,5,6,7], previous=['w00']),
    dict(name='rgpe.w02', cpus=[12,13,14,15], previous=['w02']),
    dict(name='rgpe.w03', cpus=[16,17,18,19], previous=['w03']),
    dict(name='rgpe.w04', cpus=[24,25,26,27], previous=['w05']),
    dict(name='rgpe.w05', cpus=[28,29,30,31], previous=['w06']),
    dict(name='rgpe.w06', cpus=[36,37,38,39], previous=['w08','w09']),
    dict(name='rgpe.w07', cpus=[42,43,46,47], previous=['w11','w13']),
]


def read(path):
    return json.loads(Path(path).read_text())


def lock(path):
    handle = Path(path).open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def completed_record(history, result, job, budget):
    if result is None:
        return False
    rows = history['observations']
    ok = sum(r['trial_state']==0 and math.isfinite(r['objectives'][0]) for r in rows)
    if (len(rows)!=budget or result.get('job')!=job or result.get('returncode')!=0
            or result.get('dummy') is not False or result.get('n_observations')!=budget
            or result.get('n_successful')!=ok
            or result.get('status')!=('complete' if ok==budget else 'complete_with_failures')):
        raise ValueError('Completed result does not match its history')
    return True


def physical(cpus):
    cores = set()
    for cpu in cpus:
        root = Path('/sys/devices/system/cpu')/('cpu%d' % cpu)/'topology'
        cores.add((int((root/'physical_package_id').read_text()), int((root/'core_id').read_text())))
    return cores


def occupied(cpus):
    intended = physical(cpus)
    hits = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(path.name)
            if pid in (os.getpid(), os.getppid()):
                continue
            state = path.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]
            affinity = os.sched_getaffinity(pid)
            if state not in ('Z','X') and len(affinity) < os.cpu_count() and physical(affinity) & intended:
                hits.append(dict(pid=pid, cpus=sorted(affinity), name=path.joinpath('comm').read_text().strip()))
        except (OSError, ValueError):
            pass
    return hits


def prepare():
    if CAMPAIGN.exists():
        raise ValueError('Campaign exists; preserve its provenance')
    split = read_split()
    CAMPAIGN.mkdir()
    ignore = shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(FSBO/'code', CAMPAIGN/'code', ignore=ignore)
    scripts = CAMPAIGN/'code/examples/benchmark/domains'
    for name in ('run_taskset_rgpe.py','taskset_rgpe_worker.py','launch_taskset_rgpe.py','taskset_transfer_common.py'):
        shutil.copy2(Path(__file__).with_name(name), scripts/name)
    ml = Path('/root/data/cys/blockdata_archive/llmoptimizer_benchmark')
    shutil.copy2(ml/'examples/benchmark/rgpe_worker.py', CAMPAIGN/'code/examples/benchmark/rgpe_worker.py')
    for name in ('protocol.py','schema.py'):
        shutil.copy2(ml/'mindware/components/optimizers/transfer_hpo'/name,
                     CAMPAIGN/'code/mindware/components/optimizers/transfer_hpo'/name)
    shutil.copytree('/root/data/cys/baseline/repos/transfer-hpo-framework/rgpe', CAMPAIGN/'code/rgpe', ignore=ignore)
    for name in ('split.json','task_baselines.json'):
        shutil.copy2(SOURCE/name, CAMPAIGN/name)
    for name in ('runtime_template','eval_wrapper','taskset_data'):
        shutil.copytree(SOURCE/name, CAMPAIGN/name, symlinks=True, ignore=ignore)
    for name in ('assets','locks','workers','results','job_logs','runtimes','preflight'):
        (CAMPAIGN/name).mkdir()
    for name in ('configspace.json','manifest.json'):
        shutil.copy2(TRAINING/'corpus'/name, CAMPAIGN/'assets'/name)
    shutil.copy2(TRAINING/'rgpe/status.json', CAMPAIGN/'assets/training_status.json')
    tasks = [j['task_id'] for j in read(SOURCE/'distributed/plan.json')['jobs'] if j['side']=='test' and j['method']=='tpe']
    assert len(tasks)==210 and set(tasks)==set(split['test'])
    recipe = dict(method='rgpe', host='LLMOptimizer4', hostname=HOSTNAME, slots=SLOTS, tasks=tasks,
                  jobs=[dict(id='test.rgpe.'+t, task_id=t, method='rgpe', side='test',
                             output_root=str(CAMPAIGN/'test')) for t in tasks],
                  seed=1, evaluations_per_task=100, protocol=PROTOCOL, objective_metric=METRIC,
                  startup_trials=3, candidates=256, ranking_samples=1000,
                  acquisition='TAF', normalization='None', sampling_mode='bootstrap',
                  weight_dilution_strategy='probabilistic-ld', split_sha256=digest(CAMPAIGN/'split.json'),
                  runner_sha256=digest(scripts/'run_taskset_rgpe.py'),
                  worker_sha256=digest(scripts/'taskset_rgpe_worker.py'), created_at=time.time())
    atomic_json(CAMPAIGN/'recipe.json', recipe)
    print(json.dumps(dict(campaign=str(CAMPAIGN), slots=SLOTS)))


def seal():
    meta = read(CAMPAIGN/'assets/ensemble.json')
    if meta['status'] != 'complete' or digest(CAMPAIGN/'assets/ensemble.pkl') != meta['cache_sha256']:
        raise ValueError('Source ensemble is incomplete')
    recipe = read(CAMPAIGN/'recipe.json')
    recipe.update(cache_sha256=meta['cache_sha256'], source_count=len(meta['source_tasks']))
    atomic_json(CAMPAIGN/'recipe.json', recipe)
    hashes = {str(p.relative_to(CAMPAIGN)):digest(p)
              for name in ('code','assets','runtime_template','eval_wrapper','taskset_data')
              for p in (CAMPAIGN/name).rglob('*') if p.is_file() and not p.is_symlink()}
    hashes.update({name:digest(CAMPAIGN/name) for name in ('recipe.json','split.json','task_baselines.json')})
    atomic_json(CAMPAIGN/'source_hashes.json', hashes)
    print(json.dumps(dict(source_count=recipe['source_count'], cache_sha256=recipe['cache_sha256'])))


def verify():
    if socket.gethostname() != HOSTNAME:
        raise ValueError('Launch requires LLMOptimizer4')
    for name, expected in read(CAMPAIGN/'source_hashes.json').items():
        if digest(CAMPAIGN/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    seen = set()
    for slot in SLOTS:
        cores = physical(slot['cpus'])
        if len(cores)!=4 or cores & seen:
            raise ValueError('Worker physical CPUs overlap')
        seen |= cores
        if occupied(slot['cpus']):
            raise ValueError('Allocation is occupied: '+str(slot))
        for previous in slot['previous']:
            if previous == 'w06':
                if read(TRAINING/'controller_state.json')['status'] != 'complete':
                    raise ValueError('Source-model training still owns CPUs 28-31')
            elif read(SOURCE/'distributed/workers'/('LLMOptimizer4.'+previous+'.json'))['status'] != 'complete':
                raise ValueError('Previous worker has not completed: '+previous)
    limit, used = int(Path('/sys/fs/cgroup/memory.max').read_text()), int(Path('/sys/fs/cgroup/memory.current').read_text())
    if limit-used < 64*2**30:
        raise ValueError('Less than 64 GiB of container memory headroom')
    return read(CAMPAIGN/'recipe.json')


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


def work(name, check=False):
    if socket.gethostname() != HOSTNAME:
        raise ValueError('Wrong worker host')
    recipe = read(CAMPAIGN/'recipe.json')
    slot = next(s for s in SLOTS if s['name']==name)
    if occupied(slot['cpus']):
        raise ValueError('Worker allocation is occupied')
    locks = [lock(SOURCE/'locks'/('worker.LLMOptimizer4.'+w+'.lock')) for w in slot['previous']]
    locks += [lock(SOURCE/'locks'/('core.LLMOptimizer4.%d.lock' % c)) for c in slot['cpus']]
    os.sched_setaffinity(0, slot['cpus'])
    import sys
    sys.path.insert(0, str(CAMPAIGN/'code'))
    from mindware.domains.history import history_exists, load_openbox_history
    from mindware.domains.taskset_archive import pack_job
    state_path = CAMPAIGN/'workers'/(name+('.check' if check else '')+'.json')
    base = dict(slot, pid=os.getpid(), host='LLMOptimizer4', started_at=time.time(),
                start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    jobs = [j for j in recipe['jobs'] if j['task_id']=='quadratic_family_seed0'] if check else recipe['jobs']
    budget = 4 if check else 100
    try:
        for job in jobs:
            try:
                job_lock = lock(CAMPAIGN/'locks'/(job['id']+('.check' if check else '')+'.lock'))
            except BlockingIOError:
                continue
            with job_lock:
                run = CAMPAIGN/('preflight' if check else 'test')/job['task_id']/'rgpe/seed1'
                if history_exists(run/'history.json'):
                    h = load_openbox_history(run/'history.json')
                    meta = h['meta_info']
                    if (h['task_id']!=job['task_id'] or meta['optimizer']!='rgpe' or meta['seed']!=1
                            or meta['cache_sha256']!=recipe['cache_sha256'] or meta['budget']!=budget
                            or meta['protocol']!=PROTOCOL or meta['objective_metric']!=METRIC
                            or meta['runner_sha256']!=recipe['runner_sha256']
                            or meta['worker_sha256']!=recipe['worker_sha256']):
                        raise ValueError('Existing RGPE history identity mismatch')
                    if len(h['observations'])==budget:
                        result_path = CAMPAIGN/'preflight_result.json' if check else CAMPAIGN/'results'/(job['id']+'.json')
                        saved = read(result_path) if result_path.exists() else None
                        if completed_record(h,saved,job,budget):
                            if not check and not run.with_name('seed1.tar.gz').exists():
                                pack_job(CAMPAIGN,job,cleanup=True,lock=job_lock)
                            continue
                runtime = CAMPAIGN/'runtimes'/(job['id']+('.check' if check else ''))
                if not runtime.exists():
                    shutil.copytree(CAMPAIGN/'runtime_template', runtime, symlinks=True)
                command = [PYTHON, str(CAMPAIGN/'code/examples/benchmark/domains/run_taskset_rgpe.py'),
                           '--campaign',str(CAMPAIGN),'--task',job['task_id'],'--trials',str(budget)]
                if check:
                    command.append('--check')
                log = CAMPAIGN/'job_logs'/(job['id']+('.check' if check else '')+'.log')
                with log.open('a') as stream:
                    child = subprocess.Popen(command, env=environment(runtime,slot['cpus']), cwd=CAMPAIGN/'code',
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                        pass_fds=tuple(f.fileno() for f in [*locks,job_lock]))
                while child.poll() is None:
                    progress = read(run/'progress.json') if (run/'progress.json').exists() else {}
                    atomic_json(state_path,dict(base,status='running',job=job,child_pid=child.pid,
                                                progress=progress,updated_at=time.time()))
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError('RGPE task exited %s: %s' % (child.returncode,log))
                history = load_openbox_history(run/'history.json')
                rows = history['observations']
                if len(rows)!=budget:
                    raise ValueError('RGPE process exited with incomplete history')
                ok = sum(r['trial_state']==0 and math.isfinite(r['objectives'][0]) for r in rows)
                result = dict(job=job,host='LLMOptimizer4',worker=name,returncode=0,dummy=False,
                              n_observations=len(rows),n_successful=ok,finished_at=time.time(),
                              status='complete' if ok==budget else 'complete_with_failures')
                if check:
                    selection = rows[3]['extra_info']['info']['selection']
                    if ok!=4 or selection['phase']!='rgpe_taf' or selection['n_sources']!=838:
                        raise ValueError('Real preflight did not complete a full RGPE decision')
                    result['cache_sha256'] = recipe['cache_sha256']
                    result['runner_sha256'] = recipe['runner_sha256']
                    atomic_json(CAMPAIGN/'preflight_result.json', result)
                else:
                    atomic_json(CAMPAIGN/'results'/(job['id']+'.json'), result)
                    pack_job(CAMPAIGN,job,cleanup=True,lock=job_lock)
        atomic_json(state_path,dict(base,status='complete',updated_at=time.time()))
    except BaseException as exc:
        atomic_json(state_path,dict(base,status='failed',error=repr(exc),updated_at=time.time()))
        raise


def launch():
    recipe = verify()
    check = read(CAMPAIGN/'preflight_result.json')
    if (check['returncode'] or check['dummy'] or check['n_successful']!=4
            or check['cache_sha256']!=recipe['cache_sha256'] or check['runner_sha256']!=recipe['runner_sha256']):
        raise ValueError('A matching real RGPE preflight must pass first')
    for slot in SLOTS:
        with (CAMPAIGN/'workers'/(slot['name']+'.log')).open('a') as stream:
            child = subprocess.Popen([PYTHON,__file__,'worker','--name',slot['name']],
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                         MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
        atomic_json(CAMPAIGN/'workers'/(slot['name']+'.launch.json'),dict(slot,pid=child.pid,launched_at=time.time()))
        print(json.dumps(dict(name=slot['name'],pid=child.pid,cpus=slot['cpus'])),flush=True)
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','seal','verify','check','launch','worker'))
    parser.add_argument('--name',default='rgpe.w00')
    args = parser.parse_args()
    if args.action=='prepare': prepare()
    elif args.action=='seal': seal()
    elif args.action=='verify': print(json.dumps(verify()['slots']))
    elif args.action=='launch': launch()
    else: work(args.name,check=args.action=='check')
