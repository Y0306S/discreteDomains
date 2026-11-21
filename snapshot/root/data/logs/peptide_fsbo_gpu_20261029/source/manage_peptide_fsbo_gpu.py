#!/usr/bin/env python3
"""Start bounded FSBO services and resume the existing queues after validation."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import sqlite3
import subprocess
import time
import urllib.request

import numpy as np

TOOLS = Path(__file__).resolve().parent
CAMPAIGN = Path('/root/data/logs/peptide_transfer_test200_20261028_v1')
OUTPUT = Path('/root/data/logs/peptide_fsbo_gpu_20261029')
PYTHON = '/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def live(pid, marker):
    try:
        command = (Path('/proc')/str(pid)/'cmdline').read_bytes()
        return marker.encode() in command
    except FileNotFoundError:
        return False


def start(count):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    token = OUTPUT/'service_token'
    if not token.exists():
        with os.fdopen(os.open(str(token), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as stream:
            stream.write(secrets.token_hex(32)+'\n')
    plan_path = OUTPUT/'services.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else dict(services=[])
    slots = [(3, 45, 18931), (3, 4, 18932), (3, 6, 18933), (3, 7, 18934)]
    for gpu, cpu, port in slots[:count]:
        existing = next((r for r in plan['services'] if r['port'] == port), None)
        if existing and live(existing['pid'], 'peptide_fsbo_gpu_service.py'):
            continue
        command = [PYTHON, '-u', str(TOOLS/'peptide_fsbo_gpu_service.py'),
            '--recipe', str(CAMPAIGN/'fsbo/recipe.json'), '--token-file', str(token),
            '--port', str(port), '--cpu', str(cpu), '--memory-fraction', '0.08']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        log = OUTPUT/('service_%d.log' % port)
        with log.open('ab') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=env, start_new_session=True)
        row = dict(pid=child.pid, gpu=gpu, cpu=cpu, port=port,
                   endpoint='http://127.0.0.1:%d' % port, log=str(log), started=time.time())
        plan['services'] = [r for r in plan['services'] if r['port'] != port]+[row]
        write(plan_path, plan)
    print(json.dumps(plan,indent=2))


def activate():
    validation_path = OUTPUT/'validation/validation.json'
    validation = json.loads(validation_path.read_text())
    if not validation['passed'] or not validation['checks'].get('evaluation_resume'):
        raise ValueError('Full-scale and evaluation-resume validation are required')
    for name, expected in validation['code_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Validated GPU source changed: ' + name)
    plan = json.loads((OUTPUT/'services.json').read_text())
    endpoints = [row['endpoint'] for row in plan['services']]
    for endpoint in endpoints:
        with urllib.request.urlopen(endpoint+'/health',timeout=10) as response:
            health = json.load(response)
        if health['errors'] or health['recipe_sha256'] != validation['recipe_sha256']:
            raise ValueError('GPU service is not healthy: ' + endpoint)
    config = dict(enabled=True, recipe_sha256=validation['recipe_sha256'],
        code_sha256=validation['code_sha256'], token_file=str(OUTPUT/'service_token'),
        validation=dict(path=str(validation_path), sha256=digest(validation_path)),
        endpoints=endpoints, activated_at=time.time(),
        policy='retain all evaluations; fit every round on CUDA; CPU task coordinators',
        server_gpu=3, memory_cap_gib_per_service=6.4)
    baseline = {}
    for side in ('local','remote'):
        queue = json.loads((CAMPAIGN/'workers'/('queue_'+side+'.json')).read_text())
        for worker in queue['active']:
            if worker['method'] != 'fsbo':
                continue
            history = json.loads((CAMPAIGN/'fsbo/runs'/worker['task']/'history.json').read_text())
            observed = [[r['sequence'],r['score'],r['cached'],r['feasible']] for r in history['observations']]
            baseline[worker['task']] = dict(side=side,rows=len(observed),
                proposals=history['proposals'],fresh_evaluations=history['fresh_evaluations'],
                prefix_sha256=hashlib.sha256(json.dumps(observed,separators=(',',':')).encode()).hexdigest())
    write(OUTPUT/'migration_baseline.json',baseline)
    write(CAMPAIGN/'fsbo/gpu_runtime.json', config)
    print(json.dumps(config,indent=2))


def validate_evaluation():
    from check_peptide_fsbo_gpu import StateUnpickler
    from peptide_fsbo_gpu_eval import GPUModelClient
    from peptide_transfer_policy import effective_recipe

    os.sched_setaffinity(0, {45})
    scale_path = OUTPUT/'validation/scale_validation.json'
    report_path = OUTPUT/'validation/validation.json'
    if not scale_path.exists():
        shutil.copy2(report_path, scale_path)
    report = json.loads(scale_path.read_text())
    plan = json.loads((OUTPUT/'services.json').read_text())
    endpoints = [r['endpoint'] for r in plan['services']]
    config = dict(enabled=True, recipe_sha256=report['recipe_sha256'],
        code_sha256=report['code_sha256'], token_file=str(OUTPUT/'service_token'),
        endpoints=endpoints,
        validation=dict(path=str(scale_path), sha256=digest(scale_path)))
    recipe_path = CAMPAIGN/'fsbo/recipe.json'
    recipe = effective_recipe(recipe_path)
    with sqlite3.connect('file:' + str(CAMPAIGN/'fsbo/runs/pep_151/runtime.sqlite') + '?mode=ro', uri=True) as db:
        payload = db.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0]
    saved = StateUnpickler(io.BytesIO(payload)).load()
    features = np.asarray(saved['features'])
    low, high = np.asarray(recipe['lower']), np.asarray(recipe['upper'])
    x = ((features-low)/np.where(high > low, high-low, 1.)).astype(np.float32)
    scores = np.asarray([r['score'] for r in saved['rows']])
    indices = np.arange(5100) % len(x)
    x, scores = x[indices], scores[indices]
    candidates = np.random.RandomState(32).uniform(0,1,(5000,256)).astype(np.float32)
    def exercise(endpoint):
        client = GPUModelClient(recipe_path, recipe, config, endpoint)
        fit = client.request('fit', x=x, score=scores, seed=1337)
        draw = client.request('sample', candidates=candidates, count=50, seed=23)
        assert np.isfinite(draw['draws']).all() and draw['draws'].shape == (50,5000)
        return dict(endpoint=endpoint, fit=fit, sample=draw['info'])
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        stress = list(pool.map(exercise,endpoints))
    write(OUTPUT/'validation/concurrency.json', dict(passed=True,
          elapsed_seconds=time.monotonic()-started, services=stress))
    # Use a SQLite backup of an active real task; no production state is edited.
    component = OUTPUT/'evaluation_resume'
    task_dir = component/'runs/pep_151'
    task_dir.mkdir(parents=True, exist_ok=True)
    for name in ('recipe.json','target_fit_amendment.json','preflight_passed.json'):
        shutil.copy2(CAMPAIGN/'fsbo'/name, component/name)
    write(component/'gpu_runtime.json',config)
    with sqlite3.connect('file:' + str(CAMPAIGN/'fsbo/runs/pep_151/runtime.sqlite') + '?mode=ro',uri=True) as source, \
            sqlite3.connect(str(task_dir/'runtime.sqlite')) as target:
        source.backup(target)
        before = StateUnpickler(io.BytesIO(target.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0])).load()
    python = '/root/data/cys/envs/bolt_clean_venv/bin/python'
    command = [python,str(TOOLS/'peptide_fsbo_gpu_eval.py'),'run-task',
        '--campaign',str(component),'--task','pep_151','--cpu','45',
        '--server','http://127.0.0.1:18871']
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    for phase in ('proposal','score','commit'):
        extra = ['--stop-after-events','1'] if phase == 'commit' else ['--stop-after-stage',phase]
        with (OUTPUT/('evaluation_'+phase+'.log')).open('ab') as stream:
            subprocess.run(command+extra, env=env,stdout=stream,stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        print(json.dumps(dict(validation_stage=phase,status='passed')),flush=True)
    after = json.loads((task_dir/'history.json').read_text())
    assert after['proposals'] == before['progress']['proposals']+50
    assert len(after['observations']) == after['initialization']+after['proposals']
    assert after['fresh_evaluations']-before['progress']['fresh_evaluations'] <= 50
    with sqlite3.connect('file:'+str(task_dir/'runtime.sqlite')+'?mode=ro',uri=True) as db:
        pending = db.execute("SELECT value FROM meta WHERE key='pending'").fetchone()
        assert pending is None or __import__('pickle').loads(pending[0]) is None
    report['checks'].update(bounded_concurrency=True,evaluation_resume=True,
                            pending_proposal_recovery=True,pending_score_recovery=True,
                            evaluation_accounting=True)
    report['evaluation_resume'] = dict(task='pep_151',before_proposals=before['progress']['proposals'],
        after_proposals=after['proposals'],fresh_evaluations=after['fresh_evaluations']-before['progress']['fresh_evaluations'])
    report['concurrency_report'] = dict(path=str(OUTPUT/'validation/concurrency.json'),
                                        sha256=digest(OUTPUT/'validation/concurrency.json'))
    report['validated_at'] = time.time()
    write(report_path,report)
    print(json.dumps(report,indent=2))


def restart_queue(side):
    status = CAMPAIGN/'workers'/('queue_'+side+'.json')
    launch_path = status.with_name('queue_'+side+'_launch.json')
    launch = json.loads(launch_path.read_text())
    runtime = json.loads((CAMPAIGN/'fsbo/gpu_runtime.json').read_text())
    if not runtime['enabled']:
        raise ValueError('GPU runtime is not active')
    old_pid = json.loads(status.read_text())['pid']
    if live(old_pid, 'queue_peptide_transfer.py'):
        os.kill(old_pid, signal.SIGTERM)
        deadline = time.monotonic()+45
        while live(old_pid, 'queue_peptide_transfer.py') and time.monotonic() < deadline:
            time.sleep(0.5)
        if live(old_pid, 'queue_peptide_transfer.py'):
            raise RuntimeError('Queue did not release its lease')
    archive = OUTPUT/('queue_%s_before_gpu.json' % side)
    if not archive.exists():
        write(archive, json.loads(status.read_text()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with Path(launch['log']).open('ab') as stream:
        child = subprocess.Popen(launch['command'], stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    launch.update(pid=child.pid, script_sha256=digest(TOOLS/'queue_peptide_transfer.py'),
                  started=time.time(), gpu_runtime=str(CAMPAIGN/'fsbo/gpu_runtime.json'))
    write(launch_path, launch)
    print(json.dumps(dict(side=side, old_queue=old_pid, new_queue=child.pid,
        transition='graceful current-batch commit and same-task resume')))


def tunnel():
    plan_path = OUTPUT/'tunnel.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    if plan and live(plan['pid'], 'ssh'):
        print(json.dumps(plan,indent=2))
        return
    services = json.loads((OUTPUT/'services.json').read_text())['services']
    command = ['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none',
        '-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=10',
        '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-N']
    for row in services:
        command += ['-R','127.0.0.1:%d:127.0.0.1:%d' % (row['port'],row['port'])]
    command += ['LLMOptimizer4']
    log = OUTPUT/'tunnel.log'
    with log.open('ab') as stream:
        child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    plan = dict(pid=child.pid,command=command,log=str(log),started=time.time())
    write(plan_path,plan)
    print(json.dumps(plan,indent=2))


def status():
    baseline_path = OUTPUT/'migration_baseline.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    result = dict(workers={},prefix_errors=[],gpu_fit_tasks=0,proposals=0,fresh_evaluations=0)
    for side in ('local','remote'):
        queue = json.loads((CAMPAIGN/'workers'/('queue_'+side+'.json')).read_text())
        active = [r for r in queue['active'] if r['method'] == 'fsbo']
        gpu = [r for r in active if str(TOOLS/'peptide_fsbo_gpu_eval.py') in r.get('command',[])]
        result['workers'][side] = dict(active=len(active),gpu=len(gpu),
            cpu_transition_remaining=len(active)-len(gpu),halted=queue.get('halted_cpus',[]))
        for worker in active:
            history = json.loads((CAMPAIGN/'fsbo/runs'/worker['task']/'history.json').read_text())
            result['proposals'] += history['proposals']
            result['fresh_evaluations'] += history['fresh_evaluations']
            fits = [r['fit'] for r in history['observations'] if r.get('fit')]
            result['gpu_fit_tasks'] += bool(fits and fits[-1].get('device') == 'cuda')
            before = baseline.get(worker['task'])
            if before:
                observed = [[r['sequence'],r['score'],r['cached'],r['feasible']]
                            for r in history['observations'][:before['rows']]]
                if hashlib.sha256(json.dumps(observed,separators=(',',':')).encode()).hexdigest() != before['prefix_sha256']:
                    result['prefix_errors'].append(worker['task'])
    result['checked_at'] = time.time()
    write(OUTPUT/'migration_status.json',result)
    print(json.dumps(result,indent=2))


def refresh_queue_gate():
    subprocess.run(['/root/miniconda3/bin/python','-m','unittest','discover',
        '-s',str(TOOLS.parent/'tests'),'-p','test_queue_peptide_transfer.py'],
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)
    report_path = OUTPUT/'validation/validation.json'
    report = json.loads(report_path.read_text())
    previous = report['code_sha256'][str(TOOLS/'queue_peptide_transfer.py')]
    backup = OUTPUT/'validation/before_shared_cpu_accounting.json'
    if not backup.exists():
        shutil.copy2(report_path,backup)
    report['code_sha256'][str(TOOLS/'queue_peptide_transfer.py')] = digest(TOOLS/'queue_peptide_transfer.py')
    report['queue_cpu_sharing_validation'] = dict(passed=True,tests=15,
        previous_queue_sha256=previous,checked_at=time.time(),
        change='GPU RPC services share coordinator CPU slots; all numerical GPU code unchanged')
    write(report_path,report)
    runtime_path = CAMPAIGN/'fsbo/gpu_runtime.json'
    config = json.loads(runtime_path.read_text())
    config.update(code_sha256=report['code_sha256'],
        validation=dict(path=str(report_path),sha256=digest(report_path)))
    write(runtime_path,config)
    print(json.dumps(dict(passed=True,queue_sha256=report['code_sha256'][str(TOOLS/'queue_peptide_transfer.py')],
        history_baseline_retained=True)))


def rebind_cpu():
    plan_path = OUTPUT/'services.json'
    plan = json.loads(plan_path.read_text())
    cpus = {18931:45,18932:4,18933:6,18934:7}
    for row in plan['services']:
        if not live(row['pid'],'peptide_fsbo_gpu_service.py'):
            raise ValueError('Owned GPU service is not alive')
        cpu = cpus[row['port']]
        row.setdefault('startup_cpu',row['cpu'])
        for tid in (Path('/proc')/str(row['pid'])/'task').iterdir():
            try:
                os.sched_setaffinity(int(tid.name),{cpu})
            except ProcessLookupError:
                pass
        row.update(cpu=cpu,rebound_at=time.time(),
            observed_affinity=sorted(os.sched_getaffinity(row['pid'])))
    write(plan_path,plan)
    print(json.dumps(plan,indent=2))


def migrate_uncommitted(task):
    from check_peptide_fsbo_gpu import StateUnpickler

    status_path = CAMPAIGN/'workers/queue_local.json'
    queue = json.loads(status_path.read_text())
    worker = next(r for r in queue['active'] if r['task'] == task and r['method'] == 'fsbo')
    parent = Path('/proc')/str(worker['pid'])
    if not live(worker['pid'],'peptide_transfer_eval.py'):
        raise ValueError('Only an existing CPU evaluator can be migrated this way')
    if (parent/'stat').read_text().rsplit(')',1)[1].split()[19] != worker['process_start']:
        raise ValueError('CPU evaluator process identity changed')
    if (parent/'wchan').read_text().strip() != 'pipe_read':
        raise ValueError('CPU evaluator is not blocked on its model worker')
    models = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[1]) == worker['pid'] and b'peptide_transfer_models.py' in (path/'cmdline').read_bytes():
                models.append(int(path.name))
        except FileNotFoundError:
            pass
    if len(models) != 1:
        raise ValueError('Expected exactly one owned CPU model worker')
    controller = queue['pid']
    if live(controller,'queue_peptide_transfer.py'):
        os.kill(controller,signal.SIGTERM)
        deadline = time.monotonic()+45
        while live(controller,'queue_peptide_transfer.py') and time.monotonic()<deadline:
            time.sleep(0.5)
        if live(controller,'queue_peptide_transfer.py'):
            raise RuntimeError('Queue lease has not been released')
    frozen = []
    try:
        for pid in (models[0],worker['pid']):
            try:
                os.kill(pid,signal.SIGSTOP)
            except ProcessLookupError:
                print(json.dumps(dict(task=task,status='CPU request completed naturally before interruption')))
                return
            frozen.append(pid)
        with sqlite3.connect('file:'+str(CAMPAIGN/'fsbo/runs'/task/'runtime.sqlite')+'?mode=ro',uri=True,timeout=5) as db:
            pending = db.execute("SELECT value FROM meta WHERE key='pending'").fetchone()
            if pending is not None and __import__('pickle').loads(pending[0]) is not None:
                raise ValueError('A proposed/evaluated pending batch exists; retain graceful migration')
            payload = db.execute("SELECT value FROM meta WHERE key='state'").fetchone()[0]
        saved = StateUnpickler(io.BytesIO(payload)).load()
        record = dict(task=task,cpu=worker['cpu'],coordinator_pid=worker['pid'],
            model_pid=models[0],proposals=saved['progress']['proposals'],
            fresh_evaluations=saved['progress']['fresh_evaluations'],
            state_payload_sha256=hashlib.sha256(payload).hexdigest(),
            pending_batch=None,discarded_evaluations=0,created_at=time.time(),
            reason='Cancel only an uncommitted CPU model request; resume the unchanged task checkpoint on GPU')
        write(OUTPUT/('uncommitted_migration_'+task+'.json'),record)
        # Both processes are frozen, and the evaluator has not launched a batch.
        for pid in frozen:
            os.kill(pid,signal.SIGKILL)
        frozen = []
        queue = json.loads(status_path.read_text())
        queue['active'] = [r for r in queue['active'] if r['pid'] != worker['pid']]
        queue.setdefault('reroute_resume',{})[str(worker['cpu'])] = dict(method='fsbo',task=task)
        write(status_path,queue)
    finally:
        for pid in frozen:
            try:
                os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:
                pass
        restart_queue('local')
    print(json.dumps(record,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'validate-evaluation', 'tunnel', 'activate', 'restart-queue', 'status', 'refresh-queue-gate', 'rebind-cpu', 'migrate-uncommitted'))
    parser.add_argument('--count', type=int, choices=range(1,5), default=1)
    parser.add_argument('--side', choices=('local','remote'), default='local')
    parser.add_argument('--task')
    args = parser.parse_args()
    if args.action == 'start':
        start(args.count)
    elif args.action == 'activate':
        activate()
    elif args.action == 'validate-evaluation':
        validate_evaluation()
    elif args.action == 'tunnel':
        tunnel()
    elif args.action == 'status':
        status()
    elif args.action == 'refresh-queue-gate':
        refresh_queue_gate()
    elif args.action == 'rebind-cpu':
        rebind_cpu()
    elif args.action == 'migrate-uncommitted':
        if not args.task:
            parser.error('--task is required')
        migrate_uncommitted(args.task)
    else:
        restart_queue(args.side)


if __name__ == '__main__':
    main()
