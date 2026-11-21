"""Finite prerequisite queue and resource owner for the peptide BORA campaign."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
from common import ROOT, alive, atomic, config, proc_identity, read, sha, status, verify

ORACLE = '/root/data/cys/envs/bolt_clean_venv/bin/python'
INFER = '/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'


def predecessor_gate(cfg, process_check=True):
    predecessor = Path(cfg['predecessor'])
    state = read(predecessor/'status.json')
    if state.get('status') != 'complete':
        return False, dict(reason='waiting_for_semanticopt_completion', predecessor_status=state.get('status'),
            completed=state.get('completed'), evaluations=state.get('evaluations'))
    summary = read(predecessor/'summary.json')
    if summary.get('status') != 'complete' or not summary.get('audits_passed'):
        return False, dict(reason='waiting_for_semanticopt_audit')
    tasks = read(predecessor/'stop_policy.json')['tasks']
    for task in tasks:
        st = read(predecessor/'peptide/runs'/task/'status.json')
        if st.get('status') != 'complete' or st.get('evaluations') != 4000:
            raise RuntimeError('Predecessor completion evidence inconsistent: '+task)
        db = sqlite3.connect('file:'+str(predecessor/'peptide/runs'/task/'observations.sqlite')+'?mode=ro', uri=True)
        count, low, high = db.execute('select count(*),min(n),max(n) from records').fetchone()
        pending = db.execute("select value from state where key='pending'").fetchone()
        db.close()
        if (count,low,high) != (4000,0,3999):
            raise RuntimeError('Predecessor evaluation ledger incomplete: '+task)
    if process_check:
        active = []
        paths = {str(predecessor/'control/controller.py'), str(predecessor/'services/qwen_inference_server.py'),
                 str(predecessor/'code/evaluate.py')}
        for p in Path('/proc').glob('[0-9]*'):
            identity = proc_identity(int(p.name))
            if identity and paths.intersection(identity['argv']):
                active.append(identity['pid'])
        if active:
            return False, dict(reason='waiting_for_semanticopt_processes_to_exit', pids=active)
    return True, dict(reason='semanticopt_completed_audited_and_drained', tasks=len(tasks), evaluations=40000)


def resource_gate(cfg):
    if shutil.disk_usage(ROOT).free < cfg['resource_min_free_gib']*2**30:
        return False, dict(reason='insufficient_free_storage')
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0])*1024
    cg = Path('/sys/fs/cgroup')
    limit = (cg/'memory.max').read_text().strip()
    if limit != 'max':
        available = min(available, int(limit)-int((cg/'memory.current').read_text()))
    if available < (cfg['memory_reserve_gib']+16)*2**30:
        return False, dict(reason='insufficient_available_RAM', available_bytes=available)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    free = {int(row.split(',')[0]):int(row.split(',')[1]) for row in gpu.splitlines()}
    # Previously validated two-A16 layout; retain at least 7.5 GiB on each card.
    assigned = [g for pair in cfg['gpu_pairs'] for g in pair]
    if any(free[g] < 7680 for g in assigned):
        return False, dict(reason='waiting_for_assigned_GPU_memory', free_mib={g:free[g] for g in assigned})
    quota,period = (cg/'cpu.max').read_text().split()
    cpus = len(os.sched_getaffinity(0)) if quota == 'max' else int(quota)/int(period)
    if cpus < 14 or not set(cfg['worker_cpus']+cfg['inference_cpus']) <= os.sched_getaffinity(0):
        return False, dict(reason='configured_CPU_allocation_unavailable', quota=cpus)
    return True, dict(reason='resources_available', memory_available_gib=available/2**30,
        gpu_free_mib={g:free[g] for g in assigned}, cpu_quota=cpus)


def health(port):
    import urllib.request
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f'http://127.0.0.1:{port}/health',timeout=10) as r:
            return json.load(r)
    except Exception:
        return None


def spawn(command, label, env):
    stream = open(ROOT/'logs'/(label+'.log'), 'ab', buffering=0)
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
        env=env, start_new_session=True)
    stream.close()
    return process


def campaign_status(active=None):
    cfg = config()
    states = {}
    for task in cfg['tasks']:
        p = ROOT/'runs'/task/'status.json'
        if p.exists():
            states[task] = read(p)
    complete = sum((ROOT/'runs'/t/'result.json').exists() and
        read(ROOT/'runs'/t/'result.json').get('status') == 'complete' for t in cfg['tasks'])
    failed = [t for t,s in states.items() if s.get('status') == 'failed']
    return dict(tasks=200, tasks_started=len(states), tasks_complete=complete,
        evaluations=sum(s.get('evaluations',0) for s in states.values()), target=800000,
        failed_tasks=failed, live_workers=len(active or {}),
        workers={str(k):dict(task=v['task'],pid=v['process'].pid) for k,v in (active or {}).items()})


def run_campaign(cfg, resume=False):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_QWEN_DEVICE_MAP='auto', PYTHONDONTWRITEBYTECODE='1')
    services, workers = [], {}
    validation = None
    try:
        for i,pair in enumerate(cfg['gpu_pairs']):
            services.append(spawn(['taskset','-c',str(cfg['inference_cpus'][i]),INFER,'-u',
                str(ROOT/'services/qwen_inference_server.py'),'replica','--config',str(ROOT/f'services/replica{i:02}.json')],
                f'replica{i:02}', dict(env,CUDA_VISIBLE_DEVICES=','.join(map(str,pair)))))
        status(ROOT/'services/processes.json', services=[proc_identity(p.pid) for p in services])
        deadline = time.monotonic()+1200
        while not all((health(p) or {}).get('ready') for p in cfg['replica_ports']):
            if any(p.poll() is not None for p in services):
                raise RuntimeError('BORA inference replica failed to load; inspect logs')
            if time.monotonic() > deadline:
                raise TimeoutError('BORA inference loading timeout')
            status(ROOT/'queue_status.json', status='starting', stage='loading_untuned_Qwen')
            time.sleep(5)
        services.append(spawn([INFER,'-u',str(ROOT/'services/qwen_inference_server.py'),'broker',
            '--config',str(ROOT/'services/broker.json')],'broker',dict(env,CUDA_VISIBLE_DEVICES='')))
        deadline = time.monotonic()+120
        while not (health(cfg['broker_port']) or {}).get('ready'):
            if services[-1].poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('BORA broker readiness failed')
            time.sleep(2)
        status(ROOT/'services/processes.json', services=[proc_identity(p.pid) for p in services])
        status(ROOT/'evidence/inference_health.json', health=health(cfg['broker_port']))
        # Real-model validation covers assistant suggestions and exact candidate selection.
        if not resume:
            status(ROOT/'queue_status.json', status='validating', stage='live_assistant_and_full_worker')
            validation = spawn(['taskset','-c',str(cfg['worker_cpus'][0]),ORACLE,'-u',
                str(ROOT/'code/validate.py'),'--live'],'validation_live',dict(env,CUDA_VISIBLE_DEVICES=''))
            while validation.poll() is None:
                time.sleep(5)
            if validation.returncode or not read(ROOT/'evidence/live_validation.json').get('passed'):
                raise RuntimeError('BORA live validation failed; campaign remains stopped')
        atomic(ROOT/'evidence/launch_gate.json',dict(at=time.time(),live_validation_passed=True,
            predecessor=predecessor_gate(cfg)[1],resources=resource_gate(cfg)[1]))
        tasks = [t for t in cfg['tasks'] if not (ROOT/'runs'/t/'result.json').exists()]
        if resume:
            tasks.sort(key=lambda t: read(ROOT/'runs'/t/'status.json').get('status') != 'failed'
                if (ROOT/'runs'/t/'status.json').exists() else True)
        canary_done = False
        initial_evaluations = campaign_status()['evaluations'] if resume else 0
        while tasks or workers:
            for slot, job in list(workers.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                result = ROOT/'runs'/job['task']/'result.json'
                if code or not result.exists() or read(result).get('status') != 'complete':
                    raise RuntimeError(f'Worker {job["task"]} exited {code}; saved observations retained')
                del workers[slot]
            current = campaign_status(workers)
            successes = current['evaluations']
            if successes-initial_evaluations >= cfg['scale_after_successes']:
                canary_done = True
            maximum = cfg['max_workers'] if canary_done else cfg['initial_workers']
            for slot in range(maximum):
                if slot in workers or not tasks:
                    continue
                okay, info = resource_gate(cfg)
                # GPU weights are now loaded; admission checks only RAM and storage.
                if info.get('reason') not in ['resources_available','waiting_for_assigned_GPU_memory']:
                    status(ROOT/'admission_status.json', status='waiting', **info)
                    break
                task = tasks.pop(0)
                cpu = cfg['worker_cpus'][slot]
                process = spawn(['taskset','-c',str(cpu),ORACLE,'-u',str(ROOT/'code/worker.py'),
                    '--task',task],'worker_'+task,dict(env,CUDA_VISIBLE_DEVICES=''))
                workers[slot] = dict(task=task,process=process)
            if any(p.poll() is not None for p in services):
                raise RuntimeError('BORA inference service exited')
            status(ROOT/'status.json', status='running', **campaign_status(workers))
            status(ROOT/'queue_status.json', status='running', stage='BORA_test200', **campaign_status(workers))
            time.sleep(10)
        from report import report
        report()
        status(ROOT/'status.json', status='complete', **campaign_status())
        status(ROOT/'queue_status.json', status='complete', stage='reported', **campaign_status())
    finally:
        # Only terminate children created by this queue; never use broad process matching.
        owned = [j['process'] for j in workers.values()]+services+([validation] if validation is not None else [])
        for process in owned:
            if process.poll() is None:
                process.terminate()
        for process in owned:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--check-gate', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(dict(queue=read(ROOT/'queue_status.json'), campaign=campaign_status()),indent=2))
        return
    cfg = verify()
    if args.resume:
        recovery = read(ROOT/'evidence/recovery_ready.json')
        if (not recovery.get('passed') or recovery.get('identity') != cfg['identity']
                or not read(ROOT/'evidence/live_validation.json').get('passed')
                or not recovery.get('memory_regression_passed')):
            raise RuntimeError('Resume requires checkpoint audit and inference memory validation')
        existing = {str(p) for p in (ROOT/'runs').glob('*/checkpoint.pkl')}
        if existing != set(recovery['checkpoint_sha256']):
            raise RuntimeError('Recovery audit does not cover every existing checkpoint')
        for path, expected in recovery['checkpoint_sha256'].items():
            if sha(path) != expected:
                raise RuntimeError('Checkpoint changed after recovery audit: '+path)
    if args.check_gate:
        print(json.dumps(dict(predecessor=predecessor_gate(cfg),resources=resource_gate(cfg)),indent=2))
        return
    guard = open(ROOT/'services/queue.lock','a')
    fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not read(ROOT/'evidence/offline_validation.json').get('passed'):
        raise RuntimeError('Offline validation has not passed')
    def terminate(signum, frame):
        raise SystemExit(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    atomic(ROOT/'services/queue_process.json',dict(**proc_identity(os.getpid()),host='LLMOptimizer3',started_at=time.time()))
    try:
        while True:
            if (ROOT/'STOP').exists():
                status(ROOT/'queue_status.json',status='stopped',reason='STOP file')
                return
            ready, predecessor = predecessor_gate(cfg)
            if ready:
                resources_ready, resources = resource_gate(cfg)
                if resources_ready:
                    break
            else:
                resources = None
            status(ROOT/'queue_status.json', status='waiting', predecessor=predecessor,
                resources=resources, tasks=200, fresh_budget=4000, target=800000)
            time.sleep(cfg['queue_poll_seconds'])
        run_campaign(cfg, resume=args.resume)
    except BaseException:
        status(ROOT/'queue_status.json', status='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
