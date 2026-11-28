#!/usr/bin/env python3
"""Failure-isolated supervisors; experiment recipes and evaluator code stay pinned."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

OPS = Path(__file__).resolve().parent
BASE = OPS.parent
ROOTS = {'semanticopt': BASE/'semanticopt_query_400_pilot800_20261123',
         'bolt': BASE/'query_bolt30_lolbo_total800_20261123'}
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
       '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=3',
       '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3']


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps({**value, 'at':time.time()}, indent=2)+'\n')
    temp.replace(path)


def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def configuration(kind):
    root = ROOTS[kind]
    if kind == 'bolt':
        cfg = load(root/'recipe.json')
        return cfg['workers'] + [load(root/'control/extra_worker.json')], cfg['tasks'], {}
    cfg = load(root/'allocation.json')
    ext = load(root/'extension.json')
    other = [t for t in ext['tasks'] if t not in ext['pilot_tasks']]
    source = Path(ext['source'])
    other.sort(key=lambda t: load(source/'query/runs'/t/'status.json', {}).get('elapsed_seconds', 0), reverse=True)
    return cfg['workers'], ext['pilot_tasks'] + other, ext


def directory(kind, task):
    return ROOTS[kind]/('query/runs' if kind == 'semanticopt' else 'runs')/task


def completed(kind, task, ext):
    folder = directory(kind, task)
    if kind == 'bolt':
        return (folder/'complete.json').exists()
    state = load(folder/'status.json', {})
    budget = ext['pilot_budget'] if task in ext['pilot_tasks'] else ext['default_budget']
    return state.get('status') == 'complete' and state.get('evaluations') == budget


def statusfile(kind, number):
    return ROOTS[kind]/('query' if kind == 'semanticopt' else 'services')/f'worker{number:02}.status.json'


def database_check(spec):
    # Called on the worker host. An unhealthy DB pauses this slot only.
    import psycopg2
    with psycopg2.connect(host='127.0.0.1', port=spec['db_port'], dbname='imdb',
                          user='imdb', password='imdb', connect_timeout=5) as connection:
        with connection.cursor() as cursor:
            cursor.execute('SHOW data_directory')
            actual = cursor.fetchone()[0]
            if Path(actual).resolve() != Path(spec['pgdata']).resolve():
                raise RuntimeError(f'Database mismatch: {actual}')
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND state='active'")
            active = cursor.fetchone()[0]
            if active:
                raise RuntimeError(f'Database has {active} other active sessions; needs inspection')
            settings = {}
            for name in ['shared_buffers','work_mem','max_parallel_workers_per_gather','temp_file_limit','statement_timeout']:
                cursor.execute('SHOW '+name)
                settings[name] = cursor.fetchone()[0]
            return dict(data_directory=actual, active_sessions=active, settings=settings)


def environment(kind, spec):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    if kind == 'bolt':
        env.update(DB_HOST='127.0.0.1', DB_PORT=str(spec['db_port']), DB_NAME='imdb',
                   DB_USER='imdb', DB_PASSWORD='imdb', QUERY_LOLBO_VAE_OFFSET=str(spec['slot']))
    return env


def task_command(kind, spec, task):
    root = ROOTS[kind]
    prefix = ['taskset', '-c', str(spec['optimizer_cpu']), PYTHON, '-u']
    if kind == 'semanticopt':
        return prefix + [str(root/'code/evaluate.py'), 'query', task, '--port', str(spec['db_port'])]
    # A new interpreter per query avoids PyTorch's second set_num_interop_threads call.
    return prefix + [str(OPS/'fleet.py'), 'task', 'bolt', '--task', task]


def quarantine(kind, task, error):
    atomic(ROOTS[kind]/'task_pauses'/f'{task}.json', error)
    atomic(directory(kind, task)/'error.json', error)


def slot(kind, number):
    root = ROOTS[kind]
    specs, tasks, ext = configuration(kind)
    spec = next(s for s in specs if s['slot'] == number)
    own = lock(root/('leases' if kind == 'semanticopt' else 'services')/f'slot{number}.lock')
    statepath = statusfile(kind, number)
    env = environment(kind, spec)
    slotpause = root/'slot_pauses'/f'worker{number:02}.PAUSE'
    try:
        while not (root/'PAUSE').exists():
            if slotpause.exists():
                atomic(statepath, dict(status='paused_slot', slot=number, host=spec['host']))
                time.sleep(5)
                continue
            picked = None
            for task in tasks:
                if (root/'task_pauses'/f'{task}.json').exists() or completed(kind, task, ext):
                    continue
                folder = directory(kind, task)
                folder.mkdir(parents=True, exist_ok=True)
                leasepath = root/'leases'/f'{task}.lock' if kind == 'semanticopt' else folder/'task.lock'
                try:
                    lease = lock(leasepath)
                except BlockingIOError:
                    continue
                if (root/'task_pauses'/f'{task}.json').exists() or completed(kind, task, ext):
                    lease.close()
                    continue
                picked = task, lease
                break
            if picked is None:
                if all(completed(kind, t, ext) or (root/'task_pauses'/f'{t}.json').exists() for t in tasks):
                    atomic(statepath, dict(status='idle', slot=number, host=spec['host']))
                    return
                atomic(statepath, dict(status='waiting_for_tasks', slot=number, host=spec['host']))
                time.sleep(5)
                continue
            task, lease = picked
            try:
                database_check(spec)
                logpath = root/'logs'/f'{task}.isolated.log'
                with logpath.open('ab', buffering=0) as log:
                    child = subprocess.Popen(task_command(kind, spec, task), env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    atomic(statepath, dict(status='running', slot=number, host=spec['host'], task=task,
                        port=spec['db_port'], pid=os.getpid(), child_pid=child.pid, log=str(logpath)))
                    code = child.wait()
                if completed(kind, task, ext):
                    continue
                if (root/'PAUSE').exists():
                    atomic(statepath, dict(status='paused', slot=number, task=task, exit_code=code))
                    return
                error = dict(status='task_paused', slot=number, host=spec['host'], task=task,
                             exit_code=code, log=str(logpath), reason='Task exited before its target; inspect journal before retry')
                quarantine(kind, task, error)
                atomic(statepath, error)
                # Keep this DB slot productive on another task if its database is healthy.
                database_check(spec)
            finally:
                lease.close()
        atomic(statepath, dict(status='paused', slot=number, host=spec['host']))
    except BaseException as exc:
        # No global PAUSE mutation, and no unsafe replay of an interrupted measurement.
        error = dict(status='failed_slot', slot=number, host=spec['host'], error=repr(exc), traceback=traceback.format_exc())
        atomic(statepath, error)
        raise
    finally:
        own.close()


def summarize(kind, tasks, ext, children, started):
    root = ROOTS[kind]
    complete = sum(completed(kind, t, ext) for t in tasks)
    failures = [dict(slot=s['slot'], host=s['host'], exit_code=p.returncode)
                for s, p in children if p.poll() not in (None, 0)]
    pauses = [p.stem for p in sorted((root/'task_pauses').glob('*.json'))]
    slotstates = [load(statusfile(kind, s['slot']), {}) for s, _ in children]
    degraded = bool(failures or pauses or any(s.get('status') in ('paused_slot','failed_slot') for s in slotstates))
    state = 'complete' if complete == len(tasks) else 'paused' if (root/'PAUSE').exists() else 'degraded' if degraded else 'running'
    if kind == 'bolt':
        count = sum(load(directory(kind, t)/'progress.json', {}).get('rows', 30) for t in tasks)
        extra = dict(warm_evaluations=6000, new_evaluations=count-6000, new_evaluation_target=154000, vae_replicas=8)
    else:
        count = sum(load(directory(kind, t)/'status.json', {}).get('evaluations', 200) for t in tasks)
        extra = dict(retained_evaluations=40000, new_evaluations=count-40000, new_evaluation_target=44000, llm_replicas=20)
    return dict(status=state, completed_tasks=complete, total_tasks=len(tasks), evaluations=count,
        live_worker_slots=sum(p.poll() is None for _, p in children), failures=failures,
        paused_tasks=pauses, elapsed_seconds=time.time()-started, supervisor=str(OPS/'fleet.py'), **extra)


def process_ticks(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


class AdoptedProcess:
    def __init__(self, record, kind):
        self.pid = record['pid']
        self.ticks = record['start_ticks']
        self.kind, self.number = kind, record['slot']
        self.returncode = None

    def poll(self):
        if process_ticks(self.pid) == self.ticks:
            return None
        state = load(statusfile(self.kind, self.number), {}).get('status')
        self.returncode = 0 if state in ('idle', 'paused') else 1
        return self.returncode


def controller(kind):
    root = ROOTS[kind]
    own = lock(root/'control/controller.lock')
    specs, tasks, ext = configuration(kind)
    children = []
    previous = {r['slot']: r for r in load(root/'control/workers.process.json', {}).get('workers', [])}
    started = time.time()
    for spec in specs:
        prior = previous.get(spec['slot'], {})
        if prior.get('start_ticks') and process_ticks(prior['pid']) == prior['start_ticks']:
            children.append((spec, AdoptedProcess(prior, kind)))
            continue
        cmd = [PYTHON, '-u', str(OPS/'fleet.py'), 'slot', kind, '--slot', str(spec['slot'])]
        if spec['host'] != 'LLMOptimizer2':
            if spec['host'] == 'LLMOptimizer4':
                # Reuse established masters to avoid the host's saturated SSH authentication queue.
                mux = '/tmp/query-fleet-semanticopt-LLMOptimizer4-1' if kind == 'semanticopt' and spec['slot'] < 16 else '/tmp/.ssh-mux-root@ssh-cn-huabei1.ebcloud.com:34979'
            else:
                mux = f'/tmp/query-fleet-{kind}-{spec["host"]}-{spec["slot"]//6}'
            cmd = SSH + ['-o', 'ControlPath='+mux, spec['host'], shlex.join(cmd)]
        with (root/'logs'/f'slot{spec["slot"]:02}.isolated.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((spec, process))
        time.sleep(0.5)  # Avoid simultaneous SSH authentication bursts.
    atomic(root/'control/workers.process.json', dict(supervisor=str(OPS/'fleet.py'),
        workers=[dict(slot=s['slot'], host=s['host'], pid=p.pid, start_ticks=process_ticks(p.pid)) for s, p in children]))
    while True:
        status = summarize(kind, tasks, ext, children, started)
        atomic(root/'status.json', status)
        if status['status'] == 'complete':
            return
        if (root/'PAUSE').exists() and not any(p.poll() is None for _, p in children):
            return
        # A failed or paused slot never terminates the controller or other children.
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['controller','slot','task','preflight'])
    parser.add_argument('kind', choices=list(ROOTS))
    parser.add_argument('--slot', type=int)
    parser.add_argument('--task')
    args = parser.parse_args()
    if args.command == 'controller':
        controller(args.kind)
    elif args.command == 'slot':
        slot(args.kind, args.slot)
    elif args.command == 'task':
        if args.kind != 'bolt':
            raise ValueError('Only BOLT uses the task wrapper')
        sys.path.insert(0, str(ROOTS['bolt']))
        import run_lolbo
        run_lolbo.run_task(args.task)
    else:
        specs, _, _ = configuration(args.kind)
        print(json.dumps([dict(slot=s['slot'], **database_check(s)) for s in specs
                          if args.slot is None or s['slot'] == args.slot], indent=2))


if __name__ == '__main__':
    main()
