#!/usr/bin/env python3
"""Reuse released TPE slots for validated FSBO, then RGPE test200 tasks."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
TRAINING = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
CAMPAIGN = Path('/root/data/logs/peptide_transfer_test200_20261028_v1')
COMPONENT = Path('/root/data/logs/peptide_transfer_rgpe_full800_check_20261028')
REPORTS = Path('/root/reports/peptide_transfer_rgpe_full800_check_20261028')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
METHODS = ('fsbo', 'rgpe')
RESERVATION = dict(fsbo=6.0, rgpe=8.0, validation=8.0)


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        words = [w.decode() for w in (root / 'cmdline').read_bytes().split(b'\0') if w]
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if not words or stat[0] == 'Z':
            return None
        return dict(pid=pid, words=words, start=stat[19])
    except (OSError, ValueError, IndexError):
        return None


def process_start(pid):
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def option(row, name, default=None):
    words = row['words']
    return words[words.index(name) + 1] if name in words else default


def processes():
    return [row for p in Path('/proc').iterdir() if p.name.isdigit()
            for row in [process(int(p.name))] if row is not None]


def tpe_workers(rows):
    return [r for r in rows if str(ROOT / 'peptide_tpe_runtime.py') in r['words']
            and 'worker' in r['words'] and option(r, '--campaign') == str(BASELINE / 'tpe')]


def occupied_cpus(rows):
    occupied = set()
    markers = ('peptide_', '/pg16/bin/postgres', 'postgres:', 'manage.py',
               'bolt_taskset_top10_full_sft/', 'train_semanticopt')
    for row in rows:
        if row['pid'] == os.getpid():
            continue
        text = ' '.join(row['words'])
        if not any(marker in text for marker in markers):
            continue
        # GPU services intentionally share coordinator cores while RPC waits.
        if str(ROOT / 'peptide_fsbo_gpu_service.py') in row['words']:
            continue
        cpu = option(row, '--cpu')
        if cpu is not None:
            occupied.add(int(cpu))
        else:
            try:
                affinity = os.sched_getaffinity(row['pid'])
                if len(affinity) < os.cpu_count():
                    occupied.update(affinity)
            except ProcessLookupError:
                pass
    return occupied


def completed(method):
    directory = CAMPAIGN / method / 'runs'
    return {entry.name[:-7] for entry in os.scandir(directory)
            if entry.name.endswith('.tar.gz')} if directory.exists() else set()


def slot_failed(side, cpu, index):
    return ((CAMPAIGN / 'workers' / ('error_%s_cpu%d.json' % (side, cpu))).exists()
            or (BASELINE / 'tpe' / 'worker_errors' / ('%d.json' % index)).exists())


def configured_server(side, cpu, index, servers):
    routes = read(CAMPAIGN / 'workers' / 'vae_routes.json', {})
    server = routes.get(side, {}).get(str(cpu), servers[index % len(servers)])
    address = urllib.parse.urlsplit(server)
    if (address.scheme != 'http' or address.hostname != '127.0.0.1'
            or not address.port or address.path or address.query or address.fragment):
        raise ValueError('VAE routes must use loopback HTTP service ports')
    return server


def task_runner(method):
    if method == 'fsbo':
        runtime_path = CAMPAIGN / method / 'gpu_runtime.json'
        config = read(runtime_path, {})
        if config.get('enabled'):
            from peptide_fsbo_gpu_eval import runtime
            runtime(runtime_path)
            return ROOT / 'peptide_fsbo_gpu_eval.py'
    return ROOT / 'peptide_transfer_eval.py'


def request_reroute(row, server):
    if row['method'] == 'validation' or row.get('reroute_to'):
        return
    current = process(row['pid'])
    if not current or current['start'] != row['process_start']:
        return
    runner = task_runner(row['method'])
    if row['server'] == server and str(runner) in current['words']:
        return
    # The evaluator handles SIGTERM after committing its current batch.
    try:
        os.kill(row['pid'], signal.SIGTERM)
    except ProcessLookupError:
        return
    row.update(reroute_to=server, reroute_runner=str(runner), reroute_requested_at=time.time())


def reroute_checkpoint(row, result):
    if not row.get('reroute_to') or result not in (None, 0):
        return False
    directory = CAMPAIGN / row['method'] / 'runs' / row['task']
    progress = directory / 'progress.json'
    value = read(progress, {})
    recipe = CAMPAIGN / row['method'] / 'recipe.json'
    try:
        return (value.get('identity', {}).get('task') == row['task']
                and value.get('identity', {}).get('recipe_sha256') == digest(recipe)
                and progress.stat().st_mtime >= row['reroute_requested_at']) if value else False
    except FileNotFoundError:
        return False


def ready(method):
    directory = CAMPAIGN / method
    if (directory / 'paused.json').exists():
        return False
    gate = read(directory / 'preflight_passed.json')
    if not gate:
        return False
    report = Path(gate['report'])
    if digest(directory / 'recipe.json') != gate['recipe_sha256'] or digest(report) != gate['report_sha256']:
        raise ValueError('Recipe-bound launch gate changed: ' + method)
    result = read(report)
    return bool(result['passed'] and result['production_scale_passed'])


@contextlib.contextmanager
def claim(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        yield None
        return
    try:
        yield stream
    finally:
        stream.close()


def memory_available(active):
    root = Path('/sys/fs/cgroup')
    maximum = (root / 'memory.max').read_text().strip()
    if maximum == 'max':
        raise ValueError('This allocation requires a finite container memory limit')
    stats = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
    available = (int(maximum) - int((root / 'memory.current').read_text())
                 + int(stats.get('inactive_file', 0))) / 2**30
    growth = 0.0
    rows = processes()
    for row in active:
        rss = 0.0
        pids = [row['pid']]
        for model in rows:
            if option(model, '--recipe') != str(CAMPAIGN / row['method'] / 'recipe.json'):
                continue
            try:
                if row['cpu'] in os.sched_getaffinity(model['pid']):
                    pids.append(model['pid'])
            except ProcessLookupError:
                pass
        for pid in pids:
            try:
                status = (Path('/proc') / str(pid) / 'status').read_text().splitlines()
                rss += int(next(s.split()[1] for s in status if s.startswith('VmRSS:'))) / 2**20
            except (OSError, StopIteration):
                pass
        growth += max(0.0, RESERVATION[row['method']] - rss)
    # Preserve the source-fitting workers' reserved growth on machine4.
    fitting = read(TRAINING / 'remote_queue.json', {})
    live = {r['pid'] for r in rows if str(ROOT / 'peptide_rgpe_fit.py') in r['words']}
    if any(row['pid'] in live for row in fitting.get('rgpe_active', {}).values()):
        growth += fitting.get('memory_growth_reserved_gib', 0.0)
    return available - growth - 16.0


def validate_rgpe(cpu, server):
    os.sched_setaffinity(0, {cpu})
    commands = [
        [PYTHON, str(ROOT / 'peptide_transfer_eval.py'), 'prepare', '--campaign', str(CAMPAIGN), '--methods', 'rgpe'],
        [PYTHON, str(ROOT / 'peptide_transfer_eval.py'), 'prepare', '--campaign', str(COMPONENT),
         '--methods', 'rgpe', '--component', '--source-limit', '800', '--rank-samples', '256'],
        [PYTHON, str(ROOT / 'check_peptide_transfer_eval.py'), '--campaign', str(COMPONENT),
         '--output', str(REPORTS), '--methods', 'rgpe', '--cpu', str(cpu), '--server', server],
    ]
    for command in commands:
        subprocess.run(command, env=ENV, check=True)
    import copy
    import numpy as np
    import peptide_transfer_eval as adapter
    from check_peptide_transfer_eval import archived_state, production_pool_check
    directory = CAMPAIGN / 'rgpe'
    recipe = adapter.verify(directory)
    component = adapter.verify(COMPONENT / 'rgpe')
    report_path = REPORTS / 'rgpe_report.json'
    report = read(report_path)
    if (not report['passed'] or not report['production_scale_passed'] or report['source_models'] != 800
            or component['code'] != recipe['code'] or component['corpus_sha256'] != recipe['corpus_sha256']
            or report['recipe_sha256'] != digest(COMPONENT / 'rgpe' / 'recipe.json')
            or component['rgpe_model_hashes'] != recipe['rgpe_model_hashes']
            or component['rgpe_rank_samples'] != recipe['rgpe_rank_samples']):
        raise ValueError('Complete-bank component check does not match production')
    sys.path.insert(0, recipe['cpu_upstream'])
    state = copy.deepcopy(archived_state(COMPONENT / 'rgpe', report['task']))
    state['identity'] = dict(format=adapter.FORMAT, recipe_sha256=digest(directory / 'recipe.json'), task=report['task'])
    model = adapter.ModelClient(directory / 'recipe.json', recipe)
    try:
        adapter.check_memory(recipe, len(state['rows']))
        fit = model.request('fit', x=adapter.normalized(state['features'], recipe),
                            score=np.asarray([r['score'] for r in state['rows']]), seed=recipe['seed'] + 2)
        state['model_snapshot'] = model.request('snapshot')
    finally:
        model.close()
    pool = production_pool_check(directory, report['task'], recipe, state, server)
    result = dict(report, recipe_sha256=state['identity']['recipe_sha256'],
                  component_report=str(report_path), component_report_sha256=digest(report_path),
                  production_target_fit=fit, production_pool_check=pool, launched=False,
                  gate_checker_sha256=digest(Path(__file__)))
    path = directory / 'preflight_report.json'
    write(path, result)
    write(directory / 'preflight_passed.json', dict(recipe_sha256=result['recipe_sha256'],
          report=str(path), report_sha256=digest(path)))
    adapter.preflight(directory)


def recover_inference(side):
    recovered = []
    for path in (CAMPAIGN / 'workers').glob('error_' + side + '_cpu*.json'):
        error = read(path)
        worker = error.get('worker', {})
        if worker.get('method') not in METHODS or not worker.get('log'):
            continue
        log = Path(worker['log']).read_text(errors='replace')
        if 'ConnectionRefusedError' not in log or 'GPU service unavailable after retries' not in log:
            continue
        with urllib.request.urlopen(worker['server'] + '/health', timeout=10) as response:
            health = json.load(response)
        if health.get('errors', 0):
            raise RuntimeError('Inference service still reports errors')
        archive = path.parent / 'recovered_errors' / (path.stem + '_%d.json' % time.time_ns())
        archive.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, archive)
        recovered.append(dict(cpu=worker['cpu'], task=worker['task'], evidence=str(archive)))
    print(json.dumps(dict(side=side, recovered=recovered, results_preserved=True)), flush=True)


def dispatch(args):
    status_path = CAMPAIGN / 'workers' / ('queue_' + args.side + '.json')
    old = read(status_path, {})
    active = old.get('active', [])
    resume = old.get('reroute_resume', {})
    recorded = {row['pid'] for row in active}
    for row in processes():
        directory = option(row, '--campaign')
        if (not {str(ROOT / 'peptide_transfer_eval.py'), str(ROOT / 'peptide_fsbo_gpu_eval.py')}
                .intersection(row['words']) or 'run-task' not in row['words']
                or directory not in {str(CAMPAIGN / m) for m in METHODS} or row['pid'] in recorded):
            continue
        cpu = int(option(row, '--cpu'))
        if cpu in args.cpus:
            active.append(dict(pid=row['pid'], process_start=row['start'], method=Path(directory).name,
                               task=option(row, '--task'), cpu=cpu, server=option(row, '--server'),
                               adopted=True, started=time.time()))
    released = set(old.get('released_cpus', []))
    halted = set(old.get('halted_cpus', []))
    owners = old.get('baseline_owners', {})
    settings = read(BASELINE / 'workers' / 'tpe_runtime.json', {})
    offset = len(settings.get('cpus', {}).get('local', [])) if args.side == 'remote' else 0
    for index, cpu in enumerate(args.cpus):
        if not slot_failed(args.side, cpu, offset + index):
            halted.discard(cpu)
    children, claims = {}, {}
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with claim(status_path.with_suffix('.lock')) as lease:
        if lease is None:
            raise RuntimeError('A queue already owns this machine')
        while not stopped:
            rows = processes()
            live = {r['pid']: r for r in rows}
            baseline = {int(option(r, '--cpu')): r for r in tpe_workers(rows)}
            for cpu, worker in baseline.items():
                task = next((option(row, '--task') for row in rows
                             if str(ROOT / 'peptide_tpe_runtime.py') in row['words']
                             and 'run-task' in row['words'] and option(row, '--cpu') == str(cpu)), None)
                previous = owners.get(str(cpu), {})
                owners[str(cpu)] = dict(pid=worker['pid'], process_start=worker['start'],
                                       index=int(option(worker, '--index')),
                                       last_task=task or previous.get('last_task'))
            for row in active[:]:
                if row['pid'] in live and live[row['pid']]['start'] == row['process_start']:
                    index = args.cpus.index(row['cpu'])
                    server = configured_server(args.side, row['cpu'], index, args.servers)
                    request_reroute(row, server)
                    continue
                result = children.pop(row['pid']).wait() if row['pid'] in children else None
                handle = claims.pop(row['pid'], None)
                if handle:
                    handle.__exit__(None, None, None)
                success = ready('rgpe') if row['method'] == 'validation' else row['task'] in completed(row['method'])
                if not success and reroute_checkpoint(row, result):
                    resume[str(row['cpu'])] = dict(method=row['method'], task=row['task'])
                elif not success or result not in (None, 0):
                    halted.add(row['cpu'])
                    write(CAMPAIGN / 'workers' / ('error_%s_cpu%d.json' % (args.side, row['cpu'])),
                          dict(worker=row, returncode=result, time=time.time()))
                active.remove(row)
            occupied = occupied_cpus(rows)
            for index, cpu in enumerate(args.cpus):
                if cpu not in baseline and cpu not in occupied:
                    worker_index = owners.get(str(cpu), {}).get('index', offset + index)
                    if slot_failed(args.side, cpu, worker_index):
                        halted.add(cpu)
                    else:
                        halted.discard(cpu)
                        released.add(cpu)
            done = {method: completed(method) for method in METHODS}
            gates = {method: ready(method) for method in METHODS}
            admissible = memory_available(active)
            for index, cpu in enumerate(args.cpus):
                if cpu not in released or cpu in occupied or cpu in halted:
                    continue
                server = configured_server(args.side, cpu, index, args.servers)
                method = None
                validation = read(TRAINING / 'remote_queue.json', {}).get('status') == 'complete'
                gate_lease = None
                if (args.side == 'remote' and validation and not gates['rgpe']
                        and not any(r['method'] == 'validation' for r in active)
                        and admissible >= RESERVATION['validation']):
                    gate_lease = claim(CAMPAIGN / 'workers' / 'rgpe_validation.lock')
                    if gate_lease.__enter__() is not None:
                        method, task = 'validation', 'full800'
                    else:
                        gate_lease.__exit__(None, None, None)
                        gate_lease = None
                task_lease = gate_lease
                if method is None:
                    for candidate in METHODS:
                        if not gates[candidate] or admissible < RESERVATION[candidate]:
                            continue
                        recipe = read(CAMPAIGN / candidate / 'recipe.json')
                        tasks = recipe['test']
                        preferred = resume.get(str(cpu), {})
                        first = [preferred['task']] if preferred.get('method') == candidate else []
                        ordered = first + tasks[index::len(args.cpus)] + tasks
                        for task in dict.fromkeys(ordered):
                            if task in done[candidate]:
                                continue
                            task_lease = claim(CAMPAIGN / candidate / 'dispatch' / (task + '.lock'))
                            if task_lease.__enter__() is None:
                                task_lease.__exit__(None, None, None)
                                continue
                            with claim(CAMPAIGN / candidate / 'locks' / (task + '.lock')) as available:
                                if available is None:
                                    task_lease.__exit__(None, None, None)
                                    continue
                            method = candidate
                            break
                        if method:
                            break
                if method is None:
                    continue
                if method == 'validation':
                    command = [PYTHON, str(Path(__file__).resolve()), 'validate-rgpe', '--cpu', str(cpu), '--server', server]
                else:
                    command = [PYTHON, str(task_runner(method)), 'run-task',
                               '--campaign', str(CAMPAIGN / method), '--task', task,
                               '--cpu', str(cpu), '--server', server]
                log = CAMPAIGN / 'workers' / ('%s_%s_cpu%d.log' % (args.side, method, cpu))
                try:
                    with log.open('ab') as stream:
                        child = subprocess.Popen(['taskset', '-c', str(cpu)] + command,
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                            env=ENV, start_new_session=True)
                except Exception:
                    task_lease.__exit__(*sys.exc_info())
                    raise
                started = process_start(child.pid)
                if started is None:
                    result = child.wait()
                    task_lease.__exit__(None, None, None)
                    halted.add(cpu)
                    write(CAMPAIGN / 'workers' / ('error_%s_cpu%d.json' % (args.side, cpu)),
                          dict(method=method, task=task, returncode=result, log=str(log), time=time.time()))
                    continue
                row = dict(pid=child.pid, process_start=started, method=method, task=task,
                           cpu=cpu, server=server, command=command, log=str(log), started=time.time())
                active.append(row)
                resume.pop(str(cpu), None)
                children[child.pid], claims[child.pid] = child, task_lease
                occupied.add(cpu)
                admissible -= RESERVATION[method]
                write(status_path, dict(side=args.side, pid=os.getpid(), status='running', active=active,
                      baseline_owners=owners, released_cpus=sorted(released),
                      halted_cpus=sorted(halted), reroute_resume=resume, updated=time.time()))
            finished = all(len(done[m]) == 200 for m in METHODS) and not active
            write(status_path, dict(side=args.side, pid=os.getpid(),
                  status='complete' if finished else ('running' if active else 'queued'), active=active,
                  cpu_slots=args.cpus, released_cpus=sorted(released), halted_cpus=sorted(halted),
                  baseline_owners=owners,
                  reroute_resume=resume,
                  complete={m: len(done[m]) for m in METHODS}, ready=gates,
                  tpe_workers=len(baseline), memory_available_after_reservations_gib=admissible,
                  updated=time.time()))
            if finished:
                break
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    queue = sub.add_parser('queue')
    queue.add_argument('--side', choices=('local', 'remote'), required=True)
    queue.add_argument('--cpus', required=True)
    queue.add_argument('--servers', required=True)
    queue.add_argument('--launch', action='store_true')
    validation = sub.add_parser('validate-rgpe')
    validation.add_argument('--cpu', type=int, required=True)
    validation.add_argument('--server', required=True)
    recovery = sub.add_parser('recover-inference')
    recovery.add_argument('--side', choices=('local', 'remote'), required=True)
    args = parser.parse_args()
    if args.action == 'validate-rgpe':
        validate_rgpe(args.cpu, args.server)
        return
    if args.action == 'recover-inference':
        recover_inference(args.side)
        return
    args.cpus = [int(c) for c in args.cpus.split(',')]
    args.servers = args.servers.split(',')
    if len(args.cpus) != len(set(args.cpus)) or not args.servers:
        parser.error('CPU slots must be distinct and inference servers are required')
    CAMPAIGN.joinpath('workers').mkdir(parents=True, exist_ok=True)
    if args.launch:
        settings = read(BASELINE / 'workers' / 'tpe_runtime.json')
        if args.cpus != settings['cpus'][args.side]:
            raise ValueError('Queue slots must match the existing TPE allocation')
        previous = read(CAMPAIGN / 'workers' / ('queue_' + args.side + '.json'), {})
        current = process(previous.get('pid', -1))
        if current and str(Path(__file__).resolve()) in current['words']:
            raise RuntimeError('This machine already has a transfer queue')
        command = [PYTHON, str(Path(__file__).resolve())] + [a for a in sys.argv[1:] if a != '--launch']
        log = CAMPAIGN / 'workers' / ('queue_' + args.side + '.log')
        with log.open('ab') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
        write(CAMPAIGN / 'workers' / ('queue_' + args.side + '_launch.json'),
              dict(side=args.side, pid=child.pid, command=command, log=str(log),
                   script_sha256=digest(Path(__file__)), cpu_slots=args.cpus,
                   servers=args.servers, priority=list(METHODS), started=time.time()))
        print(json.dumps(dict(side=args.side, pid=child.pid, log=str(log))), flush=True)
    else:
        dispatch(args)


if __name__ == '__main__':
    main()
