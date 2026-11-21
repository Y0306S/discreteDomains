#!/usr/bin/env python3
"""Launch the approved peptide evaluation CPU layout on machines 2 and 4."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

CAMPAIGN = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
WORKER = Path('/root/data/cys/llmoptimizer_benchmark/tools/peptide_test200.py')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def method_paused(method):
    return (CAMPAIGN / method / 'paused.json').exists()


def method_complete(method):
    return sum(p.name.endswith('.tar.gz') for p in (CAMPAIGN / method / 'runs').iterdir()) == 200


def runtime_settings():
    path = CAMPAIGN / 'workers' / 'tpe_runtime.json'
    if not path.exists():
        return {}
    settings = json.loads(path.read_text())
    gate = Path(settings['validation_report'])
    if (hashlib.sha256(gate.read_bytes()).hexdigest() != settings['validation_sha256']
            or not json.loads(gate.read_text())['passed']
            or hashlib.sha256(Path(settings['wrapper']).read_bytes()).hexdigest() != settings['wrapper_sha256']):
        raise ValueError('Optimized TPE runtime has not passed its pinned validation')
    report = json.loads(gate.read_text())
    for name in ('wrapper', 'rpc'):
        digest = hashlib.sha256(Path(settings[name]).read_bytes()).hexdigest()
        if digest != settings[name + '_sha256'] or digest != report[name + '_sha256']:
            raise ValueError('Validated TPE runtime asset changed: ' + name)
    return settings


def worker_script(method):
    return Path(runtime_settings().get('wrapper', WORKER)) if method == 'tpe' else WORKER


def worker_alive(row):
    try:
        words = Path('/proc/%d/cmdline' % row['pid']).read_bytes().split(b'\0')
        scripts = {str(WORKER).encode(), str(worker_script(row['method'])).encode()}
        return (bool(scripts.intersection(words))
                and str(CAMPAIGN / row['method']).encode() in words)
    except OSError:
        return False


def server_port(side, method, cpu, index):
    routes = CAMPAIGN / 'workers' / 'vae_routes.json'
    mapping = json.loads(routes.read_text()) if routes.exists() else {}
    return int(mapping.get('%s:%s:%d' % (side, method, cpu),
                           (18871, 18872, 18873)[index % 3]))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def health():
    values = {}
    routes = CAMPAIGN / 'workers' / 'vae_routes.json'
    configured = json.loads(routes.read_text()) if routes.exists() else {}
    for port in sorted({18871, 18872, 18873} | {int(p) for p in configured.values()}):
        try:
            with urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=5) as reply:
                values[str(port)] = json.load(reply)
        except OSError as exc:
            values[str(port)] = dict(error=str(exc))
    return values


def tunnels():
    command = ['ssh', '-o', 'BatchMode=yes', 'LLMOptimizer4', PYTHON, str(Path(__file__).resolve()), 'health']
    remote = json.loads(subprocess.check_output(command, text=True))
    missing = [int(port) for port, value in remote.items() if 'error' in value]
    if missing:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                   '-o', 'ServerAliveInterval=30', '-N']
        for port in missing:
            command += ['-R', '127.0.0.1:%d:127.0.0.1:%d' % (port, port)]
        command.append('LLMOptimizer4')
        path = CAMPAIGN / 'workers' / 'tunnel.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('ab') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2)
        if child.poll() not in (None, 0):
            raise RuntimeError('Loopback inference tunnel failed to start')
        write(CAMPAIGN / 'workers' / 'tunnel.json', dict(pid=child.pid, ports=missing))
        remote = json.loads(subprocess.check_output(
            ['ssh', '-o', 'BatchMode=yes', 'LLMOptimizer4', PYTHON, str(Path(__file__).resolve()), 'health'], text=True))
    if any('error' in value or value['errors'] for value in remote.values()):
        raise RuntimeError('Remote inference service health failed')
    return remote


def start(side, staged=False):
    sys.path.insert(0, str(WORKER.parent))
    import peptide_test200 as adapter
    for method in ('tpe', 'lolbo'):
        adapter.verify(CAMPAIGN / method)
        adapter.preflight(CAMPAIGN / method)
    bounds = json.loads(Path('/root/reports/peptide_train800_test200_audit_20261027/test200_latent_bounds.json').read_text())
    if not bounds['passed']:
        raise ValueError('Initialization representation audit failed')
    current = health()
    if any('error' in v or v['errors'] for v in current.values()):
        raise ValueError('GPU inference health check failed')
    if side == 'local':
        cpus = dict(tpe=[12,13,14,15,16,17,19,20,24,25,26], lolbo=list(range(27,39)))
        indices = dict(tpe=list(range(11)), lolbo=list(range(12)))
        from importlib.machinery import SourceFileLoader
        resources = SourceFileLoader('peptide_resources', '/root/data/logs/peptide_bo_launch_20261027_v5/resources.py').load_module()
        snapshot = resources.snapshot()
        worker_record = CAMPAIGN / 'workers' / (side + '.json')
        current_workers = json.loads(worker_record.read_text()) if worker_record.exists() else []
        already_live = 0
        for row in current_workers:
            already_live += int(worker_alive(row))
        settings = runtime_settings()
        if settings:
            cpus['tpe'] = settings['cpus']['local']
            indices['tpe'] = list(range(len(cpus['tpe'])))
        requested = sum(min(2, len(v)) if staged else len(v) for m, v in cpus.items()
                        if not method_paused(m) and not method_complete(m))
        if snapshot['unreserved_cpu_quota'] < requested - already_live:
            raise ValueError('Local CPU quota cannot accommodate the approved layout')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
        free = {int(row.split(',')[0]): int(row.split(',')[1]) for row in gpu.splitlines()}
        if any(free[g] < 8192 for g in (0,2,3)):
            raise ValueError('Shared GPU inference requires 8 GiB free headroom')
    else:
        cpus = dict(tpe=list(range(10,28)), lolbo=list(range(28,46)))
        indices = dict(tpe=list(range(12,30)), lolbo=list(range(12,30)))
        settings = runtime_settings()
        if settings:
            cpus['tpe'] = settings['cpus']['remote']
            offset = len(settings['cpus']['local'])
            indices['tpe'] = list(range(offset, offset + len(cpus['tpe'])))
        from importlib.machinery import SourceFileLoader
        resources = SourceFileLoader('peptide_resources', '/root/data/logs/peptide_bo_launch_20261027_v5/resources.py').load_module()
        snapshot = resources.snapshot()
        if any(cpu < 10 for cpu in cpus['tpe']):
            raise ValueError('Remote CPUs 0-9 are reserved for FSBO/RGPE training')
        if snapshot['available_gib'] < 32:
            raise ValueError('Remote transfer training needs at least 32 GiB memory headroom')
    record = CAMPAIGN / 'workers' / (side + '.json')
    existing = json.loads(record.read_text()) if record.exists() else []
    live_cpus = {r['cpu'] for r in existing if worker_alive(r)}
    requested_cpus = {cpu for method, values in cpus.items()
                      if not method_paused(method) and not method_complete(method)
                      for cpu in (values[:2] if staged else values)}
    blocked = requested_cpus - set(snapshot['free_physical_cpus']) - live_cpus
    if blocked or len(requested_cpus - live_cpus) > snapshot['unreserved_cpu_quota']:
        raise ValueError('CPU slots unavailable; preserve queue: ' + str(sorted(blocked)))
    rows = list(existing)
    for method in ('tpe', 'lolbo'):
        if method_paused(method) or method_complete(method):
            continue
        allocation = list(zip(cpus[method], indices[method]))
        if staged:
            allocation = allocation[:2]
        for cpu, index in allocation:
            previous = next((r for r in existing if r['method'] == method and r['cpu'] == cpu), None)
            if previous:
                if worker_alive(previous):
                    continue
                rows.remove(previous)
            port = server_port(side, method, cpu, index)
            total_workers = runtime_settings().get('workers', 30) if method == 'tpe' else 30
            command = [PYTHON, str(worker_script(method)), 'worker', '--campaign', str(CAMPAIGN / method),
                       '--cpu', str(cpu), '--index', str(index), '--workers', str(total_workers),
                       '--server', 'http://127.0.0.1:%d' % port]
            log = CAMPAIGN / 'workers' / ('%s_%s_cpu%d.log' % (side, method, cpu))
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('ab') as stream:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, env=ENV, start_new_session=True)
            rows.append(dict(method=method, cpu=cpu, index=index, pid=child.pid, server_port=port,
                             command=command, log=str(log), started=time.time(), host=socket.gethostname()))
            write(record, rows)
    print(json.dumps(dict(side=side, workers=sum(worker_alive(row) for row in rows),
                         worker_records=len(rows), staged=staged, record=str(record))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('local', 'remote', 'health', 'tunnels'))
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    if args.action == 'health':
        print(json.dumps(health()))
    elif args.action == 'tunnels':
        print(json.dumps(tunnels()))
    else:
        start(args.action, args.staged)


if __name__ == '__main__':
    main()
