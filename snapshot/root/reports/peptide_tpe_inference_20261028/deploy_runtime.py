#!/usr/bin/env python3
"""Validate and deploy the authorized infrastructure-only TPE runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
CAMPAIGN = Path('/root/data/logs/peptide_test200_baselines_20261027_v2')
SERVER = Path('/root/data/logs/peptide_bo_launch_20261027_v5/peptide_vae_server.py')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
PORTS = (18871, 18872, 18873, 18875, 18876, 18877, 18878)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def health(port):
    with urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=5) as response:
        return json.load(response)


def processes():
    result = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            words = (entry / 'cmdline').read_bytes().split(b'\0')
            command = [v.decode() for v in words if v]
            if str(SERVER) in command:
                result[int(command[command.index('--port') + 1])] = (int(entry.name), command)
        except (OSError, ValueError):
            pass
    return result


def upgrade():
    validation = json.loads((ROOT / 'validation.json').read_text())
    assert validation['passed']
    rows = []
    for port in PORTS:
        pid, command = processes()[port]
        old_pid = pid
        if '--skip-identical-weights' not in command or '--http-keepalive' not in command:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + 30
            while port in processes() and time.monotonic() < deadline:
                time.sleep(0.2)
            if port in processes():
                raise RuntimeError('Inference service did not stop: %d' % port)
            command += ['--skip-identical-weights', '--http-keepalive']
            log = CAMPAIGN / 'workers' / ('vae_optimized_%d.log' % port)
            with log.open('ab') as output:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            pid = child.pid
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError('Inference service failed to start: %d' % port)
                try:
                    if health(port)['http_keepalive']:
                        break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('Inference service readiness timed out: %d' % port)
        current = health(port)
        assert current['errors'] == 0 and current['skip_identical_weights'] and current['http_keepalive']
        row = dict(port=port, pid=pid, previous_pid=old_pid, command=command, started=time.time(),
                   gpu=current['gpu'], cpu=int(command[command.index('--cpu') + 1]),
                   gpu_memory_fraction=float(command[command.index('--gpu-memory-fraction') + 1])
                   if '--gpu-memory-fraction' in command else 0.05,
                   inference_only=True, export_service_untouched=True, health=current)
        previous = CAMPAIGN / 'workers' / ('vae_replica_%d.json' % port)
        if previous.exists():
            row['previous_record'] = json.loads(previous.read_text())
            write(previous, row)
        rows.append(row)
        write(ROOT / 'deployment.json', dict(services=rows, export_port_excluded=18874,
                                            server_sha256=digest(SERVER)))
        print(json.dumps(dict(port=port, pid=pid, status='ready')), flush=True)


def verify_services():
    os.sched_setaffinity(0, {12})
    sys.path.insert(0, str(TOOLS))
    import peptide_test200 as adapter
    from mindware.domains.protein.peptide_rpc import dumps
    from peptide_tpe_rpc import PersistentRemoteObjective
    from types import SimpleNamespace
    torch = adapter.torch
    torch.set_num_threads(1)
    recipe = adapter.verify(CAMPAIGN / 'tpe')
    weights = torch.load(recipe['vae'], map_location='cpu', weights_only=False)
    version = hashlib.sha256(dumps(weights)).hexdigest()
    objective = SimpleNamespace(vae=SimpleNamespace(state_dict=lambda: weights), vae_forward=None)
    identity = 'tpe_pool_verification_' + str(time.time_ns())
    clients = []
    try:
        z = None
        expected = None
        results = []
        for port in (18876,) + tuple(p for p in PORTS if p != 18876):
            before = health(port)
            client = PersistentRemoteObjective(objective, 'http://127.0.0.1:%d' % port, identity)
            clients.append(client)
            client.sync()
            version = client.version
            if z is None:
                z = client.request(dict(op='encode_mean', version=version, seed=73,
                    sequences=['RRYYEQLEQASRKGNRGFRR', 'KLKLKLKLKL']))['z']
            answer = client.request(dict(op='decode', version=version, seed=93, z=z))['sequences']
            if expected is None:
                expected = answer
            assert answer == expected
            socket_id = id(client.connection.sock)
            assert client.request(dict(op='decode', version=version, seed=93, z=z,
                                       verification_repeat=1))['sequences'] == expected
            assert id(client.connection.sock) == socket_id
            after = health(port)
            assert after['errors'] == 0
            assert after['weight_loads'] - before['weight_loads'] <= 1
            assert after['weight_loads_skipped'] > before['weight_loads_skipped']
            results.append(dict(port=port, seeded_equivalence=True, persistent_socket=True,
                                repeated_weight_load_skipped=True, errors=after['errors']))
        write(ROOT / 'pool_validation.json', dict(passed=True, services=results, oracle_calls=0,
            server_sha256=digest(SERVER), rpc_sha256=digest(TOOLS / 'peptide_tpe_rpc.py')))
        print(json.dumps(dict(passed=True, services=len(results), oracle_calls=0)), flush=True)
    finally:
        for client in clients:
            client.close()


def configure():
    report = CAMPAIGN / 'workers' / 'tpe_runtime_validation.json'
    write(report, json.loads((ROOT / 'validation.json').read_text()))
    pool = json.loads((ROOT / 'pool_validation.json').read_text())
    assert pool['passed'] and pool['server_sha256'] == digest(SERVER)
    values = json.loads(report.read_text())
    assert values['passed']
    cpus = dict(local=list(range(12,18)) + [19,20] + list(range(24,38)),
                remote=list(range(10,46)))
    runtime = dict(wrapper=str(TOOLS / 'peptide_tpe_runtime.py'),
                   rpc=str(TOOLS / 'peptide_tpe_rpc.py'),
                   wrapper_sha256=values['wrapper_sha256'], rpc_sha256=values['rpc_sha256'],
                   validation_report=str(report), validation_sha256=digest(report),
                   workers=58, cpus=cpus, infrastructure_only=True,
                   search_space=dict(dimensions=256, bounds=[-4.0, 4.0], batch_size=1,
                                     decoder='same frozen author peptide VAE as LOLBO'),
                   reserved_remote_cpus=list(range(10)), export_port_excluded=18874)
    for name in ('wrapper', 'rpc'):
        assert digest(Path(runtime[name])) == runtime[name + '_sha256']
    write(CAMPAIGN / 'workers' / 'tpe_runtime.json', runtime)
    route_path = CAMPAIGN / 'workers' / 'vae_routes.json'
    routes = json.loads(route_path.read_text())
    for side, values in cpus.items():
        for index, cpu in enumerate(values):
            routes['%s:tpe:%d' % (side, cpu)] = PORTS[index % len(PORTS)]
    write(route_path, routes)
    print(json.dumps(dict(configured=True, workers=58, local=22, remote=36)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('upgrade', 'verify-services', 'configure'))
    args = parser.parse_args()
    {'upgrade': upgrade, 'verify-services': verify_services, 'configure': configure}[args.action]()


if __name__ == '__main__':
    main()
