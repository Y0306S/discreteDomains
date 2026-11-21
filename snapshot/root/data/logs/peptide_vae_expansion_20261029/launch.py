"""Expand the frozen peptide VAE pool while keeping both worker pools isolated."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
CAMPAIGN = Path('/root/data/logs/peptide_transfer_test200_20261028_v1')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
SERVER = Path('/root/data/logs/peptide_bo_launch_20261027_v5/peptide_vae_server.py')
VAE = Path('/root/data/cys/BOLT/optimization/peptides/uniref_vae/saved_models/') / (
    'dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118_pretrained_authors.pkl')
LAYOUT = [('local', cpu, gpu, 18881 + gpu) for gpu, cpu in enumerate((3, 4, 6, 7))]
LAYOUT += [('remote', cpu, gpu, 18885 + gpu) for gpu, cpu in enumerate((9, 10, 21, 22))]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def health(port):
    with urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=10) as reply:
        return json.load(reply)


def pinned_work():
    reserved = set()
    kinds = ('peptide_transfer_eval.py', 'peptide_transfer_models.py', 'peptide_vae_server.py',
             'peptide_rgpe_fit.py', 'peptide_tpe_runtime.py', 'peptide_bo_v5.py')
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            words = path.joinpath('cmdline').read_bytes().decode().split('\0')
            if any(word.endswith(kind) for word in words for kind in kinds):
                reserved.update(os.sched_getaffinity(int(path.name)))
        except (OSError, ValueError):
            pass
    return reserved


def start():
    reserved = pinned_work()
    quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    quota = int(quota) / int(period)
    existing = {}
    for pool, cpu, gpu, port in LAYOUT:
        record = ROOT / ('service_%d.json' % port)
        if record.exists():
            value = json.loads(record.read_text())
            try:
                words = Path('/proc/%d/cmdline' % value['pid']).read_bytes().decode().split('\0')
                if str(SERVER) in words and str(port) in words:
                    existing[port] = value
            except OSError:
                pass
        if port not in existing and cpu in reserved:
            raise RuntimeError('Requested service CPU is occupied: %d' % cpu)
    requested = {cpu for _, cpu, _, port in LAYOUT if port not in existing}
    if len(reserved | requested) > min(40, quota):
        raise RuntimeError('Expanded inference pool would exceed the 40-core quota')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                  '--format=csv,noheader,nounits'], text=True)
    free = {int(line.split(',')[0]): int(line.split(',')[1]) for line in gpu.splitlines()}
    if any(free[g] < 8192 for g in range(4)):
        raise RuntimeError('Keep at least 8 GiB free per GPU before expansion')
    for pool, cpu, gpu, port in LAYOUT:
        if port in existing:
            continue
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
        command = [PYTHON, str(SERVER), '--gpu', str(gpu), '--cpu', str(cpu), '--port', str(port),
                   '--vae', str(VAE), '--gpu-memory-fraction', '0.02',
                   '--skip-identical-weights', '--http-keepalive']
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (ROOT / ('service_%d.log' % port)).open('ab') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
        value = dict(pool=pool, cpu=cpu, gpu=gpu, port=port, pid=child.pid, command=command,
                     started=time.time(), host='LLMOptimizer2')
        write(ROOT / ('service_%d.json' % port), value)
        existing[port] = value
    deadline = time.monotonic() + 600
    waiting = set(existing)
    while waiting:
        for port in list(waiting):
            try:
                value = health(port)
                if value['errors'] or value['gpu'] != existing[port]['gpu']:
                    raise RuntimeError('Expanded service health mismatch')
                waiting.remove(port)
            except OSError:
                if not Path('/proc/%d' % existing[port]['pid']).exists():
                    raise RuntimeError('Expanded VAE service exited: %d' % port)
        if waiting:
            if time.monotonic() >= deadline:
                raise RuntimeError('Expanded service startup timed out: %s' % sorted(waiting))
            time.sleep(2)
    write(ROOT / 'services.json', dict(services=list(existing.values()),
                                     reserved_cores=len(reserved | requested), checked=time.time()))
    print(json.dumps(dict(services=8, host='LLMOptimizer2', reserved_cores=len(reserved | requested))))


def validate():
    import numpy as np
    import torch
    sys.path.insert(0, '/root/data/logs/peptide_bo_release_20261027_v5')
    from mindware.domains.protein.peptide_rpc import dumps, loads
    weights = torch.load(VAE, map_location='cpu', weights_only=False)
    version = hashlib.sha256(dumps(weights)).hexdigest()
    sequences = ['KTLKIIRLLF', 'RQKNHGIHFRVLAKALR']
    identity = 'vae-expansion-audit-20261029'

    def request(port, value):
        payload = dumps(dict(value, identity=identity))
        call = urllib.request.Request('http://127.0.0.1:%d/rpc' % port, data=payload,
            headers={'Content-Type': 'application/octet-stream',
                     'X-Request-ID': hashlib.sha256(payload).hexdigest()})
        with urllib.request.urlopen(call, timeout=120) as reply:
            result = loads(reply.read())
        if result.get('error') or result.get('missing_weights'):
            raise RuntimeError(result)
        return result

    reference = None
    reports = []
    for port in [18871] + [port for _, _, _, port in LAYOUT]:
        request(port, dict(op='weights', weights=weights, version=version))
        z = request(port, dict(op='encode_mean', version=version, seed=123, sequences=sequences))['z']
        decoded = request(port, dict(op='decode', version=version, seed=456, z=z))['sequences']
        scores = request(port, dict(op='oracle', sequences=sequences, nonce=20261029))['scores']
        if reference is None:
            reference = (z, decoded, scores)
        checks = dict(finite_encoding=bool(torch.isfinite(z).all()),
                      encoding_matches=bool(torch.allclose(z, reference[0], atol=1e-6, rtol=1e-6)),
                      decoding_matches=decoded == reference[1],
                      oracle_matches=bool(np.allclose(scores, reference[2], atol=1e-6, rtol=1e-6)),
                      service_errors=health(port)['errors'] == 0)
        reports.append(dict(port=port, checks=checks))
        request(port, dict(op='release'))
        if not all(checks.values()):
            raise RuntimeError('Expanded service parity failed: %s' % reports[-1])
    write(ROOT / 'validation.json', dict(passed=True, reports=reports, checked=time.time(),
                                        campaign_evaluations=False))
    print(json.dumps(dict(passed=True, checked_services=len(reports))))


def tunnel():
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
               '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
               '-o', 'ServerAliveCountMax=3', '-N']
    for port in range(18885, 18889):
        command.extend(['-R', '127.0.0.1:%d:127.0.0.1:%d' % (port, port)])
    command.append('LLMOptimizer4')
    with (ROOT / 'tunnel.log').open('ab') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT / 'tunnel.json', dict(pid=child.pid, command=command, started=time.time()))
    time.sleep(2)
    if child.poll() is not None:
        raise RuntimeError('Expanded VAE reverse tunnel exited')
    script = ('import json,urllib.request\n'
              'for p in range(18885,18889):\n'
              ' with urllib.request.urlopen("http://127.0.0.1:%d/health"%p,timeout=10) as r:d=json.load(r)\n'
              ' assert d["errors"]==0\n'
              ' print(json.dumps(dict(port=p,errors=d["errors"])))\n')
    report = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                      'LLMOptimizer4', PYTHON, '-'], input=script, text=True)
    write(ROOT / 'remote_health.json', dict(passed=True, results=report.splitlines(), checked=time.time()))
    print(report, end='')


def routes():
    if not json.loads((ROOT / 'validation.json').read_text())['passed']:
        raise RuntimeError('Service parity must pass before rerouting')
    if not json.loads((ROOT / 'remote_health.json').read_text())['passed']:
        raise RuntimeError('Remote service health must pass before rerouting')
    pools = dict(local=[18871, 18872, 18873, 18881, 18882, 18883, 18884],
                 remote=[18875, 18876, 18877, 18878, 18885, 18886, 18887, 18888])
    value = dict(prepared=time.time(), validated=str(ROOT / 'validation.json'))
    for side, ports in pools.items():
        state = json.loads((CAMPAIGN / 'workers' / ('queue_%s.json' % side)).read_text())
        value[side] = {str(cpu): 'http://127.0.0.1:%d' % ports[i % len(ports)]
                       for i, cpu in enumerate(state['cpu_slots'])}
    write(CAMPAIGN / 'workers' / 'vae_routes.json', value)
    print(json.dumps(dict(local_servers=len(pools['local']), remote_servers=len(pools['remote']),
                         route_file=str(CAMPAIGN / 'workers' / 'vae_routes.json'))))


def restart_queues():
    script = '''import json,os,signal,subprocess,sys,time
from pathlib import Path
side=sys.argv[1]
root=Path('/root/data/logs/peptide_transfer_test200_20261028_v1/workers')
state=json.loads((root/('queue_'+side+'.json')).read_text())
launch=json.loads((root/('queue_'+side+'_launch.json')).read_text())
pid=state['pid']
process=Path('/proc')/str(pid)
words=process.joinpath('cmdline').read_bytes().decode().split('\\0')
assert any(w.endswith('queue_peptide_transfer.py') for w in words) and side in words
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+45
while process.exists():
 try:
  status=process.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]
  if status=='Z':break
 except FileNotFoundError:break
 if time.monotonic()>deadline:raise RuntimeError('Queue did not stop cleanly')
 time.sleep(1)
subprocess.run(launch['command']+['--launch'],check=True)
print(json.dumps(dict(side=side,previous_pid=pid,cpu_workers_preserved=True)))
'''
    for side in ('local', 'remote'):
        command = [PYTHON, '-', side]
        if side == 'remote':
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'LLMOptimizer4'] + command
        subprocess.run(command, input=script, text=True, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'validate', 'tunnel', 'routes', 'restart-queues'))
    arguments = parser.parse_args()
    globals()[arguments.action.replace('-', '_')]()
