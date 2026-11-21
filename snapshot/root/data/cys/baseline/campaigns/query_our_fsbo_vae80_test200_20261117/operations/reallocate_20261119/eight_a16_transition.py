"""Audited resource handoff; retain SQL journals and all independent peptide jobs."""
import concurrent.futures
import json
import os
from pathlib import Path
import pickle
import shutil
import socket
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'batch_code'))
sys.path.insert(0, str(ROOT / 'operations/recovery'))
from query_protocol import atomic, digest
import resources
import recovery

def load(path):
    return json.loads(Path(path).read_text())

def probes():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(lambda h: recovery.probe(h) if h == 'LLMOptimizer2'
                               else recovery.host_call(h, 'probe'), resources.config()['hosts']))
    atomic(OPS / 'eight_a16_live_probes.json', result)
    print(json.dumps([dict(host=p['host'], alive=sum(w['alive'] for w in p['workers']),
                           ready=sum(d['ready'] for d in p['databases']),
                           sql=sum(bool(d.get('active')) for d in p['databases']),
                           unrecorded=p['unrecorded_workers']) for p in result]), flush=True)
    return result

def pin_reviewed(names):
    path = ROOT / 'evidence/batch_code_pins.json'
    pins = load(path)
    for name in names:
        p = ROOT / name
        pins[str(p)] = digest(p)
    atomic(path, pins)
    recovery.verify_pins()

def drain():
    assert (ROOT / 'PAUSE').exists()
    while recovery.pause_age() < 292:
        time.sleep(min(30, 292 - recovery.pause_age()))
    # Existing recovery policy allows stopping generation-blocked optimizers
    # only after five minutes and a fresh check of idle private databases.
    for host in resources.config()['hosts']:
        result = recovery.drain_generation(host) if host == 'LLMOptimizer2' else recovery.host_call(host, 'drain')
        print(json.dumps(result), flush=True)
    deadline = time.monotonic() + 60
    while recovery.process_matches(load(ROOT / 'services/batch_controller.process.json')):
        assert time.monotonic() < deadline
        time.sleep(1)
    print('All Query optimizers and controller exited; waiting for in-flight model calls.', flush=True)
    deadline = time.monotonic() + 600
    while True:
        h = resources.json_get('http://127.0.0.1:21811/health')
        busy = sum(bool(v.get('busy')) for v in h['replica_health'])
        if h['queued'] == 0 and busy == 0:
            break
        assert time.monotonic() < deadline
        print('Remaining model calls:', busy, 'queued:', h['queued'], flush=True)
        time.sleep(10)
    print('Model calls drained; durable broker can be handed off.', flush=True)

def cutover():
    assert (ROOT / 'PAUSE').exists()
    assert not load(ROOT / 'operations/recovery/config.json')['enabled']
    ps = probes()
    assert not any(p['unrecorded_workers'] for p in ps)
    assert not any(w['alive'] for p in ps for w in p['workers'])
    assert all(d['ready'] and not d['active'] for p in ps for d in p['databases'])
    assert not recovery.process_matches(load(ROOT / 'services/batch_controller.process.json'))
    # Completed tasks have independently validated completion records. Audit
    # unfinished journals only; no entries are rewritten during this handoff.
    audited = []
    for path in (ROOT / 'orpo_batch10/runs').glob('*/runtime.sqlite'):
        if path.with_name('complete.json').exists():
            continue
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=120) as db:
            assert db.execute('PRAGMA quick_check').fetchone() == ('ok',), str(path)
            pending = [key for key, blob in db.execute('SELECT key,value FROM kv')
                       if key.startswith('measurement/') and pickle.loads(blob).get('status') == 'started']
        assert not pending, (str(path), pending)
        audited.append(str(path))
    h = resources.json_get('http://127.0.0.1:21811/health')
    assert h['queued'] == 0 and not any(v.get('busy') for v in h['replica_health'])
    atomic(OPS / 'eight_a16_boundary_audit.json', dict(at=time.time(), probes=ps,
           unfinished_journals=audited, unresolved_SQL=0, broker=h))
    resources.stop_service('orpo_resources_broker')
    deadline = time.monotonic() + 30
    while resources.record_alive(load(ROOT / 'services/orpo_resources_broker.process.json')):
        assert time.monotonic() < deadline
        time.sleep(.2)
    for name in ['llm12', 'llm13', 'llm14']:
        resources.stop_service(name + '_resources', 'LLMOptimizer3')
    current = resources.config()
    current['llm'] = [w for w in current['llm'] if w['name'] not in ['llm12', 'llm13', 'llm14']]
    assert len(current['llm']) == 15
    current['capacity_reallocation_20261119'].update(llm_replicas=15,
        llm_layout=dict(A800=12, A16=3), transition='five full-context pairs loading')
    atomic(ROOT / 'resources.json', current)
    urls = [f"http://127.0.0.1:{w['port']}" for w in current['llm']]
    for role in ['orpo', 'warm_start']:
        path = ROOT / f'services/{role}.resources.broker.json'
        config = load(path); config['replicas'] = urls; atomic(path, config)
    # Change host 3 transport only; retain the healthy host 4 database/VAE paths.
    for name in ['resources_host3_tunnel', 'resources_host3_context_tunnel']:
        resources.stop_service(name)
    deadline = time.monotonic() + 30
    while any(resources.record_alive(load(ROOT / f'services/{name}.process.json'))
              for name in ['resources_host3_tunnel', 'resources_host3_context_tunnel']):
        assert time.monotonic() < deadline
        time.sleep(.2)
    command = ['ssh', '-NT', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
               '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3']
    for port in range(21940, 21948):
        command += ['-L', f'127.0.0.1:{port}:127.0.0.1:{port}']
    ports = [21810, 21811] + [w['port'] for w in current['vae']] + [w['port'] for w in current['llm'][:12]]
    for port in ports:
        command += ['-R', f'127.0.0.1:{port}:127.0.0.1:{port}']
    resources.start(command + ['LLMOptimizer3'], 'resources_host3_tunnel')
    time.sleep(3)
    assert resources.record_alive(load(ROOT / 'services/resources_host3_tunnel.process.json'))
    for w in current['llm']:
        assert resources.placement_ok(w, resources.json_get(f"http://127.0.0.1:{w['port']}/health")), w
    # Existing model processes stay loaded; restart only the durable broker.
    deadline = time.monotonic() + 30
    while True:
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', 21811))
            break
        except OSError:
            assert time.monotonic() < deadline
            time.sleep(.2)
    resources.start(['/usr/bin/python3', '-u', str(ROOT / 'batch_code/qwen_inference_server.py'),
                     'broker', '--config', str(ROOT / 'services/orpo.resources.broker.json')], 'orpo_resources_broker')
    deadline = time.monotonic() + 60
    while True:
        try:
            h = resources.json_get('http://127.0.0.1:21811/health')
            if h['replicas'] == 15 and all(x.get('ready') for x in h['replica_health']):
                break
        except OSError:
            pass
        assert time.monotonic() < deadline
        time.sleep(1)
    path = ROOT / 'services/active_inference_stage.json'
    marker = load(path); marker.update(replicas=15, placement_verified=True, at=time.time()); atomic(path, marker)
    pin_reviewed(['resources.json', 'services/orpo.resources.broker.json', 'services/warm_start.resources.broker.json',
                  'operations/reallocate_20261119/admit_verified.py'])
    resources.health()
    atomic(OPS / 'eight_a16_transition.json', dict(phase='cutover_ready', at=time.time(),
           live_LLMs=15, unresolved_SQL=0, checkpoints_preserved=True))
    print('Cutover ready; 12 A800 and 3 A16 replicas preserved.', flush=True)

def start_pairs():
    assert load(OPS / 'eight_a16_transition.json')['phase'] == 'cutover_ready'
    planned = load(OPS / 'resources.final_planned.json')
    reviewed = []
    for spec in planned['llm']:
        if spec['name'] not in ['llm12', 'llm13', 'llm14', 'llm18', 'llm19']:
            continue
        for role in ['orpo', 'warm_start']:
            base = load(ROOT / f'services/{role}.llm15.resources.json')
            base.update(name=f"query_hybrid_{role}_{spec['name']}", cpus=[spec['cpu']],
                        port=spec['port'], gpu_devices=spec['gpu_devices'])
            path = ROOT / f"services/{role}.{spec['name']}.resources.json"
            backup = OPS / 'before_eight_a16' / path.relative_to(ROOT)
            if path.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, backup)
            atomic(path, base); reviewed.append(str(path.relative_to(ROOT)))
        resources.start_service([resources.INFER, '-u', str(ROOT / 'batch_code/qwen_inference_server.py'),
              'replica', '--config', str(ROOT / f"services/orpo.{spec['name']}.resources.json")],
              spec['name'] + '_resources', 'LLMOptimizer3')
        print('Loading', spec['name'], spec['gpu_devices'], flush=True)
        time.sleep(1)
    pin_reviewed(reviewed)

def resume():
    assert load(OPS / 'eight_a16_transition.json')['phase'] == 'cutover_ready'
    assert not recovery.process_matches(load(ROOT / 'services/batch_controller.process.json'))
    recovery.verify_pins()
    resources.health()
    (ROOT / 'PAUSE').unlink()
    (ROOT / 'operations/recovery/recovery_pause.json').unlink(missing_ok=True)
    pid = resources.start([resources.PYTHON, '-u', str(ROOT / 'batch_code/run.py'), 'controller'], 'batch_controller')
    atomic(OPS / 'eight_a16_transition.json', dict(phase='running_during_pair_load',
          at=time.time(), controller_pid=pid, live_LLMs=15))
    print('Controller resumed', pid, flush=True)

if __name__ == '__main__':
    {'probe': probes, 'drain': drain, 'cutover': cutover, 'start_pairs': start_pairs, 'resume': resume}[sys.argv[1]]()
