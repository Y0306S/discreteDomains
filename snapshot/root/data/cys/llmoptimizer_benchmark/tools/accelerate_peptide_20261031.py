#!/usr/bin/env python3
"""Checkpoint-preserving operational rollout; no model or budget changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sqlite3
import subprocess
import tarfile
import time
import urllib.request

TOOLS = Path(__file__).resolve().parent
OUT = Path('/root/data/logs/peptide_acceleration_20261031')
CAMPAIGN = Path('/root/data/logs/peptide_transfer_test200_20261028_v1')
TRAIN = Path('/root/data/logs/peptide_transfer_train800_20261027_v2')
SERVICES = Path('/root/data/logs/peptide_fsbo_gpu_20261029')
SETTINGS = Path('/root/data/logs/peptide_test200_baselines_20261027_v2/workers/rgpe_runtime.json')
RGPE_PYTHON = '/root/data/cys/baseline/envs/bl_rgpe/bin/python'
FSBO_PYTHON = '/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid):
    try:
        root = Path('/proc')/str(pid)
        stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(start=stat[19], words=[x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x])
    except FileNotFoundError:
        return None


def signal_verified(pid, script, start=None):
    row = process(pid)
    if row is None:
        return
    if str(TOOLS/script) not in row['words'] or (start is not None and row['start'] != start):
        raise RuntimeError('Process identity changed: '+str(pid))
    os.kill(pid, signal.SIGTERM)


def wait_exit(pid, seconds):
    deadline = time.monotonic()+seconds
    while process(pid):
        if time.monotonic() > deadline:
            raise RuntimeError('Graceful stop timed out: '+str(pid))
        time.sleep(.5)


def launch(command, log, cuda=''):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cuda, OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with Path(log).open('ab') as stream:
        return subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True).pid


def history(task):
    root = CAMPAIGN/'fsbo/runs'/task
    archive = root.with_suffix('.tar.gz')
    if archive.exists():
        with tarfile.open(str(archive), 'r:gz') as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith('/history.json'))
            return json.load(tar.extractfile(member))
    return read(root/'history.json')


def prefix(value, count=None):
    rows = value['observations'][:count]
    # Compare every persisted field, including scores, proposals and fit metadata.
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def drain(side):
    target = OUT/('fsbo_'+side+'_drain.json')
    if target.exists():
        raise RuntimeError('Drain record already exists; inspect before retrying')
    status_path = CAMPAIGN/'workers'/('queue_'+side+'.json')
    queue = read(status_path)
    signal_verified(queue['pid'], 'queue_peptide_transfer.py')
    wait_exit(queue['pid'], 50)
    queue = read(status_path)
    write(OUT/('queue_'+side+'_before.json'), queue)
    rows = [r for r in queue['active'] if r['method'] == 'fsbo']
    baseline = {}
    for row in rows:
        h = history(row['task'])
        baseline[row['task']] = dict(rows=len(h['observations']), prefix_sha256=prefix(h),
                                     proposals=h['proposals'], fresh_evaluations=h['fresh_evaluations'])
    record = dict(status='draining', side=side, started=time.time(), workers=rows, baseline=baseline)
    write(target, record)
    for row in rows:
        signal_verified(row['pid'], 'peptide_fsbo_gpu_eval.py', row['process_start'])
    for row in rows:
        wait_exit(row['pid'], 1800)
    for row in rows:
        task = row['task']
        h, before = history(task), baseline[task]
        assert len(h['observations']) >= before['rows']
        assert prefix(h, before['rows']) == before['prefix_sha256'], 'Changed observation prefix: '+task
        assert h['proposals'] >= before['proposals'] and h['fresh_evaluations'] >= before['fresh_evaluations']
        root = CAMPAIGN/'fsbo/runs'/task
        if not root.with_suffix('.tar.gz').exists():
            with sqlite3.connect('file:'+str(root/'runtime.sqlite')+'?mode=ro', uri=True) as db:
                pending = db.execute("SELECT value FROM meta WHERE key='pending'").fetchone()
                assert pending is None or bytes(pending[0]) == pickle.dumps(None, protocol=4), 'Uncommitted batch: '+task
            queue.setdefault('reroute_resume', {})[str(row['cpu'])] = dict(method='fsbo', task=task)
        before.update(drained_rows=len(h['observations']), drained_proposals=h['proposals'],
                      drained_prefix_sha256=prefix(h))
    queue['active'] = [r for r in queue['active'] if r['method'] != 'fsbo']
    queue.update(status='gpu_topology_handoff', updated=time.time())
    write(status_path, queue)
    record.update(status='drained', finished=time.time(), all_prefixes_preserved=True,
                  pending_batches_cleared=True)
    write(target, record)
    print(json.dumps(dict(side=side, status='drained', workers=len(rows))), flush=True)


def resume(side):
    assert read(OUT/('fsbo_'+side+'_drain.json'))['status'] == 'drained'
    status = read(CAMPAIGN/'workers'/('queue_'+side+'.json'))
    if process(status['pid']):
        raise RuntimeError('Queue is already running')
    path = CAMPAIGN/'workers'/('queue_'+side+'_launch.json')
    plan = read(path)
    pid = launch(plan['command'], plan['log'])
    plan.update(pid=pid, started=time.time(), acceleration_record=str(OUT))
    write(path, plan)
    write(OUT/('fsbo_'+side+'_resume.json'), dict(pid=pid, started=time.time()))
    print(json.dumps(dict(side=side, pid=pid)), flush=True)


def health(endpoint):
    with urllib.request.urlopen(endpoint+'/health', timeout=10) as response:
        return json.load(response)


def redistribute():
    assert all(read(OUT/('fsbo_'+side+'_drain.json'))['status'] == 'drained' for side in ('local', 'remote'))
    path = SERVICES/'services.json'
    plan = read(path)
    write(OUT/'services_before.json', plan)
    write(OUT/'runtime_before.json', read(CAMPAIGN/'fsbo/gpu_runtime.json'))
    for gpu, row in enumerate(sorted(plan['services'], key=lambda r:r['port'])):
        h = health(row['endpoint'])
        assert not h['busy'] and h['waiting'] == 0 and h['errors'] == 0
        if gpu == 3:
            continue
        signal_verified(row['pid'], 'peptide_fsbo_gpu_service.py')
        wait_exit(row['pid'], 30)
        command = [FSBO_PYTHON, '-u', str(TOOLS/'peptide_fsbo_gpu_service.py'),
                   '--recipe', str(CAMPAIGN/'fsbo/recipe.json'), '--token-file', str(SERVICES/'service_token'),
                   '--port', str(row['port']), '--cpu', str(row['cpu']), '--memory-fraction', '0.08']
        row.update(pid=launch(command, row['log'], str(gpu)), gpu=gpu, started=time.time())
        write(path, plan)
        deadline = time.monotonic()+180
        while True:
            if process(row['pid']) is None:
                raise RuntimeError('New GPU service exited: '+str(gpu))
            try:
                h = health(row['endpoint'])
                assert h['pid'] == row['pid'] and int(h['gpu']) == gpu and h['errors'] == 0
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)
    runtime_path = CAMPAIGN/'fsbo/gpu_runtime.json'
    runtime = read(runtime_path)
    runtime.pop('server_gpu', None)
    runtime.update(server_gpus=[0,1,2,3], topology_changed_at=time.time())
    write(runtime_path, runtime)
    write(OUT/'fsbo_topology.json', dict(services=plan['services'], healthy=[health(r['endpoint']) for r in plan['services']]))
    print('One FSBO service per GPU; existing ports and model implementation preserved', flush=True)


def rgpe_remote():
    assignment_path = OUT/'rgpe_assignments.json'
    assert not assignment_path.exists(), 'Assignments already exist'
    # Stop only the dispatcher; its detached in-progress fits keep running.
    pids = []
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            p = process(int(path.name))
            if p and str(TOOLS/'peptide_rgpe_queue.py') in p['words']:
                pids.append(int(path.name))
    assert len(pids) == 1, pids
    signal_verified(pids[0], 'peptide_rgpe_queue.py')
    wait_exit(pids[0], 10)
    active = read(TRAIN/'remote_queue.json')['rgpe_active']
    write(OUT/'rgpe_before.json', read(TRAIN/'remote_queue.json'))
    manifest = read(TRAIN/'corpus/manifest.json')
    with sqlite3.connect(str(TRAIN/'rgpe/models.sqlite'), timeout=120) as db:
        complete = dict(db.execute('SELECT task,sha256 FROM models').fetchall())
    live_tasks = {r['task'] for r in active.values()}
    pending = [t for t in manifest['source_tasks'] if t not in complete and t not in live_tasks]
    # Reserve a proportional tail for eight additional workers; no live fits move.
    count = round(len(pending)*8/19)
    reserved = pending[-count:] if count else []
    write(assignment_path, dict(default_owner='remote', owners={t:'machine3' for t in reserved},
          corpus_sha256=digest(TRAIN/'corpus/manifest.json'), created=time.time()))
    write(OUT/'rgpe_bank_before.json', complete)
    settings = read(SETTINGS)
    settings.update(fit_cpus=list(range(8)), queue_cpus=[0], requested_workers=8)
    write(OUT/'rgpe_machine3_settings.json', settings)
    command = [RGPE_PYTHON, str(TOOLS/'peptide_rgpe_queue.py'), '--assignments', str(assignment_path)]
    pid = launch(command, OUT/'rgpe_remote_queue.log')
    write(OUT/'rgpe_remote_launch.json', dict(pid=pid, command=command, started=time.time(), previous_pid=pids[0]))
    print(json.dumps(dict(pid=pid, reserved_tasks=len(reserved), preserved_fits=len(active))), flush=True)


def rgpe_machine3():
    assert read(OUT/'rgpe_assignments.json')['owners']
    command = [RGPE_PYTHON, str(TOOLS/'peptide_rgpe_queue.py'), '--name', 'machine3',
               '--settings', str(OUT/'rgpe_machine3_settings.json'), '--assignments', str(OUT/'rgpe_assignments.json')]
    pid = launch(command, OUT/'rgpe_machine3_queue.log')
    write(OUT/'rgpe_machine3_launch.json', dict(pid=pid, command=command, started=time.time()))
    print(json.dumps(dict(pid=pid)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['drain','resume','redistribute','rgpe-remote','rgpe-machine3'])
    parser.add_argument('--side', choices=['local','remote'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action in ('drain','resume'):
        assert args.side
        globals()[args.action](args.side)
    else:
        globals()[args.action.replace('-', '_')]()
