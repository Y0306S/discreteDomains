"""One-time, audited launch of the authorized query-only FSBO campaign."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

campaign = Path(__file__).resolve().parent
assert socket.gethostname() == 'cs-fa498-8793f-server'
lock = (campaign / 'launch.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
plan = json.loads((campaign / 'worker_plan.json').read_text())
assert not plan['launched'] and not (campaign / 'launch.json').exists()
recipe = json.loads((campaign / 'recipe.json').read_text())
assert recipe['timeout_seconds'] == 50 and recipe['method'] == 'fsbo'
assert not recipe['update_e2e'] and not recipe['recenter']
assert len(recipe['test']) == 200 and len(recipe['train']) == 800
assert not set(recipe['test']) & set(recipe['train'])
assert recipe['allowed_tasks'] == recipe['test']
assert len(recipe['workers']) == len(plan['workers']) == 16
assert recipe['initialization'] == 49 and recipe['proposals'] == 1000
assert recipe['batch_size'] == 10 and recipe['n_candidates'] == 5000
for path, expected in recipe['pinned_files'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
assert hashlib.sha256((campaign / 'tasks.json').read_bytes()).hexdigest() == recipe['tasks_sha256']
maximum = int(Path('/sys/fs/cgroup/memory.max').read_text())
current = int(Path('/sys/fs/cgroup/memory.current').read_text())
assert maximum - current > 48 * 2**30
vae = json.loads(Path(recipe['vae_pool']).read_text())

def health(worker):
    with urllib.request.urlopen(worker['url'] + '/health', timeout=15) as stream:
        value = json.load(stream)
    assert value['status'] == 'ready', worker['name']
    assert value['domain'] == 'ceb_query_runtime'
    return dict(name=worker['name'], health=value)

with concurrent.futures.ThreadPoolExecutor(max_workers=26) as pool:
    checks = list(pool.map(health, recipe['workers'] + vae['workers']))
endpoint_locks = []
for worker in recipe['workers']:
    endpoint = '127.0.0.1:%s:imdb' % worker['db_port']
    path = Path('/tmp') / ('ceb_baseline_db_' + hashlib.sha256(endpoint.encode()).hexdigest()[:16] + '.lock')
    stream = path.open('a')
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    endpoint_locks.append(stream)
audit = dict(at=time.time(), host=socket.gethostname(), memory_current=current,
             memory_max=maximum, recipe_sha256=hashlib.sha256((campaign/'recipe.json').read_bytes()).hexdigest(),
             checks=checks, workers=[])
(campaign / 'preflight.json').write_text(json.dumps(audit, indent=2) + '\n')
for stream in endpoint_locks:
    stream.close()
logs = campaign / 'logs'
logs.mkdir(exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
for worker in plan['workers']:
    logfile = logs / (worker['name'] + '.log')
    command = ['taskset', '-c', str(worker['cpus'][0])] + worker['command']
    with logfile.open('ab', buffering=0) as output:
        child = subprocess.Popen(command, cwd='/root/data/cys/llmoptimizer_benchmark',
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
    entry = dict(worker, pid=child.pid, actual_command=command, log=str(logfile),
                 started_at=time.time(), process_start_ticks=Path('/proc/%s/stat' % child.pid).read_text().split()[21])
    audit['workers'].append(entry)
    (campaign / 'launch.json').write_text(json.dumps(audit, indent=2) + '\n')
plan['launched'] = True
plan['launched_at'] = time.time()
(campaign / 'worker_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(launched=len(audit['workers']), workers=[dict(name=w['name'], pid=w['pid']) for w in audit['workers']])))
