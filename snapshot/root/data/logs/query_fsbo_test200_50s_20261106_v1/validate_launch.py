"""Read-only launch snapshot; safe to rerun while the campaign progresses."""
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import sqlite3
import time
import urllib.request
import zlib

p = Path(__file__).resolve().parent
launch = json.loads((p/'launch.json').read_text())
if (p/'current_workers.json').exists():
    launch = json.loads((p/'current_workers.json').read_text())
recipe = json.loads((p/'recipe.json').read_text())
workers = []
errors = []
for w in launch['workers']:
    proc = Path('/proc')/str(w['pid'])
    alive = proc.exists() and proc.joinpath('stat').read_text().split()[2] != 'Z'
    text = Path(w['log']).read_text()
    if 'Traceback (most recent call last)' in text:
        errors.append(dict(worker=w['name'], log=w['log'], tail=text[-4000:]))
    children = proc.joinpath('task',str(w['pid']),'children').read_text().split() if alive else []
    workers.append(dict(name=w['name'], pid=w['pid'], alive=alive,
                        affinity=sorted(os.sched_getaffinity(w['pid'])) if alive else [],
                        children=[int(x) for x in children], log=w['log']))
tasks = []
for directory in sorted((p/'runs').iterdir()):
    if not directory.is_dir():
        continue
    results, pending = [], []
    dbpath = directory/'artifacts.sqlite3'
    if dbpath.exists():
        with sqlite3.connect('file:'+str(dbpath)+'?mode=ro', uri=True, timeout=3) as db:
            for name, payload in db.execute('SELECT name,payload FROM artifacts'):
                if name.startswith('proposal_') and name.endswith('.json'):
                    data = json.loads(zlib.decompress(payload))
                    if name.endswith('.pending.json'):
                        pending.append(data)
                    elif not name.endswith('.database_attempts.json'):
                        results.append(data)
    status = json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else None
    if status and status['status'] == 'failed':
        errors.append(dict(task=directory.name,status=status))
    timeouts = sorted(set([x['proposal']['timeout'] for x in results]+[x['timeout'] for x in pending]))
    assert all(value == 50 for value in timeouts)
    tasks.append(dict(task=directory.name, checkpoint=(directory/'checkpoint.pt').exists(),
                      status=status['status'] if status else 'initial_fit_or_first_batch',
                      committed_proposals=len(results), database_calls=sum(x.get('database_call',False) for x in results),
                      cache_hits=sum(not x.get('database_call',False) for x in results),
                      censored_results=sum(x['censored'] for x in results),
                      pending_database_calls=len(pending), proposal_timeouts=timeouts))
pool = json.loads(Path(recipe['vae_pool']).read_text())
def health(w):
    with urllib.request.urlopen(w['url']+'/health',timeout=15) as stream:
        value = json.load(stream)
    return dict(name=w['name'],status=value['status'],requests=value['requests'],errors=value['errors'])
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    vae = list(executor.map(health,pool['workers']))
result = dict(at=time.time(),host=socket.gethostname(),campaign=str(p),query_plan_timeout_seconds=50,
              update_e2e=False,recenter=False,workers=workers,tasks=tasks,vae_workers=vae,errors=errors,
              workers_alive=sum(w['alive'] for w in workers),tasks_admitted=len(tasks),
              committed_proposals=sum(t['committed_proposals'] for t in tasks),
              database_calls=sum(t['database_calls'] for t in tasks),
              memory_current=int(Path('/sys/fs/cgroup/memory.current').read_text()),
              memory_max=int(Path('/sys/fs/cgroup/memory.max').read_text()))
(p/'launch_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['at','workers_alive','tasks_admitted','committed_proposals','database_calls','errors','memory_current']}))
assert result['workers_alive'] == 16 and len(tasks) == 16 and not errors
assert all(w['status']=='ready' for w in vae)
assert result['database_calls'] > 0
