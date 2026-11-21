"""Snapshot production identities, task ownership, and newly committed evidence."""
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import subprocess
import time
import zlib

root = Path(__file__).resolve().parent
launched = json.loads((root/'production_processes.json').read_text())
campaign = Path(json.loads((root/'reservation.json').read_text())['campaign'])
recipe_sha = hashlib.sha256((campaign/'recipe.json').read_bytes()).hexdigest()
assert recipe_sha == launched['recipe_sha256']
workers = []
for entry in launched['workers']:
    pid = entry['pid']
    proc = Path('/proc')/str(pid)
    stat = (proc/'stat').read_text().split()
    assert stat[2] != 'Z' and stat[21] == entry['process_start_ticks']
    assert b'run_worker.py' in (proc/'cmdline').read_bytes()
    assert sorted(os.sched_getaffinity(pid)) == [entry['optimizer_cpu']]
    state = json.loads((root/'workers'/entry['name']/'production.json').read_text())
    assert state['pid'] == pid and state['status'] not in ['failed', 'stopped', 'finished']
    assert state['recipe_sha256'] == recipe_sha
    assert 'Traceback (most recent call last)' not in Path(entry['log']).read_text()
    locks = []
    for fd in (proc/'fd').iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if str(campaign/'locks')+'/' in target:
            locks.append(Path(target).stem)
    task = state.get('task')
    directory = campaign/'runs'/str(task)
    proposals = 0
    recent = []
    if (directory/'artifacts.sqlite3').exists():
        connection = sqlite3.connect((directory/'artifacts.sqlite3').as_uri()+'?mode=ro', uri=True, timeout=5)
        names = [r[0] for r in connection.execute("SELECT name FROM artifacts WHERE name GLOB 'proposal_[0-9][0-9][0-9][0-9].json' ORDER BY name")]
        proposals = len(names)
        for name in names[-10:]:
            payload, checksum = connection.execute('SELECT payload,sha256 FROM artifacts WHERE name=?', (name,)).fetchone()
            raw = zlib.decompress(payload)
            assert hashlib.sha256(raw).hexdigest() == checksum
            row = json.loads(raw)
            assert row['proposal']['timeout'] == 50
            for attempt in row.get('database_attempt_history', []):
                if attempt.get('started_at', 0) >= entry['started_at'] and attempt.get('finished_at'):
                    assert attempt['status'] == 'returned'
                    recent.append(dict(proposal=name, status=row['status'], runtime=row['runtime'],
                                       started_at=attempt['started_at'], finished_at=attempt['finished_at']))
        connection.close()
    workers.append(dict(name=entry['name'], pid=pid, alive=True, cpus=entry['cpus'],
                        task=task, state=state['status'], task_locks=locks, proposals=proposals,
                        new_database_results=recent, progressing=bool(recent) and locks == [task]))

remote_code = '''from pathlib import Path
import json
campaign=Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1')
result=[]
for w in json.loads((campaign/'current_workers.json').read_text())['workers']:
    if w['host']!='LLMOptimizer4':continue
    p=Path('/proc')/str(w['pid'])
    try:
        stat=(p/'stat').read_text().split()
        valid=stat[2]!='Z' and b'query_fsbo_eval.py' in (p/'cmdline').read_bytes()
        if w.get('process_start_ticks'):valid=valid and str(w['process_start_ticks'])==stat[21]
    except OSError:valid=False
    result.append(dict(name=w['name'],pid=w['pid'],alive=valid))
print(json.dumps(result))
'''
remote = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                         'LLMOptimizer4', 'python3 -'], input=remote_code,
                        capture_output=True, text=True, timeout=20, check=True)
remote_workers = json.loads(remote.stdout)
assert len(remote_workers) == 16 and all(w['alive'] for w in remote_workers)
ledger = json.loads((campaign/'current_workers.json').read_text())
assert len(ledger['workers']) == 20 and len({w['name'] for w in ledger['workers']}) == 20
assert len({w['task'] for w in workers}) == 4
health = json.loads((root/'vae_health_at_expansion.json').read_text())
assert len(health['workers']) == 10
for service in health['workers']:
    assert service['reachable'] and service['health']['errors'] == 0
    assert b'query_vae_service.py' in (Path('/proc')/str(service['health']['pid'])/'cmdline').read_bytes()
counts = {}
for path in (campaign/'runs').glob('*/status.json'):
    status = json.loads(path.read_text())['status']
    counts[status] = counts.get(status, 0)+1
assert not counts.get('failed')
cg = Path('/sys/fs/cgroup')
free_gib = (int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text()))/2**30
assert free_gib >= 232
value = dict(at=time.time(), passed=all(w['progressing'] for w in workers),
             workers=workers, remote_workers=remote_workers, total_cpu_workers=20,
             shared_vae_services=10, campaign_status_counts=counts,
             available_memory_gib=free_gib, recipe_sha256=recipe_sha,
             peptide_tpe_launched=False)
(root/'production_verification.json').write_text(json.dumps(value, indent=2)+'\n')
print(json.dumps(value, indent=2))
