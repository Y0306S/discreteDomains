"""Audit recovery receipts, committed data, settings and live worker identities."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zlib

import psycopg

ROOT=Path(__file__).resolve().parent
recipe=json.loads((ROOT/'recipe.json').read_text())
current=json.loads((ROOT/'current_workers.json').read_text())
workers=[]
for worker in current['workers']:
    proc=Path('/proc')/str(worker['pid'])
    alive=proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z'
    same_identity=alive and proc.joinpath('stat').read_text().split()[21]==worker['process_start_ticks']
    with psycopg.connect(host='127.0.0.1',port=worker['db_port'],dbname='imdb',user='imdb',
                         password='imdb',autocommit=True,connect_timeout=5) as connection:
        connection.execute('SET statement_timeout=5000')
        setting=connection.execute('SHOW temp_tablespaces').fetchone()[0]
        expected='fsbo_tmp_'+worker['name']
        assert setting==expected
    text=Path(worker['log']).read_text()
    workers.append(dict(worker=worker['name'],pid=worker['pid'],alive=alive,
                        same_identity=same_identity,temp_tablespaces=setting,
                        traceback='Traceback (most recent call last)' in text))
recoveries=[]
for path in sorted((ROOT/'recovery').glob('*_proposal_*.json')):
    evidence=json.loads(path.read_text())
    run=ROOT/'runs'/evidence['task']
    with sqlite3.connect('file:'+str(run/'artifacts.sqlite3')+'?mode=ro',uri=True,timeout=5) as db:
        rows={name:json.loads(zlib.decompress(payload)) for name,payload in db.execute('SELECT name,payload FROM artifacts')}
    preserved=all(name in rows and hashlib.sha256(json.dumps(rows[name],sort_keys=True).encode()).hexdigest()==digest
                  for name,digest in evidence['committed_result_hashes'].items())
    prefix='proposal_%04d'%evidence['proposal_index']
    attempts=rows[prefix+'.database_attempts.json']
    result=rows.get(prefix+'.json')
    assert attempts[0]['status']=='error' and attempts[0]['sqlstate']=='53100'
    if result:
        assert result['proposal']['timeout']==50
        assert len(attempts)==2 and attempts[1]['status']=='returned'
        assert result['status'] in ['completed','timeout','query_resource_error']
        assert result.get('sqlstate')!='53100'
    recoveries.append(dict(task=evidence['task'],proposal_index=evidence['proposal_index'],
        original_error_retained=True,prior_committed_results=len(evidence['committed_result_hashes']),
        committed_results_unchanged=preserved,recovered=bool(result),
        attempt_statuses=[x['status'] for x in attempts],
        outcome={k:result.get(k) for k in ['status','runtime','censored','database_attempts']} if result else None))
statuses=[]
for path in sorted((ROOT/'runs').glob('*/status.json')):
    status=json.loads(path.read_text())
    statuses.append({k:status.get(k) for k in ['task_id','status','proposals','database_calls','unresolved_database_calls','error']})
pins=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected for path,expected in recipe['pinned_files'].items())
def free(path):
    fs=os.statvfs(path)
    return round(fs.f_bavail*fs.f_frsize/2**30,2)
passed=(len(workers)==16 and all(w['alive'] and w['same_identity'] and not w['traceback'] for w in workers)
        and len(recoveries)==7 and all(r['recovered'] and r['committed_results_unchanged'] for r in recoveries)
        and not any(s['status']=='failed' for s in statuses) and pins)
result=dict(at=time.time(),passed=passed,pinned_assets_unchanged=pins,
    query_plan_timeout_seconds=50,update_e2e=False,recenter=False,
    workers_alive=sum(w['alive'] and w['same_identity'] for w in workers),
    recovered_errors=sum(r['recovered'] for r in recoveries),workers=workers,recoveries=recoveries,
    task_statuses=statuses,disk_free_gib=free(ROOT),shm_free_gib=free('/dev/shm'),
    memory_current=int(Path('/sys/fs/cgroup/memory.current').read_text()),
    memory_max=int(Path('/sys/fs/cgroup/memory.max').read_text()))
(ROOT/'recovery_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['at','passed','workers_alive','recovered_errors','pinned_assets_unchanged','disk_free_gib','shm_free_gib']}))
for row in recoveries:
    print(json.dumps(row))
