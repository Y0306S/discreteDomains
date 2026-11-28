"""Check retained prefixes and unfinished SQL markers before continuation."""
import hashlib
import pickle
import sqlite3
from common import *

original=load(ROOT/'snapshots/initial/manifest.json')['records']
drained=load(ROOT/'snapshots/drained/manifest.json')['records']
lookup={(r['kind'],r['task']):r for r in drained}
uncertain=[];deferred_uncertain=[];checked=0
for row in original:
    new=lookup[(row['kind'],row['task'])]
    with sqlite3.connect(row['backup']) as old,sqlite3.connect(new['backup']) as db:
        if row['kind']=='semanticopt':
            before=old.execute('SELECT n,value FROM records ORDER BY n').fetchall()
            after=db.execute('SELECT n,value FROM records ORDER BY n LIMIT ?',(len(before),)).fetchall()
            assert before==after, row['task']
            for k,v in db.execute('SELECT key,value FROM state'):
                if k in ('in_flight','pending') and json.loads(v) is not None:
                    previous=old.execute('SELECT value FROM state WHERE key=?',(k,)).fetchone()
                    pause=load(SEM/'task_pauses'/f'{row["task"]}.json',{})
                    if previous and previous[0]==v and pause.get('requires_audit_before_retry'):
                        deferred_uncertain.append((row['kind'],row['task'],k))
                    else:uncertain.append((row['kind'],row['task'],k))
        else:
            for k,v in old.execute('SELECT key,value FROM kv'):
                if k.startswith('measurement/'):
                    val=pickle.loads(v)
                    if val.get('status')=='complete':
                        assert db.execute('SELECT value FROM kv WHERE key=?',(k,)).fetchone()[0]==v
            for k,v in db.execute('SELECT key,value FROM kv'):
                if k.startswith('measurement/') and pickle.loads(v).get('status')=='started':
                    uncertain.append((row['kind'],row['task'],k))
    checked+=1
result=dict(checked=checked,retained_prefixes_unchanged=True,uncertain_measurements=uncertain,
            existing_deferred_uncertain_measurements=deferred_uncertain,at=time.time())
atomic(ROOT/'saved_state_audit.json',result);print(json.dumps(result))
assert not uncertain, 'Uncertain measurements must be audited before admission'
