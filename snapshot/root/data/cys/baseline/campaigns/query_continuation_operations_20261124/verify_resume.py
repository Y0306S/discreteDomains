import hashlib,json,sqlite3,time
from pathlib import Path
import fleet
results=[]
for kind,root in fleet.ROOTS.items():
 audit=fleet.load(fleet.OPS/f'{kind}_before_resume.json')
 previous=0;current=0
 for record in audit['records']:
  folder=fleet.directory(kind,record['task'])
  dbpath=folder/('observations.sqlite' if kind=='semanticopt' else 'runtime.sqlite')
  if kind=='semanticopt':
   with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True,timeout=10) as db:
    rows=db.execute('select n,value from records order by n limit ?', (record['count'],)).fetchall()
    h=hashlib.sha256()
    for n,value in rows:h.update(str(n).encode()+b'\0'+value.encode()+b'\0')
    assert h.hexdigest()==record['prefix_sha256'],(kind,record['task'])
    current+=db.execute('select count(*) from records').fetchone()[0]
   previous+=record['count']
  else:
   if dbpath.exists():
    with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True,timeout=10) as db:
     for key,want in record.get('keys_sha256',{}).items():
      value=db.execute('select value from kv where key=?',(key,)).fetchone()[0]
      assert hashlib.sha256(value).hexdigest()==want,(kind,record['task'],key)
     current+=db.execute("select count(*) from kv where key like 'measurement/%'").fetchone()[0]
   previous+=record.get('completed_new_measurements',0)
  
 status=fleet.load(root/'status.json')
 assert status['live_worker_slots']==(20 if kind=='semanticopt' else 8),status
 assert status['status']=='running' and not status['failures'] and not status['paused_tasks'],status
 assert not (root/'PAUSE').exists()
 results.append(dict(kind=kind,previous_measurements_unchanged=True,previous_count=previous,current_count=current,
                     added_since_resume=current-previous,status=status))
fleet.atomic(fleet.OPS/'verified_resume.json',dict(results=results))
print(json.dumps([{k:v for k,v in r.items() if k!='status'} for r in results],indent=2))
