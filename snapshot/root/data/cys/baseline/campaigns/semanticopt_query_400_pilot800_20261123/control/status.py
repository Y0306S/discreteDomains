"""Read-only consolidated progress for the launched query continuations."""
import json,sqlite3,time
from pathlib import Path
BASE=Path('/root/data/cys/baseline/campaigns')
def load(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None
if __name__=='__main__':
 result=dict(at=time.time())
 for name,key in [('semanticopt_query_400_pilot800_20261123','semanticopt'),('query_bolt30_lolbo_total800_20261123','bolt')]:
  root=BASE/name;state=load(root/'status.json');errors=[];count=0;inflight=0
  for p in (root/('query/runs' if key=='semanticopt' else 'runs')).glob('*/error.json'):errors.append(str(p))
  if key=='semanticopt':
   for p in (root/'query/runs').glob('*/observations.sqlite'):
    db=sqlite3.connect(f'file:{p}?mode=ro',uri=True);count+=db.execute('SELECT count(*) FROM records').fetchone()[0]-200
    x=db.execute("SELECT value FROM state WHERE key='pending'").fetchone();x=json.loads(x[0]) if x else None;inflight+=bool(x and x.get('status')=='in_flight');db.close()
  result[key]=dict(root=str(root),status=state,error_files=errors,paused=(root/'PAUSE').exists())
  if key=='semanticopt':result[key].update(new_committed_evaluations=count,in_flight_evaluations=inflight)
 print(json.dumps(result,indent=2))
