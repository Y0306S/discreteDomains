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
  if key=='bolt':
   extra=load(root/'control/extra_controller.status.json')
   if extra:
    alive=False
    try:
     proc=Path('/proc')/str(extra['supervisor_pid'])
     alive=(proc/'stat').read_text().split()[21]==extra['supervisor_start_ticks'] and str(root/'control/extra_controller.py').encode() in (proc/'cmdline').read_bytes()
    except (OSError,KeyError):pass
    fresh=time.time()-extra.get('at',0)<30
    additional=extra.get('live_worker_slots',0) if alive and fresh else 0
    state['primary_controller_live_worker_slots']=state.get('live_worker_slots',0)
    state['live_worker_slots']=state['primary_controller_live_worker_slots']+additional
    result[key]['extra_worker']=dict(extra,supervisor_alive=alive,fresh=fresh)
    error=root/'control/extra_controller.error.json'
    if error.exists():errors.append(str(error))
    if extra.get('status')=='failed' or (extra.get('status')=='running' and not (alive and fresh)):
     state['status']='failed';state['failures']=state.get('failures',[])+[dict(slot=7,error='Additional worker supervisor failed or stale')]
  if key=='semanticopt':result[key].update(new_committed_evaluations=count,in_flight_evaluations=inflight)
 print(json.dumps(result,indent=2))
