import hashlib,json,pickle,sqlite3,time
from pathlib import Path
import fleet
root=fleet.ROOTS['bolt'];folder=root/'runs/CEB_10A32';archive=fleet.OPS/'decode_recovery_10A32'
audit=fleet.load(archive/'audit.json')
with sqlite3.connect('file:'+str(folder/'runtime.sqlite')+'?mode=ro',uri=True,timeout=30) as db:
 for key,want in audit['measurement_hashes'].items():
  value=db.execute('select value from kv where key=?',(key,)).fetchone()[0]
  assert hashlib.sha256(value).hexdigest()==want,key
 state=pickle.loads(db.execute("select value from kv where key='lolbo_state'").fetchone()[0])
 pending=[k for k,v in db.execute("select key,value from kv where key like 'measurement/%'") if pickle.loads(v).get('status') not in ('complete','safe_failed')]
workers=[fleet.load(p) for p in (root/'services').glob('worker*.status.json')]
w=next(s for s in workers if s.get('task')=='CEB_10A32' and s.get('status')=='running')
proposal=fleet.load(folder/'proposal_progress.json')
assert proposal.get('recovery') and time.time()-proposal['at']<120
assert (folder/'proposal_recovery_066.pickle').exists()
for kind,root2 in fleet.ROOTS.items():
 status=fleet.load(root2/'status.json');assert status['live_worker_slots']==(20 if kind=='semanticopt' else 8),status
 assert not status['failures'] and not status['paused_tasks'],status
result=dict(original_measurements_unchanged=True,original_measurements=len(audit['measurement_hashes']),rows=len(state['rows']),completed_rounds=len(state['rounds']),worker=w,proposal={k:v for k,v in proposal.items() if k!='last_rejections'},pending_sql=pending,all_28_slots_running=True)
fleet.atomic(archive/'verification.json',result)
print(json.dumps(result,indent=2))
