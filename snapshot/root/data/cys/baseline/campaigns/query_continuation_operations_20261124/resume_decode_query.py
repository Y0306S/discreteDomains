"""Run on LLMOptimizer3; safely yield one slot and resume the audited query."""
import hashlib,json,os,pickle,shutil,signal,sqlite3,time
from pathlib import Path
import fleet

root=fleet.ROOTS['bolt'];target='CEB_10A32';folder=root/'runs'/target
lease=fleet.lock(folder/'task.lock')
archive=fleet.OPS/'decode_recovery_10A32';archive.mkdir(exist_ok=True)
with sqlite3.connect('file:'+str(folder/'runtime.sqlite')+'?mode=ro',uri=True) as db:
 records=dict(db.execute('select key,value from kv'))
 state=pickle.loads(records['lolbo_state'])
 assert len(state['rows'])==680 and len(state['rounds'])==65
 assert not any(k.startswith('measurement/66/') for k in records)
 assert all(pickle.loads(v).get('status')=='complete' for k,v in records.items() if k.startswith('measurement/'))
 hashes={k:hashlib.sha256(v).hexdigest() for k,v in records.items() if k.startswith('measurement/')}
fleet.atomic(archive/'audit.json',dict(task=target,rows=680,completed_rounds=65,unresolved_sql=[],measurement_hashes=hashes))
for path in [root/'task_pauses'/f'{target}.json',folder/'error.json',folder/'proposal_progress.json']:
 if path.exists():
  dest=archive/path.name
  if dest.exists():dest=archive/('old_'+path.parent.name+'_'+path.name)
  shutil.copy2(path,dest)
# Release only the audited proposal failure. This is not a measurement replay.
(root/'task_pauses'/f'{target}.json').unlink()
(folder/'error.json').unlink(missing_ok=True)
lease.close()
# Yield slot 4 only while its task is between SQL measurements. The old task stays resumable.
number=4;w=fleet.load(fleet.statusfile('bolt',number));other=w['task'];pid=w['child_pid']
if other==target:
 print('Target already resumed');raise SystemExit(0)
expected_ticks=fleet.process_ticks(pid)
assert expected_ticks and str(fleet.OPS/'fleet.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
spec=next(s for s in fleet.configuration('bolt')[0] if s['slot']==number)
for attempt in range(90):
 if fleet.process_ticks(pid)!=expected_ticks:
  print('Previous task ended naturally; target is eligible');break
 os.kill(pid,signal.SIGSTOP)
 safe=False
 try:
  dbpath=root/'runs'/other/'runtime.sqlite'
  if dbpath.exists():
   with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True,timeout=.2) as db:
    values=list(db.execute("select key,value from kv where key like 'measurement/%'"))
   pending=[k for k,v in values if pickle.loads(v).get('status') not in ('complete','safe_failed')]
  else:pending=[]
  if not pending:
   fleet.database_check(spec)
   safe=True
 except Exception:
  safe=False
 if safe:
  fleet.atomic(archive/'yielded_slot.json',dict(slot=number,task=other,pid=pid,start_ticks=expected_ticks,unresolved_sql=[],reason='Yield at measurement boundary to resume CEB_10A32; task remains checkpointed'))
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  print('Safely yielded slot',number,'from',other,'to',target,flush=True)
  break
 os.kill(pid,signal.SIGCONT)
 time.sleep(1)
else:raise RuntimeError('Could not find safe SQL boundary; target remains queued')
