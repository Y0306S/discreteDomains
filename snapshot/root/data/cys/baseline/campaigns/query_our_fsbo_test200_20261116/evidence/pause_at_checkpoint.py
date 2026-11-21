import os,signal,time,json,sqlite3,pickle
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'PAUSE').touch()
workers={}
for p in (ROOT/'services').glob('worker*.process.json'):
 d=json.loads(p.read_text());pid=d['pid']
 try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:continue
 if str(ROOT/'code/run.py').encode() in cmd and b'worker' in cmd:
  status=json.loads((ROOT/'services'/p.name.replace('.process.json','.status.json')).read_text())
  assert status['pid']==pid and status['stage']=='warm_start'
  workers[pid]=dict(task=status['task'],path=str(ROOT/'warm_start/runs'/status['task']/'runtime.sqlite'))
report=dict(started=time.time(),workers={},existing_controller_stopped=True)
deadline=time.monotonic()+180
while workers and time.monotonic()<deadline:
 for pid,meta in list(workers.items()):
  try:os.kill(pid,signal.SIGSTOP)
  except ProcessLookupError:
   report['workers'][str(pid)]=dict(meta,status='already_exited');del workers[pid];continue
  # Wait for the kernel to stop the coordinator before examining its journal.
  for _ in range(100):
   try:state=Path(f'/proc/{pid}/status').read_text().split('State:',1)[1].splitlines()[0]
   except FileNotFoundError:state='T'
   if 'T' in state:break
   time.sleep(.001)
  try:
   conn=sqlite3.connect(meta['path'],timeout=.1)
   rows={k:pickle.loads(v) for k,v in conn.execute('SELECT key,value FROM kv')};conn.close()
   pending=[k for k,v in rows.items() if k.startswith('measurement/') and v['status']=='started']
   if pending:
    os.kill(pid,signal.SIGCONT);continue
   measured=[k for k,v in rows.items() if k.startswith('measurement/') and v['status']=='complete']
   report['workers'][str(pid)]=dict(meta,status='paused_at_durable_boundary',completed_measurements=len(measured),saved_warm_rows=len(rows.get('warm_state',{}).get('rows',[])),at=time.time())
   # No SQL is in flight. Inference requests are independently durable at the broker.
   os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
   print(pid,meta['task'],'paused;',len(measured),'saved SQL results',flush=True);del workers[pid]
  except sqlite3.OperationalError:os.kill(pid,signal.SIGCONT)
 time.sleep(.2)
if workers:
 for pid in workers:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
 report['pending']=workers
else:report['pending']={}
report['completed_at']=time.time()
(ROOT/'evidence/pause_checkpoint_report.json').write_text(json.dumps(report,indent=2))
status=json.loads((ROOT/'status.json').read_text());status.update(status='paused_for_schedule_discussion',paused=True,worker_processes=list(workers),at=time.time(),pause_report=str(ROOT/'evidence/pause_checkpoint_report.json'))
(ROOT/'status.json').write_text(json.dumps(status,indent=2))
if workers:raise RuntimeError('Some workers still need checkpoint pause')
print('All Query CPU workers safely paused; GPU services remain loaded.',flush=True)
