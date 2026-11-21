from pathlib import Path
import os,signal,time,sqlite3,json
r=Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110')
workers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if str(r/'code/queue.py').encode() in args or str(r/'code/host.py').encode() in args:
   os.kill(int(p.name),signal.SIGTERM)
  if str(r/'code/evaluate.py').encode() in args:
   i=args.index(str(r/'code/evaluate.py').encode());domain=args[i+1].decode();task=args[i+2].decode()
   env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
   if env.get(b'SEMANTICOPT_YIELD_FILE'):Path(env[b'SEMANTICOPT_YIELD_FILE'].decode()).touch()
   os.kill(int(p.name),signal.SIGSTOP);workers.append((p,domain,task))
 except OSError:pass
# Confirm all workers stopped before inspecting durable oracle states.
for p,domain,task in workers:
 for _ in range(100):
  try:
   if (p/'stat').read_text().rsplit(') ',1)[1].startswith(('T','Z')):break
  except OSError:break
  time.sleep(.01)
 else:raise RuntimeError('Worker did not stop: '+p.name)
report=[]
for p,domain,task in workers:
 dbp=r/domain/'runs'/task/'observations.sqlite';db=sqlite3.connect(dbp,timeout=10)
 pending={k:json.loads(v) for k,v in db.execute("select key,value from state where key in ('pending','oracle_batch')")};db.close()
 inflight=any(v and v.get('status')=='in_flight' for v in pending.values())
 if inflight:
  # Let its real oracle finish and honor the yield flag at the next boundary.
  os.kill(int(p.name),signal.SIGCONT);action='finish_oracle_and_yield'
 else:os.kill(int(p.name),signal.SIGKILL);action='restart_from_checkpoint'
 report.append(dict(pid=int(p.name),task=task,domain=domain,action=action))
print(json.dumps(report),flush=True)
