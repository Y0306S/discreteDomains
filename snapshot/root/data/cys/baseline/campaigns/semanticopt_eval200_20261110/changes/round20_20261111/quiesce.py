import pathlib,os,signal,time,sqlite3,json,socket
r=pathlib.Path('/root/data/cys/baseline/campaigns/semanticopt_eval200_20261110');workers=[];stopped=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if not args or b'python' not in args[0]:continue
  if any(str(r/'code'/name).encode() in args for name in ['queue.py','host.py']):
   os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
  if str(r/'code/evaluate.py').encode() in args:
   i=args.index(str(r/'code/evaluate.py').encode());domain=args[i+1].decode();task=args[i+2].decode()
   env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
   if env.get(b'SEMANTICOPT_YIELD_FILE'):pathlib.Path(env[b'SEMANTICOPT_YIELD_FILE'].decode()).touch()
   os.kill(int(p.name),signal.SIGSTOP);workers.append((p,domain,task))
 except ProcessLookupError:pass
report=[]
for p,domain,task in workers:
 try:
  for _ in range(100):
   if (p/'stat').read_text().rsplit(') ',1)[1].startswith(('T','Z')):break
   time.sleep(.01)
  else:raise RuntimeError('Worker did not stop: '+p.name)
  try:
   db=sqlite3.connect('file:'+str(r/domain/'runs'/task/'observations.sqlite')+'?mode=ro',uri=True,timeout=.2)
   state={k:json.loads(v) for k,v in db.execute("select key,value from state where key in ('pending','oracle_batch')")};db.close()
  except sqlite3.Error:
   os.kill(int(p.name),signal.SIGCONT);report.append(dict(pid=int(p.name),task=task,action='finish_and_yield_sqlite_busy'));continue
  if any(v and v.get('status')=='in_flight' for v in state.values()):
   os.kill(int(p.name),signal.SIGCONT);action='finish_oracle_and_yield'
  else:os.kill(int(p.name),signal.SIGKILL);action='restart_from_durable_checkpoint'
  report.append(dict(pid=int(p.name),domain=domain,task=task,action=action))
 except FileNotFoundError:pass
 except BaseException:
  try:os.kill(int(p.name),signal.SIGCONT)
  except ProcessLookupError:pass
  raise
out=dict(host=socket.gethostname(),at=time.time(),controllers_stopped=stopped,workers=report)
(r/'changes/round20_20261111'/('quiesce_'+socket.gethostname()+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
