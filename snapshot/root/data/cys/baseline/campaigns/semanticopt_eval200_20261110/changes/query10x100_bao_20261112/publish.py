from pathlib import Path
import os,signal,subprocess,json,time,hashlib,shutil,sys,shlex
ROOT=Path(__file__).resolve().parents[2];E=Path(__file__).resolve().parent;S=E/'staged'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.publish.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
def command(pid):return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
def remote_signal(sig):
 code=f"import os,signal;from pathlib import Path;p=45918;cmd=(Path('/proc')/str(p)/'cmdline').read_bytes();assert {str(ROOT/'code/host.py').encode()!r} in cmd and b'--domain\\x00peptide' in cmd;os.kill(p,signal.{sig})"
 subprocess.run(['ssh','-o','BatchMode=yes','LLMOptimizer4',shlex.join(['/usr/bin/python3','-c',code])],check=True,timeout=20)
assert json.loads((E/'prompt_constraint_audit.json').read_text())['passed']
assert json.loads((E/'query_checkpoint_audit.json').read_text())['safe_to_resume']
assert str(ROOT/'code/queue.py') in command(60416)
assert str(ROOT/'code/host.py') in command(60501)
assert not (ROOT/'PAUSE').exists()
changes=['code/common.py','code/batch.py','code/evaluate.py','code/host.py','code/queue.py','code/batch_probe.py','code/query_progress.py','code/test_query_policy.py','query_policy.json']
identity=json.loads((ROOT/'identity.json').read_text());before={name:sha(ROOT/name) if (ROOT/name).exists() else None for name in changes}
# Suspend admission briefly so no new worker can verify midway through publication.
local_stopped=False;remote_stopped=False;published=False
try:
 os.kill(60416,signal.SIGSTOP);os.kill(60501,signal.SIGSTOP);local_stopped=True
 remote_signal('SIGSTOP');remote_stopped=True
 time.sleep(10)
 for name in changes:
  target=ROOT/name;tmp=target.with_name(target.name+'.publish.tmp');shutil.copy2(S/name,tmp);os.replace(tmp,target)
  identity['files'][str(target)]=sha(target)
 audit=E/'prompt_constraint_audit.json';identity['files'][str(audit)]=sha(audit)
 identity['query10x100_bao_amendment']=dict(at=time.time(),authorization='User requested query 10 x 100 and Bao initializations before optimization',evidence=str(E),files={n:dict(before=before[n],after=sha(ROOT/n)) for n in changes})
 atomic(ROOT/'identity.json',identity);published=True
 sys.path.insert(0,str(ROOT/'code'))
 from common import verify,domain_recipe
 recipe=verify();q=domain_recipe(recipe,'query');p=domain_recipe(recipe,'peptide')
 assert (q['batch_policy']['size'],q['query']['rounds'],q['query']['proposals'])==(10,100,1000)
 assert (p['batch_policy']['size'],p['peptide']['rounds'],p['peptide']['proposals'])==(50,20,1000)
 assert p['batch_policy_sha256']=='558fa7e5bcfc38f31284bb2373ae492a25be8dec788adbe0c8e382555be8919b'
 assert p['budget_policy_sha256']=='bbc88eca338dd9b084b68dda8401bc8d411c9851653796f1b93bf536695aefb2'
 atomic(ROOT/'effective_recipe.json',recipe)
 from query_progress import update_progress
 progress=update_progress(ROOT,recipe,force=True)
 assert progress['batch_size']==10 and progress['rounds']==100 and progress['status']=='queued'
 # Replace only the queue; its children retain their sessions and are adopted.
 os.kill(60416,signal.SIGTERM);os.kill(60416,signal.SIGCONT)
 for _ in range(50):
  stat=Path('/proc/60416/stat')
  if not stat.exists() or stat.read_text().rsplit(') ',1)[1].startswith('Z'):break
  time.sleep(.1)
 else:raise RuntimeError('Old queue did not exit')
 args=['/usr/bin/python3','-u',str(ROOT/'code/queue.py')]
 with (ROOT/'logs/queue.query10x100.log').open('ab') as log:
  queue=subprocess.Popen(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 ticks=(Path('/proc')/str(queue.pid)/'stat').read_text().rsplit(') ',1)[1].split()[19]
 atomic(ROOT/'queue_process.json',dict(pid=queue.pid,started_at=time.time(),command=args,process_start_ticks=ticks,budget_policy_sha256=p['budget_policy_sha256'],query_policy_sha256=recipe['query_policy_sha256'],adopts_existing_peptide_controllers=True))
 atomic(E/'publication.json',dict(published=True,at=time.time(),queue_pid=queue.pid,peptide_controllers_unchanged=[60501,45918],query_policy_sha256=recipe['query_policy_sha256'],query_batch=10,query_rounds=100,peptide_batch=50,peptide_rounds=20))
 print('Published; queue PID',queue.pid,flush=True)
finally:
 if remote_stopped:remote_signal('SIGCONT')
 if local_stopped:os.kill(60501,signal.SIGCONT)
 if not published:
  for name in changes:
   saved=E/'before'/name
   if saved.exists():shutil.copy2(saved,ROOT/name)
   elif before[name] is None:(ROOT/name).unlink(missing_ok=True)
  shutil.copy2(E/'before/identity.json',ROOT/'identity.json')
 try:os.kill(60416,signal.SIGCONT)
 except ProcessLookupError:pass
