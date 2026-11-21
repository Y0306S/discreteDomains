"""Stop CPU fits only where no durable proposal/oracle event is pending."""
import argparse,json,os,signal,sqlite3,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
C=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')

def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def snapshot():
 rows={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   a=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v];s=(p/'stat').read_text().rsplit(')',1)[1].split()
   if s[0]=='Z':continue
   rows[int(p.name)]=dict(args=a,ppid=int(s[1]),start=s[19])
  except (OSError,ValueError):pass
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--plan',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text());validation=json.loads((ROOT/'hybrid_validation.json').read_text())
 if not validation['passed']:raise ValueError('Hybrid validation did not pass')
 if os.uname().nodename!=plan['hosts'][a.host]['hostname']:raise ValueError('Wrong host')
 old=json.loads((C/'dispatch'/(a.host+'.json')).read_text());ps=snapshot();controller=ps.get(old['pid'])
 if not controller or not any(v.endswith('/repair_dispatch.py') for v in controller['args']):raise ValueError('Unexpected controller')
 records=[];backups=ROOT/'before_migration'/a.host;backups.mkdir(parents=True,exist_ok=True)
 atomic(backups/'dispatch.json',old)
 os.kill(old['pid'],signal.SIGSTOP)
 try:
  for rec in old['active']:
   pid=rec['pid'];row=ps.get(pid)
   if not row or row['start']!=rec['start']:raise ValueError('Worker identity changed: '+rec['task'])
   os.kill(pid,signal.SIGSTOP)
   run=C/'runs'/rec['task'];dbpath=run/'runtime.sqlite'
   with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True,timeout=5) as c:
    value=c.execute("select value from meta where key='pending'").fetchone()
   no_pending=value is None or bytes(value[0]) in (b'\x80\x04N.',b'\x80\x03N.',b'\x80\x05N.')
   progress=json.loads((run/'progress.json').read_text())
   if not no_pending:
    os.kill(pid,signal.SIGCONT);os.kill(pid,signal.SIGTERM)
    records.append(dict(task=rec['task'],pid=pid,action='drain_pending_event',before=progress));continue
   # No proposal, score or oracle call exists beyond the committed checkpoint.
   descendants={pid};changed=True
   while changed:
    new={p for p,r in ps.items() if r['ppid'] in descendants}-descendants;changed=bool(new);descendants.update(new)
   for child in descendants-{pid}:
    try:os.kill(child,signal.SIGSTOP)
    except ProcessLookupError:pass
   with sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True,timeout=5) as c,sqlite3.connect(backups/(rec['task']+'.sqlite')) as out:
    deadline=time.monotonic()+10
    def guard(*args):
     if time.monotonic()>deadline:raise TimeoutError('Checkpoint backup exceeded ten seconds')
    c.backup(out,pages=1000,progress=guard)
   for child in descendants:
    try:os.kill(child,signal.SIGKILL)
    except ProcessLookupError:pass
   records.append(dict(task=rec['task'],pid=pid,action='restart_committed_checkpoint',pending=False,before=progress,backup=str(backups/(rec['task']+'.sqlite'))))
  os.kill(old['pid'],signal.SIGKILL)
 except Exception:
  for rec in old['active']:
   try:os.kill(rec['pid'],signal.SIGCONT)
   except ProcessLookupError:pass
  try:os.kill(old['pid'],signal.SIGCONT)
  except ProcessLookupError:pass
  raise
 remaining=[r for r in old['active'] if any(v['pid']==r['pid'] and v['action']=='drain_pending_event' for v in records)]
 old['active']=remaining;old['plan_sha256']=__import__('hashlib').sha256(Path(a.plan).read_bytes()).hexdigest();old['gpu_transition_at']=time.time();atomic(C/'dispatch'/(a.host+'.json'),old)
 atomic(backups/'migration.json',dict(at=time.time(),host=a.host,records=records))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 command=['/usr/bin/python3','-u',str(ROOT/'code/rgpe_dispatch.py'),'--plan',a.plan,'--host',a.host,'--execute']
 with (ROOT/(a.host+'.rgpe_dispatch.log')).open('ab') as log:child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 atomic(ROOT/(a.host+'.migration_launch.json'),dict(pid=child.pid,command=command,at=time.time(),tasks=len(records),draining=len(remaining)))
 print(json.dumps(dict(host=a.host,dispatcher_pid=child.pid,tasks=len(records),draining=len(remaining))))
if __name__=='__main__':main()
