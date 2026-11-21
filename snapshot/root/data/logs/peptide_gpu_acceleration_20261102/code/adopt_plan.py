import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
C=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe/dispatch')
host=sys.argv[1];plan=Path(sys.argv[2]);p=C/(host+'.json');d=json.loads(p.read_text());pid=d['pid']
a=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
if not any(v.endswith(b'/rgpe_dispatch.py') for v in a):raise ValueError('Unexpected controller')
os.kill(pid,signal.SIGSTOP)
try:
 d=json.loads(p.read_text());children=[]
 for x in Path('/proc').iterdir():
  if not x.name.isdigit():continue
  try:
   s=(x/'stat').read_text().rsplit(')',1)[1].split();args=(x/'cmdline').read_bytes().split(b'\0')
   if int(s[1])==pid and s[0]!='Z' and any(v.endswith(b'/rgpe_launch.py') for v in args):children.append(int(x.name))
  except OSError:pass
 if set(children)!={v['pid'] for v in d['active']}:raise ValueError('Controller is between launch and heartbeat; retry adoption')
 os.kill(pid,signal.SIGKILL)
except Exception:
 os.kill(pid,signal.SIGCONT);raise
original=dict(d);d['plan_sha256']=hashlib.sha256(plan.read_bytes()).hexdigest();tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
command=['/usr/bin/python3','-u',str(ROOT/'code/rgpe_dispatch.py'),'--plan',str(plan),'--host',host,'--execute']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
with (ROOT/(host+'.rgpe_dispatch.log')).open('ab') as log:c=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
(ROOT/(host+'.memory_plan_adoption.json')).write_text(json.dumps(dict(at=time.time(),old_pid=pid,new_pid=c.pid,adopted_tasks=len(children),plan=str(plan),previous_status=original),indent=2))
print(json.dumps(dict(host=host,pid=c.pid,adopted=len(children))))
