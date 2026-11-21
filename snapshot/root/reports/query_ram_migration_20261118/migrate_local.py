import fcntl,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
R=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'operations/recovery'))
import recovery
PG=Path('/root/data/cys/envs/pg16/bin')
PREFIX=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
def run(args):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=300)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 return p.stdout
def save(name,v):
 p=O/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def checksum(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def main():
 lock=(R/'operations/recovery/tick.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (R/'operations/recovery/STOP').exists()
 probe=recovery.probe('LLMOptimizer2')
 assert not probe['unrecorded_workers'] and not any(w['alive'] for w in probe['workers'])
 assert all(d['ready'] and not d['active'] for d in probe['databases'])
 cfg=json.loads((R/'resources.json').read_text());results=[]
 assert shutil.disk_usage('/dev/shm').free>190*1024**3
 for w in cfg['workers']:
  if w['host']!='LLMOptimizer2':continue
  src=Path(w['pgdata']);dst=Path('/dev/shm/query_orpo_fsbo_20261118')/f"w{w['index']}"
  assert not dst.exists() and not any((src/'pg_tblspc').iterdir())
  opts=shlex.split((src/'postmaster.opts').read_text())
  assert str(src) in opts and str(w['db_port']) in opts
  save('state.json',dict(stage='stopping_copying',worker=w['index'],at=time.time(),completed=results))
  run(PREFIX+[PG/'pg_ctl','-D',src,'-w','-t','120','-m','fast','stop'])
  control=run(PREFIX+[PG/'pg_controldata',src]);assert 'shut down' in control
  dst.parent.mkdir(parents=True,exist_ok=True);os.chmod(dst.parent,0o755)
  shutil.copytree(src,dst,copy_function=shutil.copy2)
  manifest={}
  for p in src.rglob('*'):
   q=dst/p.relative_to(src)
   if p.is_file():
    h=checksum(p);assert h==checksum(q),str(p);manifest[str(p.relative_to(src))]=h
   os.chown(q,p.stat().st_uid,p.stat().st_gid)
  os.chown(dst,src.stat().st_uid,src.stat().st_gid);os.chmod(dst,0o700)
  assert run(PREFIX+[PG/'pg_controldata',dst])==control
  save(f"worker{w['index']}_manifest.json",manifest)
  opts=[str(dst) if s==str(src) else s for s in opts]
  (dst/'postmaster.opts').write_text(shlex.join(opts)+'\n');os.chown(dst/'postmaster.opts',10000,10000)
  run(PREFIX+['taskset','-c',','.join(map(str,w['postgres_cpus'])),PG/'pg_ctl','-D',dst,'-l',dst/'semanticopt.log','-o',shlex.join(opts[1:]),'-w','-t','180','start'])
  w['pgdata']=str(dst)
  results.append(dict(index=w['index'],port=w['db_port'],source=str(src),destination=str(dst),verified_files=len(manifest),at=time.time()))
  # Keep recovery allocation consistent after every verified migration.
  t=R/'resources.json.ram.tmp';t.write_text(json.dumps(cfg,indent=2));t.replace(R/'resources.json')
  save('state.json',dict(stage='migrating',completed=results,at=time.time()))
 recovery.ensure_databases('LLMOptimizer2')
 save('state.json',dict(stage='local_migration_verified',completed=results,probe=recovery.probe('LLMOptimizer2'),at=time.time()))
 print('LOCAL_MIGRATION_VERIFIED',flush=True)
if __name__=='__main__':main()
