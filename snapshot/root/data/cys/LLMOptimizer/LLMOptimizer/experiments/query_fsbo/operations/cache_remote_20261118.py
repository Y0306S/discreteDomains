import json,os,shlex,subprocess,sys,time
from pathlib import Path
import psycopg
R=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
O=R/'operations/ram_migration_20261118';O.mkdir(exist_ok=True)
PG=Path('/root/data/cys/envs/pg16/bin');PREFIX=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
sys.path.insert(0,str(R/'operations/recovery'));import recovery
def run(args):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True,timeout=240)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 return p.stdout
def conn(w):return psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',autocommit=True)
def main():
 assert (R/'operations/recovery/STOP').exists()
 probe=recovery.probe('LLMOptimizer4')
 assert not probe['unrecorded_workers'] and not any(w['alive'] for w in probe['workers'])
 assert all(d['ready'] and not d['active'] for d in probe['databases'])
 limit=int(Path('/sys/fs/cgroup/memory.max').read_text());used=int(Path('/sys/fs/cgroup/memory.current').read_text())
 assert limit-used>65*1024**3
 results=[]
 for w in json.loads((R/'resources.json').read_text())['workers']:
  if w['host']!='LLMOptimizer4' or w['pgdata'].startswith('/dev/shm'):continue
  d=Path(w['pgdata']);opts=shlex.split((d/'postmaster.opts').read_text())
  (O/f"worker{w['index']}.postmaster.opts.before").write_text(shlex.join(opts))
  assert 'shared_buffers=1GB' in opts
  with conn(w) as c:
   c.execute("ALTER SYSTEM SET shared_buffers='10GB'")
  run(PREFIX+[PG/'pg_ctl','-D',d,'-w','-t','120','-m','fast','stop'])
  opts=['shared_buffers=10GB' if a=='shared_buffers=1GB' else a for a in opts]
  run(PREFIX+['taskset','-c',','.join(map(str,w['postgres_cpus'])),PG/'pg_ctl','-D',d,'-l',d/'semanticopt.log','-o',shlex.join(opts[1:]),'-w','-t','180','start'])
  with conn(w) as c:
   assert c.execute('SHOW shared_buffers').fetchone()[0]=='10GB'
   c.execute('CREATE EXTENSION IF NOT EXISTS pg_prewarm')
   relations=c.execute("SELECT oid::regclass::text FROM pg_class WHERE relnamespace='public'::regnamespace AND relkind IN ('r','i','t') ORDER BY pg_relation_size(oid) DESC").fetchall()
   warmed=[]
   for rel, in relations:
    pages=c.execute('SELECT pg_prewarm(%s::regclass)',(rel,)).fetchone()[0];warmed.append(dict(relation=rel,pages=pages))
   results.append(dict(index=w['index'],port=w['db_port'],shared_buffers='10GB',warmed=warmed,at=time.time()))
  (O/'remote_cache.json').write_text(json.dumps(dict(stage='warming',completed=results),indent=2))
  print('WARMED',w['index'],flush=True)
 recovery.ensure_databases('LLMOptimizer4')
 (O/'remote_cache.json').write_text(json.dumps(dict(stage='remote_cache_verified',completed=results,at=time.time()),indent=2))
 print('REMOTE_CACHE_VERIFIED',flush=True)
if __name__=='__main__':main()
