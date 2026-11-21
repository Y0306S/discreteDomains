import concurrent.futures,json,os,sys,time
from pathlib import Path
import psycopg
R=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
host=sys.argv[1]
def check(w):
 with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',autocommit=True,options='-c statement_timeout=180000') as c:
  assert c.execute('SHOW data_directory').fetchone()[0]==w['pgdata']
  c.execute("LOAD 'pg_hint_plan'");assert c.execute('SHOW pg_hint_plan.enable_hint').fetchone()[0]=='on'
  blocks=[]
  for rel,block in [('cast_info',119884),('cast_info',70302),('name',19245)]:
   v=c.execute(f'SELECT count(*) FROM {rel} WHERE ctid >= %s::tid AND ctid < %s::tid',(f'({block},0)',f'({block+1},0)')).fetchone()[0]
   blocks.append(dict(relation=rel,block=block,tuples=v))
  counts={}
  if host=='LLMOptimizer2' or not w['pgdata'].startswith('/dev/shm'):
   for rel,expected in [('name',4167491),('cast_info',36244344),('movie_info',14835720),('title',2528312)]:
    n=c.execute(f'SELECT count(*) FROM {rel}').fetchone()[0];assert n==expected,(w['index'],rel,n);counts[rel]=n
  pid=int((Path(w['pgdata'])/'postmaster.pid').read_text().splitlines()[0])
  affinity=sorted(os.sched_getaffinity(pid));assert affinity==w['postgres_cpus']
  return dict(index=w['index'],port=w['db_port'],pgdata=w['pgdata'],cpus=affinity,failed_blocks_read=blocks,counts=counts,shared_buffers=c.execute('SHOW shared_buffers').fetchone()[0],temp_tablespaces=c.execute('SHOW temp_tablespaces').fetchone()[0],at=time.time())
workers=[w for w in json.loads((R/'resources.json').read_text())['workers'] if w['host']==host]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(check,workers))
print(json.dumps(dict(host=host,passed=True,databases=rows,at=time.time())))
