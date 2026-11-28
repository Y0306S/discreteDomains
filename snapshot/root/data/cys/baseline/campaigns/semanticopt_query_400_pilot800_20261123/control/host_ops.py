import argparse,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/root/data/cys/baseline/campaigns/semanticopt_query_400_pilot800_20261123')
HY=ROOT.parent/'query_our_fsbo_vae80_test200_20261117'
OUT=ROOT/'evidence/handoff'
PG='/root/data/cys/envs/pg16/bin/pg_ctl';PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
PREFIX=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
def atomic(p,v):q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def retire(host):
 probe=json.loads((OUT/(host+'.probe.json')).read_text());assert not any(w['alive'] for w in probe['workers']);assert not probe['unrecorded_workers'];assert not any(d.get('active') for d in probe['databases'])
 selected=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
   if any(str(HY/'batch_code'/name) in argv for name in ['qwen_inference_server.py','query_vae_service.py']):selected.append((int(p.name),argv))
  except (OSError,UnicodeError):pass
 for pid,_ in selected:os.kill(pid,signal.SIGTERM)
 atomic(OUT/(host+'.retired_services.json'),dict(at=time.time(),processes=[dict(pid=pid,argv=argv) for pid,argv in selected]))
 print(host,'retired',len(selected),'owned model services',flush=True)
def databases(host):
 import psycopg
 cfg=json.loads((ROOT/'allocation.json').read_text());selected=[w for w in cfg['workers']+cfg['bolt_workers'] if w['host']==host];used={w['index'] for w in selected}
 allrows=[w for w in json.loads((HY/'resources.json').read_text())['workers'] if w['host']==host]
 env=dict(os.environ,LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
 records=[]
 for w in allrows:
  directory=Path(w['pgdata']);base=PREFIX+[PG,'-D',str(directory)]
  # Verify exclusive idle state immediately before handoff.
  with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',password='imdb',autocommit=True,connect_timeout=10) as conn:
   active=conn.execute("SELECT pid FROM pg_stat_activity WHERE datname='imdb' AND state='active' AND pid<>pg_backend_pid()").fetchall();assert not active,(host,w['index'],active)
  subprocess.run(base+['-m','fast','-w','-t','60','stop'],env=env,cwd='/tmp',stdout=subprocess.DEVNULL,check=True)
  if w['index'] not in used:records.append(dict(index=w['index'],status='released_idle'));continue
  w=next(x for x in selected if x['index']==w['index']);sock=Path('/tmp')/f'query_cont_{w["db_port"]}';sock.mkdir(exist_ok=True,mode=0o700);os.chown(sock,10000,10000)
  opts=['-p',str(w['db_port']),'-h','127.0.0.1','-k',str(sock),'-c','dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib','-c','max_connections=12','-c','shared_buffers=1GB','-c','work_mem=64MB','-c','max_parallel_workers_per_gather=2','-c','max_parallel_workers=2','-c','temp_file_limit=8GB','-c','autovacuum=off']
  import shlex
  subprocess.run(['taskset','-c',','.join(map(str,w['postgres_cpus']))]+base+['-l',str(directory/'continuation.log'),'-o',shlex.join(opts),'-w','-t','90','start'],env=env,cwd='/tmp',stdout=subprocess.DEVNULL,check=True)
  with psycopg.connect(host='127.0.0.1',port=w['db_port'],dbname='imdb',user='imdb',password='imdb',autocommit=True) as c:
   c.execute("LOAD 'pg_hint_plan'");assert c.execute('SELECT count(*) FROM title').fetchone()[0]>1000000
   settings=dict(c.execute("SELECT name,setting FROM pg_settings WHERE name IN ('shared_buffers','work_mem','max_parallel_workers_per_gather','temp_file_limit')").fetchall())
  records.append(dict(**w,status='ready',settings=settings))
 atomic(OUT/(host+'.database_handoff.json'),dict(at=time.time(),databases=records));print(host,'private DBs ready',len(selected),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['retire','databases']);p.add_argument('host');a=p.parse_args();globals()[a.action](a.host)
