"""Cooperative CPU leases and isolated PostgreSQL instances; preserve active evaluations."""
import fcntl,json,os,subprocess,time,shlex
from pathlib import Path
from query_protocol import atomic,digest
ROOT=Path(__file__).resolve().parents[1];SEM=ROOT.parent/'semanticopt_eval200_20261110'
PG='/root/data/cys/envs/pg16/bin/pg_ctl';PREFIX=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
def reserve(sft=False,release=False):
 with (SEM/'sft_scheduling_amendment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);path=SEM/'scheduling.json';value=json.loads(path.read_text());previous=digest(path);purposes=value['reservation_purposes']['LLMOptimizer2'];key='query_warmstart_full_SFT' if sft else 'query_ORPO_RL'
  cpus=[42,43,44,45,46,47] if sft else [30,31,32,33,34,35];old=set(value['orpo_reserved_cpus']['LLMOptimizer2']);owned=set(purposes.get(key,[]))
  if not sft:
   # Inherit the completed peptide allocation; its trainer is already complete.
   prior=ROOT.parent/'peptide_our_orpo_rl800_20261111/services/queue_status.json'
   if json.loads(prior.read_text()).get('status')!='complete':raise RuntimeError('Previous ORPO/RL allocation remains active')
   if not release:purposes.pop('peptide_ORPO_RL',None)
  purposes[key]=[] if release else cpus
  value['orpo_reserved_cpus']['LLMOptimizer2']=sorted((old-owned)|set([] if release else cpus))
  value['query_training_resource_amendment']=dict(at=time.time(),campaign=str(ROOT),kind=key,released=release)
  atomic(path,value);identity=json.loads((SEM/'identity.json').read_text());identity['files'][str(path)]=digest(path);identity['query_training_resource_amendment']=dict(at=time.time(),previous_sha256=previous,scheduling_sha256=digest(path),scientific_recipe_changed=False);atomic(SEM/'identity.json',identity)

def cpu_clear(cpus):
 status=json.loads((SEM/'query/LLMOptimizer2.status.json').read_text())
 return time.time()-status['at']<40 and all(not set(cpus)&set(w['cpus']) for w in status.get('workers',[]))
def gpu_free():
 return [int(x)/1024 for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).split()]
def host_free():
 cg=Path('/sys/fs/cgroup');limit=(cg/'memory.max').read_text().strip();ram=((int(limit) if limit!='max' else 2**60)-int((cg/'memory.current').read_text()))/2**30;fs=os.statvfs(ROOT)
 return dict(ram_free_gib=ram,disk_free_gib=fs.f_bavail*fs.f_frsize/2**30,free_inodes=fs.f_favail)
def databases():
 recipe=json.loads((SEM/'recipe.json').read_text());archive=recipe['query']['database_archive'];results=[]
 env=dict(os.environ,LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
 for index,cpus in enumerate([[30,31,32],[33,34,35]]):
  directory=ROOT/'databases'/f'w{index:02}';port=20580+index
  if not (directory/'PG_VERSION').exists():
   if directory.exists():raise RuntimeError('Partial PostgreSQL extraction needs audit')
   if host_free()['disk_free_gib']<30:raise RuntimeError('Insufficient database space')
   directory.mkdir(parents=True,mode=0o700);subprocess.run(['tar','-xf',archive,'-C',str(directory),'--strip-components=1'],check=True);subprocess.run(['chown','-R','10000:10000',str(directory)],check=True)
  socket=Path('/tmp')/f'query_rl_pg_{port}';socket.mkdir(mode=0o700,exist_ok=True);os.chown(socket,10000,10000)
  opts=['-p',str(port),'-h','127.0.0.1','-k',str(socket),'-c','dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib','-c','max_connections=12','-c','shared_buffers=1GB','-c','work_mem=64MB','-c','max_parallel_workers_per_gather=2','-c','max_parallel_workers=2','-c','temp_file_limit=8GB','-c','autovacuum=off']
  base=PREFIX+[PG,'-D',str(directory)]
  if subprocess.run(base+['status'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
   subprocess.run(['taskset','-c',','.join(map(str,cpus))]+base+['-l',str(directory/'postgres.log'),'-o',shlex.join(opts),'-w','-t','90','start'],env=env,cwd='/tmp',check=True)
  import psycopg
  with psycopg.connect(host='127.0.0.1',port=port,dbname='imdb',user='imdb',password='imdb',autocommit=True) as c:
   c.execute("LOAD 'pg_hint_plan'");n=c.execute('SELECT count(*) FROM title').fetchone()[0]
   if n<1_000_000:raise RuntimeError('Wrong database')
  results.append(dict(port=port,cpus=cpus,titles=n))
 atomic(ROOT/'evidence/databases.json',dict(passed=True,databases=results,source_archive=archive,at=time.time()))
 return results

def stop_databases():
 env=dict(os.environ,LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
 for directory in (ROOT/'databases').glob('w*'):
  if (directory/'postmaster.pid').exists():
   subprocess.run(PREFIX+[PG,'-D',str(directory),'-m','fast','-w','-t','60','stop'],env=env,cwd='/tmp',check=True)
 atomic(ROOT/'services/database_release.json',dict(at=time.time(),status='stopped_after_completed_RL'))
