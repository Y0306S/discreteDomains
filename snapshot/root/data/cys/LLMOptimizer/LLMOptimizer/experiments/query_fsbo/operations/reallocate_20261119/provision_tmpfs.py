"""Create a verified private IMDb clone in RAM from the live read-only source."""
import argparse,json,os,shlex,socket,subprocess,time
from pathlib import Path
import psycopg
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PG=Path('/root/data/cys/envs/pg16/bin')
AS=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
p=argparse.ArgumentParser();p.add_argument('index',type=int);a=p.parse_args();i=a.index
assert 29<=i<=36
host='LLMOptimizer2' if i<33 else 'LLMOptimizer3'
assert socket.gethostname()=={'LLMOptimizer2':'cs-94bac-0693e-server','LLMOptimizer3':'cs-0b2cd-18deb-server'}[host]
cpus=list(range(40+4*(i-29),43+4*(i-29))) if i<33 else list(range(4*(i-33),4*(i-33)+3))
optimizer=cpus[-1]+1;port=20400+i;source=20402 if i<33 else 21402
data=Path('/dev/shm/query_expand_20261119')/f'w{i}'
scratch=ROOT/'scratch'/host/f'worker{i}';sock=Path('/tmp/queryfsbo_expand')/f'w{i}'
def owned(path):
 path.mkdir(parents=True,exist_ok=True);os.chown(path,10000,10000);path.chmod(0o700)
def run(args,label):
 with (OUT/f'w{i}.{label}.log').open('ab') as f:subprocess.run(args,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
assert not (data/'PG_VERSION').exists(),'Existing clone must be audited separately'
assert not data.exists() or not any(data.iterdir())
assert set(cpus+[optimizer])<=os.sched_getaffinity(0)
assert __import__('shutil').disk_usage('/dev/shm').free>12*2**30
with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0
owned(data);owned(scratch);owned(sock)
run(AS+['taskset','-c',','.join(map(str,cpus)),str(PG/'pg_basebackup'),'-h','127.0.0.1','-p',str(source),'-U','imdb','-w','-D',str(data),'-F','p','-X','stream','-c','fast','-r','128M','--manifest-checksums=SHA256','-P','-v'],'backup')
run(AS+[str(PG/'pg_verifybackup'),'-n',str(data)],'verify')
assert 'backup successfully verified' in (OUT/f'w{i}.verify.log').read_text()
options=['-D',str(data),'-p',str(port),'-h','127.0.0.1','-k',str(sock),'-c','dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib','-c','max_connections=12','-c','shared_buffers=1GB','-c','work_mem=64MB','-c','max_parallel_workers_per_gather=2','-c','max_parallel_workers=2','-c','temp_file_limit=8GB','-c','autovacuum=off']
run(AS+['taskset','-c',','.join(map(str,cpus)),str(PG/'pg_ctl'),'-D',str(data),'-l',str(data/'query.log'),'-o',shlex.join(options),'-w','-t','180','start'],'start')
with psycopg.connect(host='127.0.0.1',port=port,dbname='imdb',user='imdb',password='imdb',connect_timeout=10,autocommit=True) as db:
 assert db.execute('SHOW data_directory').fetchone()[0]==str(data)
 db.execute("LOAD 'pg_hint_plan'");assert db.execute('SHOW pg_hint_plan.enable_hint').fetchone()[0]=='on'
 assert db.execute('SELECT count(*) FROM title').fetchone()[0]==2528312
 assert db.execute('SELECT count(*) FROM movie_info').fetchone()[0]==14835720
 db.execute('CREATE TABLESPACE query_hybrid_temp LOCATION '+psycopg.sql.Literal(str(scratch)).as_string(db))
 db.execute("ALTER DATABASE imdb SET temp_tablespaces='query_hybrid_temp'")
record=dict(ready=True,index=i,host=host,optimizer_cpu=optimizer,postgres_cpus=cpus,db_port=port,pgdata=str(data),temp_tablespace_dir=str(scratch),at=time.time(),verified_backup_manifest=True)
(OUT/f'w{i}.ready.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
