"""Provision isolated PostgreSQL copies from a verified clean offline source."""
import concurrent.futures
import fcntl
import hashlib
import shlex
import shutil
from common import *

PG=Path('/root/data/cys/envs/pg16/bin')
SOURCE=HYB/'databases/LLMOptimizer4/w22'
OUT=ROOT/'cpu5';OUT.mkdir(exist_ok=True)
AS=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
def run(cmd,log,timeout=1800):
    with log.open('ab',buffering=0) as stream:
        subprocess.run(list(map(str,cmd)),check=True,stdout=stream,stderr=subprocess.STDOUT,
            cwd='/tmp',timeout=timeout,env=dict(os.environ,LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib'))
def provision(spec):
    i=spec['slot'];evidence=OUT/f'slot{i}';evidence.mkdir(exist_ok=True)
    if (evidence/'ready.json').exists():return load(evidence/'ready.json')
    data=Path(spec['pgdata']);scratch=Path(spec['temp_tablespace_dir'])
    sock=Path(f'/tmp/bolt_cpu5_{spec["db_port"]}')
    for path in [data,scratch,sock]:
        path.mkdir(parents=True,exist_ok=True);os.chown(path,10000,10000);path.chmod(0o700)
    assert not any(data.iterdir()),f'Target not empty: {data}'
    assert not (SOURCE/'postmaster.pid').exists()
    control=(SOURCE/'global/pg_control').read_bytes()
    assert 'Database cluster state:               shut down' in subprocess.check_output([str(PG/'pg_controldata'),str(SOURCE)],text=True)
    files=[]
    def copy_tree(src,dst):
        dst.mkdir(exist_ok=True);os.chown(dst,10000,10000);dst.chmod(0o700)
        for old in sorted(src.iterdir()):
            new=dst/old.name
            if old.is_symlink():
                assert src==SOURCE/'pg_tblspc'
                target=scratch/old.name;copy_tree(old.resolve(),target)
                new.symlink_to(target);os.lchown(new,10000,10000)
            elif old.is_dir():copy_tree(old,new)
            else:
                prior=old.stat();digest=hashlib.sha256()
                with old.open('rb') as reader,new.open('xb') as writer:
                    while chunk:=reader.read(8*1024**2):writer.write(chunk);digest.update(chunk)
                verify=hashlib.sha256()
                with new.open('rb') as reader:
                    while chunk:=reader.read(8*1024**2):verify.update(chunk)
                assert digest.digest()==verify.digest(),new
                assert (prior.st_size,prior.st_mtime_ns)==(old.stat().st_size,old.stat().st_mtime_ns)
                shutil.copystat(old,new);os.chown(new,10000,10000)
                files.append(dict(path=str(new),size=prior.st_size,sha256=digest.hexdigest()))
    copy_tree(SOURCE,data)
    assert (SOURCE/'global/pg_control').read_bytes()==control and not (SOURCE/'postmaster.pid').exists()
    atomic(evidence/'copy_manifest.json',dict(source=str(SOURCE),files=files,at=time.time()))
    opts=['-p',str(spec['db_port']),'-h','127.0.0.1','-k',str(sock),'-c',
        'dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib']
    for key,val in dict(max_connections=12,shared_buffers='1GB',work_mem='64MB',
        max_parallel_workers_per_gather=2,max_parallel_workers=2,temp_file_limit='8GB',autovacuum='off').items():
        opts+=['-c',f'{key}={val}']
    run(AS+['taskset','-c',','.join(map(str,spec['postgres_cpus'])),PG/'pg_ctl','-D',data,
        '-l',data/'bolt.log','-o',shlex.join(opts),'-w','-t','90','start'],evidence/'start.log')
    import psycopg2
    with psycopg2.connect(host='127.0.0.1',port=spec['db_port'],dbname='imdb',user='imdb',password='imdb') as conn:
        with conn.cursor() as q:
            q.execute("LOAD 'pg_hint_plan'");q.execute('SHOW pg_hint_plan.enable_hint');assert q.fetchone()[0]=='on'
            q.execute('SELECT count(*) FROM title');assert q.fetchone()[0]==2528312
            q.execute('SELECT count(*) FROM movie_info');assert q.fetchone()[0]==14835720
            q.execute('SHOW data_directory');assert q.fetchone()[0]==str(data)
            settings={}
            for name in ['work_mem','shared_buffers','max_parallel_workers_per_gather','temp_file_limit','temp_tablespaces']:
                q.execute('SHOW '+name);settings[name]=q.fetchone()[0]
    pid=int((data/'postmaster.pid').read_text().splitlines()[0])
    assert os.sched_getaffinity(pid)==set(spec['postgres_cpus'])
    result=dict(ready=True,spec=spec,pid=pid,start_ticks=ticks(pid),settings=settings,at=time.time())
    atomic(evidence/'ready.json',result);print(json.dumps(dict(slot=i,ready=True)),flush=True)
    return result
def main():
    require_host('LLMOptimizer5')
    handle=(OUT/'provision.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chmod('/root',0o711)
    assert Path('/sys/fs/cgroup/cpu.max').read_text().split()==['6400000','100000']
    policy=load(ROOT/'placement_cpu5.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(provision,policy['workers']))
    atomic(OUT/'databases_ready.json',dict(databases=results,at=time.time()))
    print('All fifteen databases ready',flush=True)
if __name__=='__main__':main()
