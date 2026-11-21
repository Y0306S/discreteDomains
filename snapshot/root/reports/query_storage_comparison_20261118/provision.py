import hashlib,json,os,shlex,shutil,socket,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PG=Path('/root/data/cys/envs/pg16/bin')
RAM=Path('/dev/shm/query_storage_comparison_20261118/ram')
DISK=Path('/root/data/cys/baseline/storage_tests/query_storage_comparison_20261118/lustre')
PREFIX=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
def run(args,**kw):
    print(shlex.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,**kw)
def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2))
def main():
    assert not RAM.exists() and not DISK.exists(), 'Never overwrite an existing cluster'
    import psycopg
    with psycopg.connect(host='127.0.0.1',port=20402,dbname='imdb',user='imdb',autocommit=True) as c:
        assert c.execute("select count(*) from pg_stat_activity where backend_type='client backend' and pid<>pg_backend_pid() and state='active'").fetchone()[0]==0
    for port in (22450,22451):
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    RAM.parent.mkdir(parents=True);os.chmod(RAM.parent,0o755);os.chown(RAM.parent,10000,10000)
    DISK.parent.mkdir(parents=True);os.chmod(DISK.parent,0o755);os.chown(DISK.parent,10000,10000)
    started=time.time()
    save('provision_state.json',dict(state='basebackup',started=started,source_port=20402))
    run(PREFIX+[PG/'pg_basebackup','-h','127.0.0.1','-p','20402','-U','imdb','-D',RAM,'-X','stream','-c','fast','--progress','--verbose'])
    run(PREFIX+[PG/'pg_verifybackup',RAM])
    save('provision_state.json',dict(state='copy_to_lustre',started=started))
    # Both test clusters start from the same verified, stopped base backup.
    run(['cp','-a',RAM,DISK])
    run(PREFIX+[PG/'pg_verifybackup',DISK])
    clusters=[]
    for label,path,port in [('ram',RAM,22450),('lustre',DISK,22451)]:
        manifest=hashlib.sha256((path/'backup_manifest').read_bytes()).hexdigest()
        sock=Path('/tmp/query_storage_comparison_'+str(port));sock.mkdir();os.chown(sock,10000,10000)
        options=['-p',str(port),'-h','127.0.0.1','-k',str(sock),'-c','dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib','-c','max_connections=12','-c','shared_buffers=1GB','-c','work_mem=64MB','-c','max_parallel_workers_per_gather=2','-c','max_parallel_workers=2','-c','temp_file_limit=8GB','-c','autovacuum=off']
        logfile=path/'storage_test.log'
        run(PREFIX+['taskset','-c','0,1,2',PG/'pg_ctl','-D',path,'-l',logfile,'-o',shlex.join(options),'-w','start'])
        pid=int((path/'postmaster.pid').read_text().splitlines()[0])
        clusters.append(dict(label=label,path=str(path),port=port,pid=pid,cpus=sorted(os.sched_getaffinity(pid)),manifest_sha256=manifest,options=options))
    assert clusters[0]['manifest_sha256']==clusters[1]['manifest_sha256']
    save('clusters.json',dict(clusters=clusters,source_port=20402,started=started,ready=time.time()))
    save('provision_state.json',dict(state='ready',elapsed_seconds=time.time()-started))
    print('READY',flush=True)
if __name__=='__main__':main()
