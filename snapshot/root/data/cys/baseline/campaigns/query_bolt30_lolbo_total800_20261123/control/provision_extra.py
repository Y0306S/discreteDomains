"""Prepare the explicitly authorized eighth private database on Optimizer3."""
import hashlib,json,os,shlex,shutil,socket,subprocess,time
from pathlib import Path
import psycopg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/worker07'
PG=Path('/root/data/cys/envs/pg16/bin')
AS=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
DATA=Path('/dev/shm/query_bolt_extra_20261123/offline_w37')
SCRATCH=ROOT/'scratch/LLMOptimizer3/worker37_offline'
SOCK=Path('/tmp/query_bolt_extra_20437')
SPEC=dict(index=37,host='LLMOptimizer3',slot=7,method='bolt',optimizer_cpu=19,
          postgres_cpus=[16,17,18],db_port=20437,pgdata=str(DATA),temp_tablespace_dir=str(SCRATCH))

def command(args,name):
    with (OUT/(name+'.log')).open('ab') as log:
        subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800,
                       cwd='/tmp',env=dict(os.environ,LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib:'+os.environ.get('LD_LIBRARY_PATH','')))

def main():
    assert socket.gethostname()=='cs-0b2cd-18deb-server'
    OUT.mkdir(parents=True,exist_ok=True)
    cg=Path('/sys/fs/cgroup')
    used=int((cg/'memory.current').read_text());maximum=int((cg/'memory.max').read_text())
    assert maximum-used>20*2**30,'Need20GiB headroom for RAM clone and worker'
    assert shutil.disk_usage('/dev/shm').free>13*2**30
    assert {16,17,18,19}<=os.sched_getaffinity(0)
    for p in Path('/proc').glob('[0-9]*'):
        try:
            aff=os.sched_getaffinity(int(p.name))
            assert not aff.issubset({16,17,18,19}),('Dedicated CPUs occupied',p.name)
        except (OSError,ProcessLookupError):pass
    with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',20437))!=0
    for path in (DATA,SCRATCH,SOCK):
        path.mkdir(parents=True,exist_ok=True)
        assert not any(path.iterdir()),('Target must be empty',path)
        os.chown(path,10000,10000);path.chmod(0o700)
    source=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117/databases/LLMOptimizer4/w22')
    assert not (source/'postmaster.pid').exists()
    control=subprocess.check_output([str(PG/'pg_controldata'),str(source)],text=True)
    assert 'Database cluster state:               shut down' in control
    control_before=(source/'global/pg_control').read_bytes()
    (OUT/'offline_source_control.txt').write_text(control)
    manifest=[]
    def copy_tree(src,dst):
        dst.mkdir(exist_ok=True);os.chown(dst,10000,10000);dst.chmod(0o700)
        for old in sorted(src.iterdir()):
            new=dst/old.name
            if old.is_symlink():
                assert src==source/'pg_tblspc',('Unexpected symlink',old)
                target=SCRATCH/old.name
                copy_tree(old.resolve(),target)
                new.symlink_to(target);os.lchown(new,10000,10000)
            elif old.is_dir():copy_tree(old,new)
            else:
                before=old.stat();digest=hashlib.sha256()
                with old.open('rb') as reader,new.open('xb') as writer:
                    while chunk:=reader.read(4*2**20):writer.write(chunk);digest.update(chunk)
                after=old.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                check=hashlib.sha256()
                with new.open('rb') as reader:
                    while chunk:=reader.read(4*2**20):check.update(chunk)
                assert digest.digest()==check.digest(),new
                shutil.copystat(old,new);os.chown(new,10000,10000)
                manifest.append(dict(source=str(old),destination=str(new),bytes=before.st_size,sha256=digest.hexdigest()))
    (OUT/'preflight.json').write_text(json.dumps(dict(at=time.time(),memory_current=used,memory_max=maximum,spec=SPEC,offline_source=str(source)),indent=2)+'\n')
    print('Copying cleanly stopped private database; verifying every file SHA256',flush=True)
    copy_tree(source,DATA)
    assert (source/'global/pg_control').read_bytes()==control_before and not (source/'postmaster.pid').exists()
    (OUT/'offline_copy_manifest.json').write_text(json.dumps(dict(at=time.time(),all_file_checksums_verified=True,files=manifest),indent=2)+'\n')
    options=['-p','20437','-h','127.0.0.1','-k',str(SOCK),'-c','dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib',
             '-c','max_connections=12','-c','shared_buffers=1GB','-c','work_mem=64MB','-c','max_parallel_workers_per_gather=2',
             '-c','max_parallel_workers=2','-c','temp_file_limit=8GB','-c','autovacuum=off']
    command(AS+['taskset','-c','16,17,18',str(PG/'pg_ctl'),'-D',str(DATA),'-l',str(DATA/'continuation.log'),'-o',shlex.join(options),'-w','-t','90','start'],'start')
    with psycopg.connect(host='127.0.0.1',port=20437,dbname='imdb',user='imdb',password='imdb',autocommit=True) as db:
        assert db.execute('SHOW data_directory').fetchone()[0]==str(DATA)
        db.execute("LOAD 'pg_hint_plan'")
        assert db.execute('SHOW pg_hint_plan.enable_hint').fetchone()[0]=='on'
        assert db.execute('SELECT count(*) FROM title').fetchone()[0]==2528312
        assert db.execute('SELECT count(*) FROM movie_info').fetchone()[0]==14835720
        assert db.execute("SELECT count(*) FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND state='active'").fetchone()[0]==0
        settings=dict(db.execute("SELECT name,setting FROM pg_settings WHERE name IN ('shared_buffers','work_mem','max_parallel_workers_per_gather','temp_file_limit','temp_tablespaces')").fetchall())
    pid=int((DATA/'postmaster.pid').read_text().splitlines()[0]);assert os.sched_getaffinity(pid)=={16,17,18}
    remaining=maximum-int((cg/'memory.current').read_text());assert remaining>8*2**30
    result=dict(ready=True,at=time.time(),spec=SPEC,postmaster_pid=pid,verified_offline_copy_manifest=True,settings=settings,memory_headroom_bytes=remaining)
    (OUT/'database.ready.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'control/extra_worker.json').write_text(json.dumps(SPEC,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
