"""Evict only this investigation's private Lustre files, never global caches."""
import ctypes,json,math,os,subprocess,time
from pathlib import Path
import benchmark as bench
from provision import PREFIX,PG

ROOT=Path(__file__).resolve().parent
LIBC=ctypes.CDLL(None,use_errno=True)
LIBC.mmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_long]
LIBC.mmap.restype=ctypes.c_void_p
LIBC.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
LIBC.munmap.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
def resident(path,evict=False):
    total=present=0;failures=[]
    for file in sorted((Path(path)/'base/16385').iterdir()):
        if not file.is_file() or not file.name.split('.')[0].isdigit():continue
        size=file.stat().st_size
        if not size:continue
        fd=os.open(file,os.O_RDONLY)
        try:
            if evict:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
            pages=math.ceil(size/os.sysconf('SC_PAGE_SIZE'))
            addr=LIBC.mmap(None,size,1,1,fd,0)
            if addr==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_errno(),'mmap')
            try:
                vec=(ctypes.c_ubyte*pages)()
                if LIBC.mincore(addr,size,vec):raise OSError(ctypes.get_errno(),'mincore')
                total+=pages;present+=sum(v&1 for v in vec)
            finally:LIBC.munmap(addr,size)
        except OSError as exc:failures.append(dict(file=str(file),error=str(exc)))
        finally:os.close(fd)
    return dict(pages=total,resident_pages=present,resident_fraction=present/total if total else None,failures=failures)
def stop(c):
    with bench.connect(c['label']) as db:
        assert db.execute("select count(*) from pg_stat_activity where backend_type='client backend' and pid<>pg_backend_pid()").fetchone()[0]==0
        assert db.execute('show data_directory').fetchone()[0]==c['path']
    subprocess.run(PREFIX+[str(PG/'pg_ctl'),'-D',c['path'],'-w','-m','fast','stop'],check=True,stdout=subprocess.DEVNULL)
def start(c):
    # pg_ctl restart settings are retained in postmaster.opts; ownership, ports
    # and CPUs are preserved. Only our isolated test clusters are restarted.
    import shlex
    subprocess.run(PREFIX+['taskset','-c','0,1,2',str(PG/'pg_ctl'),'-D',c['path'],'-l',str(Path(c['path'])/'storage_test.log'),'-o',shlex.join(c['options']),'-w','start'],check=True,stdout=subprocess.DEVNULL)
def main():
    assert json.loads((ROOT/'benchmark_state.json').read_text())['state']=='complete'
    clusters={c['label']:c for c in json.loads((ROOT/'clusters.json').read_text())['clusters']}
    rows=json.loads((ROOT/'samples.json').read_text());started=time.time()
    assert not (ROOT/'cache_probe.jsonl').exists()
    # One storage-read diagnostic per saved plan. This is separate from the
    # five warmed repetitions and is not called a physical disk-cold guarantee.
    for row in rows:
        stop(clusters['ram']);stop(clusters['lustre'])
        before=resident(clusters['lustre']['path'])
        after=resident(clusters['lustre']['path'],evict=True)
        ram=resident(clusters['ram']['path'])
        event=dict(sample=row['id'],before=before,after_advice=after,ram=ram,at=time.time())
        bench.append('cache_probe.jsonl',event)
        start(clusters['ram']);start(clusters['lustre'])
        for label in (['ram','lustre'] if row['id']%2==0 else ['lustre','ram']):
            bench.execute(row,label,'private_file_cache_advice',0)
        print('CACHE',row['id'],after['resident_fraction'],flush=True)
        bench.save('cold_probe_state.json',dict(state='running',completed=row['id']+1,total=len(rows),elapsed_seconds=time.time()-started))
    bench.save('cold_probe_state.json',dict(state='complete',completed=len(rows),elapsed_seconds=time.time()-started))
if __name__=='__main__':main()
