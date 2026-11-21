"""Finite, checkpoint-resumable 200-task queue with per-host resource admission."""
import fcntl,json,os,shutil,socket,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
ALLOC={'cs-94bac-0693e-server':('LLMOptimizer2',list(range(40,60)),0),
       'cs-0b2cd-18deb-server':('LLMOptimizer3',list(range(18,34)),20),
       'cs-fa498-8793f-server':('LLMOptimizer4',list(range(76,88)),36)}

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def usage():
    c=Path('/sys/fs/cgroup')
    stats=dict(x.split() for x in (c/'cpu.stat').read_text().splitlines())
    memory=dict(x.split() for x in (c/'memory.stat').read_text().splitlines())
    return int(stats['usage_usec']),int((c/'memory.current').read_text()),int(memory.get('inactive_file',0))

def healthy(port):
    try:
        with urllib.request.urlopen('http://127.0.0.1:%d/health'%port,timeout=2) as r:d=json.load(r)
        return not d.get('errors',0)
    except Exception:return False

def main():
    name,cpus,offset=ALLOC[socket.gethostname()]
    (ROOT/'dispatch').mkdir(exist_ok=True)
    lock=(ROOT/'dispatch'/(name+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    recipe=json.loads((ROOT/'rgpe/recipe.json').read_text())
    tasks=[t for i,t in enumerate(recipe['allowed_tasks']) if offset<=i%48<offset+len(cpus)]
    cgroup=Path('/sys/fs/cgroup')
    quota,period=map(int,(cgroup/'cpu.max').read_text().split())
    cap=quota/period
    memory_max=int((cgroup/'memory.max').read_text())
    live={};failed={};done=[];attempts={}
    previous=usage()[0];tick=time.monotonic()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    while True:
        for cpu,(proc,task) in list(live.items()):
            code=proc.poll()
            if code is not None:
                del live[cpu]
                archive=ROOT/'rgpe/runs'/(task+'.tar.gz')
                if code==0 and archive.exists():done.append(task)
                else:
                    failed[task]=dict(exit_code=code,at=time.time(),attempt=attempts[task])
                    # One retry safely reuses the durable pending proposal/oracle reply.
                    if attempts[task]<2:tasks.append(task)
        ready=False
        gate=ROOT/'evidence/canary_passed.json'
        if gate.exists():
            g=json.loads(gate.read_text());ready=g.get('passed',False)
        used,current,inactive=usage()
        now=time.monotonic();cores=(used-previous)/1e6/max(now-tick,.01)
        previous,tick=used,now
        available=memory_max-current+inactive
        # Reserve growth beyond currently resident task processes. Each task's
        # model is single-threaded; the quota guard includes all existing jobs.
        rss=0
        for proc,_ in live.values():
            try:
                rss+=int(Path('/proc',str(proc.pid),'statm').read_text().split()[1])*4096
                children=Path('/proc',str(proc.pid),'task',str(proc.pid),'children').read_text().split()
                for pid in children:rss+=int(Path('/proc',pid,'statm').read_text().split()[1])*4096
            except OSError:pass
        required=16*2**30+max(0,len(live)*4*2**30-rss)+4*2**30
        reason='ready'
        if not ready:reason='awaiting_validated_canary'
        elif cores>cap*.9:reason='CPU_quota_admission'
        elif available<required:reason='memory_growth_admission'
        elif shutil.disk_usage(ROOT).free<60*2**30:reason='disk_headroom'
        elif tasks and len(live)<len(cpus):
            cpu=next(cpu for cpu in cpus if cpu not in live)
            slot=offset+cpus.index(cpu)
            math=22440+slot%4;vae=22460+slot%8
            if not healthy(math) or not healthy(vae):reason='inference_service_unavailable'
            else:
                task=tasks.pop(0)
                if (ROOT/'rgpe/runs'/(task+'.tar.gz')).exists():done.append(task)
                else:
                    cmd=[PY,'-u',str(ROOT/'code/launch.py'),'--endpoint','http://127.0.0.1:%d'%math,
                        'run-task','--campaign',str(ROOT/'rgpe'),'--task',task,'--cpu',str(cpu),
                        '--server','http://127.0.0.1:%d'%vae]
                    attempts[task]=attempts.get(task,0)+1
                    with (ROOT/'dispatch'/(task+'.log')).open('a') as f:
                        proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,
                            stderr=subprocess.STDOUT,start_new_session=True)
                    live[cpu]=(proc,task)
                    write(ROOT/'dispatch'/(task+'.process.json'),dict(pid=proc.pid,host=name,cpu=cpu,
                        start_ticks=Path('/proc',str(proc.pid),'stat').read_text().split()[21],
                        cmd=cmd,started=time.time(),attempt=attempts[task]))
        write(ROOT/'dispatch'/(name+'.status.json'),dict(at=time.time(),host=name,max_workers=len(cpus),
            active=[dict(cpu=c,pid=p.pid,task=t) for c,(p,t) in live.items()],queued=len(tasks),
            complete=done,failures=failed,admission=reason,cpu_cores=cores,
            available_memory_gib=available/2**30,required_memory_gib=required/2**30))
        if not tasks and not live:break
        time.sleep(5)

if __name__=='__main__':main()
