"""Admit task workers against CPU quotas, observed load and cgroup RAM."""
import argparse
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from common import *

PG='/root/data/cys/envs/pg16/bin/pg_ctl'
PREFIX=['setpriv','--reuid=10000','--regid=10000','--clear-groups']
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
    LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib:'+os.environ.get('LD_LIBRARY_PATH',''))

def resources():
    root=Path('/sys/fs/cgroup')
    quota,period=(root/'cpu.max').read_text().split()
    maximum=(root/'memory.max').read_text().strip()
    return dict(cpus=min(len(os.sched_getaffinity(0)),int(quota)/int(period) if quota!='max' else os.cpu_count()),
        affinity=sorted(os.sched_getaffinity(0)),memory_max=int(maximum) if maximum!='max' else 2**60,
        memory_current=int((root/'memory.current').read_text()),
        usage_usec=int(dict(line.split() for line in (root/'cpu.stat').read_text().splitlines())['usage_usec']))

def campaign_cpu():
    # Include child postgres/torch processes even when their cmdline has no campaign path.
    procs={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:
                fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
                procs[int(path.name)]=(int(fields[1]),int(fields[11])+int(fields[12]),
                    str(ROOT).encode() in (path/'cmdline').read_bytes())
            except (OSError,ValueError):pass
    owned={p for p,(_,_,own) in procs.items() if own}
    while True:
        children={p for p,(parent,_,_) in procs.items() if parent in owned}
        if children<=owned:break
        owned|=children
    return sum(v[1] for p,v in procs.items() if p in owned)/os.sysconf('SC_CLK_TCK')

def ml_reservations(host):
    """Protect ML task affinities, inference cores and unallocated worker RAM."""
    blocked=set();worker_slots=0;memory_growth=0
    for stage in ['orpo','orpo_rl']:
        root=DEPENDENCY/stage
        path=root/'services'/(host+'.status.json')
        if not path.exists():continue
        status=load(path);active=[]
        for task in status.get('tasks',[]):
            state_path=root/'state'/(task+'.json')
            if not state_path.exists():continue
            state=load(state_path)
            if state.get('status')=='running' and state.get('host')==host:
                active.append(state);blocked.update(state['cpus'])
                worker_slots+=len(state['cpus'])
        if active:
            memory_growth=max(memory_growth,max(0,
                status.get('reserved_memory_gib',0)-status.get('memory_gib',0))*2**30)
    inference_slots=0
    if host=='LLMOptimizer2':
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                cmd=[x.decode() for x in (path/'cmdline').read_bytes().split(b'\0') if x]
                if (str(DEPENDENCY/'services/qwen_inference_server.py') not in cmd
                        or 'replica' not in cmd or '--config' not in cmd):continue
                config=load(cmd[cmd.index('--config')+1])
                blocked.update(config['cpus']);inference_slots+=len(config['cpus'])
            except (OSError,ValueError,UnicodeError):continue
    scheduling=load(ROOT/'scheduling.json')
    orpo_cpus=set(scheduling.get('orpo_reserved_cpus',{}).get(host,[]))
    orpo_extra=len(orpo_cpus-blocked)
    blocked.update(orpo_cpus)
    return dict(blocked_cpus=sorted(blocked),ml_worker_cpu_slots=worker_slots,
        ml_inference_cpu_slots=inference_slots,ml_memory_growth_bytes=int(memory_growth),
        orpo_reserved_cpus=sorted(orpo_cpus),orpo_additional_cpu_slots=orpo_extra)

def capacity(host,domain,other_cpu=0):
    r=resources();reserve=6 if host=='LLMOptimizer2' else 2
    slots=3 if domain=='query' else 1
    external=ml_reservations(host)
    charged_other=max(other_cpu,external['ml_worker_cpu_slots']+
        external['ml_inference_cpu_slots']+external['orpo_additional_cpu_slots'])
    cpu_count=max(0,math.floor((r['cpus']-reserve-charged_other)/slots))
    # Memory is checked again before every admission, including DB page cache.
    return min(200,cpu_count),dict(r,reserved_cpu=reserve,other_cpu=other_cpu,
        charged_other_cpu=charged_other,cpus_per_worker=slots,**external)

def available_slots(groups,blocked_cpus,target):
    blocked=set(blocked_cpus)
    return [i for i,group in enumerate(groups) if not blocked.intersection(group)][:target]

def database(host,index,cpus,recipe):
    directory=ROOT/'query/databases'/host/f'w{index:02}'
    port=20400+index
    ready=directory/'PG_VERSION'
    if not ready.exists():
        if directory.exists():raise RuntimeError('Partial database extraction requires audit: '+str(directory))
        directory.mkdir(parents=True,mode=0o700)
        if shutil.disk_usage(directory).free<20*2**30:raise RuntimeError('Insufficient database storage')
        subprocess.run(['tar','-xf',recipe['query']['database_archive'],'-C',str(directory),'--strip-components=1'],check=True)
        subprocess.run(['chown','-R','10000:10000',str(directory)],check=True)
        # Validate clean archive structure before ever starting PostgreSQL.
        if (directory/'postmaster.pid').exists() or not ready.exists():raise RuntimeError('Invalid offline database')
    socket_dir=Path('/tmp')/f'semanticopt_pg_{port}'
    socket_dir.mkdir(mode=0o700,exist_ok=True)
    os.chown(socket_dir,10000,10000)
    opts=['-p',str(port),'-h','127.0.0.1','-k',str(socket_dir),
        '-c','dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib',
        '-c','max_connections=12','-c','shared_buffers=1GB','-c','work_mem=64MB',
        '-c','max_parallel_workers_per_gather=2','-c','max_parallel_workers=2',
        '-c','temp_file_limit=8GB','-c','autovacuum=off']
    base=PREFIX+[PG,'-D',str(directory)]
    if subprocess.run(base+['status'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=ENV).returncode:
        with (ROOT/'logs'/f'{host}.database{index:02}.log').open('ab') as log:
            subprocess.run(['taskset','-c',','.join(map(str,cpus))]+base+
                ['-l',str(directory/'semanticopt.log'),'-o',shlex.join(opts),'-w','-t','90','start'],
                stdout=log,stderr=subprocess.STDOUT,env=ENV,cwd='/tmp',check=True)
    import psycopg
    with psycopg.connect(host='127.0.0.1',port=port,dbname='imdb',user='imdb',password='imdb',
                         autocommit=True,connect_timeout=10) as conn:
        conn.execute("LOAD 'pg_hint_plan'")
        if conn.execute('SELECT count(*) FROM title').fetchone()[0]<1000000:raise ValueError('Wrong IMDb database')
    return port

def stop_databases(host):
    for directory in (ROOT/'query/databases'/host).glob('w*'):
        subprocess.run(PREFIX+[PG,'-D',str(directory),'-m','fast','-w','-t','60','stop'],
            env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def controller(host,domain,smoke=False):
    guard=lock(ROOT/domain/(host+'.controller.lock'))
    recipe=verify();settings=recipe[domain]
    jobs={};failures={};r=resources()
    cpus=r['affinity'][6:] if host=='LLMOptimizer2' else r['affinity'][2:]
    width=3 if domain=='query' else 1
    groups=[cpus[i:i+width] for i in range(0,len(cpus)-width+1,width)]
    previous=(time.monotonic(),resources()['usage_usec'],campaign_cpu())
    while True:
        for index,job in list(jobs.items()):
            code=job['process'].poll()
            if code is None:continue
            job['lock'].close();job['log'].close();del jobs[index]
            if code:
                failures[job['task']]=failures.get(job['task'],0)+1
                status(ROOT/domain/'errors'/f'{host}.{job["task"]}.json',status='error',task=job['task'],
                    returncode=code,log=str(job['log_path']))
                (ROOT/'PAUSE').touch()
                raise RuntimeError('Evaluation failed; campaign paused, see '+str(job['log_path']))
        if (ROOT/'PAUSE').exists():raise RuntimeError('Campaign paused')
        target,allocation=capacity(host,domain)
        now=time.monotonic();usage=resources()['usage_usec'];own=campaign_cpu()
        elapsed=now-previous[0]
        if elapsed>=3:
            other=max(0,(usage-previous[1])/1e6/elapsed-(own-previous[2])/elapsed)
            previous=(now,usage,own)
            target,allocation=capacity(host,domain,other)
        allowed=available_slots(groups,allocation['blocked_cpus'],min(target,1 if smoke else 200))
        target=len(allowed)
        for index,job in jobs.items():
            if index not in allowed:job['yield_path'].touch()
        tasks=[settings['smoke_task']] if smoke else settings['test']
        pending=[]
        for task in tasks:
            folder=ROOT/domain/('smoke' if smoke else 'runs')/task
            if (folder/'status.json').exists() and load(folder/'status.json').get('status')=='complete':continue
            pending.append(task)
        status(ROOT/domain/(host+'.status.json'),status='running' if pending or jobs else 'complete',
            domain=domain,target_workers=target,active_workers=len(jobs),allocation=allocation,
            pending=len(pending),available_slots=allowed,
            workers=[dict(slot=i,task=j['task'],pid=j['process'].pid,cpus=groups[i]) for i,j in jobs.items()])
        if not pending and not jobs:
            if domain=='query':stop_databases(host)
            return
        for index in allowed:
            if index in jobs:continue
            current=resources();need=(12 if domain=='query' else 2)*2**30
            startup_reserve=0
            for job in jobs.values():
                try:
                    fields=dict(line.split(':',1) for line in (Path('/proc')/str(job['process'].pid)/'status').read_text().splitlines())
                    rss=int(fields.get('VmRSS','0 kB').split()[0])*1024
                except OSError:rss=0
                # Charge the not-yet-resident part of each admitted Python worker.
                startup_reserve+=max(0,2*2**30-rss)
            if current['memory_max']-current['memory_current']<(
                    need+12*2**30+startup_reserve+allocation['ml_memory_growth_bytes']):break
            selected=None
            for task in pending:
                try:claim=lock(ROOT/domain/'claims'/(task+'.lock'))
                except BlockingIOError:continue
                selected=task;break
            if selected is None:break
            port=database(host,index,groups[index],recipe) if domain=='query' else 0
            path=ROOT/'logs'/f'{domain}.{host}.slot{index:02}.{selected}{".smoke" if smoke else ""}.log'
            log=path.open('ab')
            yield_path=ROOT/domain/'yield'/f'{host}.slot{index:02}'
            yield_path.parent.mkdir(parents=True,exist_ok=True)
            yield_path.unlink(missing_ok=True)
            command=['taskset','-c',str(groups[index][0]),PYTHON,'-u',str(ROOT/'code/evaluate.py'),domain,
                selected,'--port',str(port)]+(['--smoke'] if smoke else [])
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env=dict(ENV,SEMANTICOPT_YIELD_FILE=str(yield_path)),start_new_session=True)
            jobs[index]=dict(process=process,lock=claim,log=log,log_path=path,task=selected,yield_path=yield_path)
            pending.remove(selected)
        time.sleep(5)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--domain',choices=['query','peptide'],default='query')
    p.add_argument('--probe',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.probe:
        verify();print(canonical(dict(host=a.host,query=capacity(a.host,'query'),peptide=capacity(a.host,'peptide'))))
    else:controller(a.host,a.domain,a.smoke)
