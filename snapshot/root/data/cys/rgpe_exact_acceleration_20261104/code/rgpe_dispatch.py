#!/usr/bin/env python3
"""Host-local, quota/memory guarded dispatch for a pinned peptide local campaign.

CPU slots are used only after earlier BORA owners have drained. Shared task
claims prevent duplicate work across hosts. The optimizer recipe is read-only.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path[:0]=["/root/data/cys/llmoptimizer_benchmark/tools","/root/reports/peptide_rgpe_sobol_recovery_20261102/code"]
from sobol_repair import verify as verify_sobol


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def processes():
    result={}
    ticks=os.sysconf('SC_CLK_TCK')
    for root in Path('/proc').iterdir():
        if not root.name.isdigit():continue
        try:
            words=[v.decode() for v in (root/'cmdline').read_bytes().split(b'\0') if v]
            stat=(root/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]=='Z':continue
            status=dict(line.split(':',1) for line in (root/'status').read_text().splitlines() if ':' in line)
            result[int(root.name)]=dict(pid=int(root.name),ppid=int(stat[1]),start=stat[19],
                args=words,cpus=sorted(os.sched_getaffinity(int(root.name))),
                seconds=(int(stat[11])+int(stat[12]))/ticks,
                rss=int(status.get('VmRSS','0 kB').split()[0])*1024,
                peak=int(status.get('VmHWM','0 kB').split()[0])*1024)
        except (OSError,ValueError,UnicodeError):pass
    return result


def descendants(snapshot,pid):
    selected={pid};changed=True
    while changed:
        added={p for p,row in snapshot.items() if row['ppid'] in selected}-selected
        changed=bool(added);selected.update(added)
    return [snapshot[p] for p in selected if p in snapshot]


def blockers(snapshot,slots,own_pids,plan):
    blocked={}
    for pid,row in snapshot.items():
        if pid in own_pids:continue
        # Ignore wide, unbound interactive processes. A pinned foreign workload
        # must release its CPU before RGPE can claim it, regardless of its label.
        cpus=set(row['cpus'])
        relevant=cpus.intersection(slots)
        if not relevant or len(cpus)>2:continue
        for cpu in relevant:
            blocked.setdefault(cpu,[]).append(dict(pid=pid,args=row['args'][:6]))
    return blocked


def admission_limit(quota,reserve,nonworker_cores,maximum):
    return max(0,min(maximum,math.floor(quota-reserve-max(0,nonworker_cores))))


def memory_admissible(ceiling,current,headroom,active,reservation):
    growth=sum(max(0,reservation-rss) for rss in active)
    return ceiling-current-headroom-growth>=reservation


def claim(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close();return None
    return stream


def recoverable_sobol_exit(record):
    legacy = any(str(word).endswith('/peptide_rgpe_merged_launch.py')
                 for word in record['command'])
    tail = Path(record['log']).read_text(errors='replace')[-20000:]
    return legacy and 'Supported range of dimensionality for SobolEngine is [1, 1111]' in tail


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text());pool=plan['hosts'][args.host]
    verify_sobol(plan['sobol_repair'])
    gpu_config=json.loads(Path(plan['gpu_config']).read_text())
    gpu_validation=json.loads(Path(gpu_config['validation']).read_text())
    if (not gpu_validation['passed'] or digest(gpu_config['validation'])!=gpu_config['validation_sha256']):raise ValueError('GPU ranking validation failed')
    for name in gpu_config['numerical_code']:
        if digest(name)!=gpu_validation['code'][name]:raise ValueError('Numerical implementation differs from validation')
    for name,expected in gpu_config['code'].items():
        if digest(name)!=expected:raise ValueError('GPU execution code changed: '+name)
    if os.uname().nodename!=pool['hostname']:raise ValueError('Wrong dispatch host')
    if digest(__file__)!=plan['dispatcher_sha256']:raise ValueError('Dispatcher changed')
    campaign=Path(plan['campaign']);recipe=json.loads((campaign/'recipe.json').read_text())
    if digest(campaign/'recipe.json')!=plan['recipe_sha256']:raise ValueError('Recipe changed')
    if (recipe['method']!='rgpe' or recipe['evaluation_scope']!='test200'
            or len(recipe['allowed_tasks'])!=200 or recipe['vae_policy']!='frozen_author'
            or recipe['e2e_every']!=0 or recipe['batch_size']!=50):
        raise ValueError('Expected frozen RGPE test200 recipe')
    if digest(recipe['adapter'])!=plan['adapter_sha256']:raise ValueError('Adapter changed')
    from peptide_rgpe_merged_policy import verify
    amendment = verify(plan['acquisition_amendment'])
    if (digest(plan['acquisition_amendment']) != plan['acquisition_amendment_sha256']
            or amendment['campaign'] != str(campaign)):
        raise ValueError('Acquisition amendment changed')
    if not args.execute:
        print(json.dumps(dict(host=args.host,pool=pool,tasks=200,launched=False)));return
    root=campaign/'dispatch';root.mkdir(exist_ok=True)
    owner=claim(root/(args.host+'.lock'))
    if owner is None:raise RuntimeError('Host dispatcher already owns its lock')
    os.sched_setaffinity(0,{pool['coordinator_cpu']})
    status_path=root/(args.host+'.json')
    previous=json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous and previous['plan_sha256']!=digest(args.plan):
        if previous['plan_sha256']!=plan.get('upgrade_from_plan_sha256'):
            raise ValueError('Recovery plan changed without an explicit versioned upgrade')
    children={};stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        # Optimizer SIGTERM drains the current durable event. Never cancel its
        # model subprocess or oracle request separately.
        for pid,record in children.items():
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    snapshot=processes()
    for old in previous.get('active',[]):
        pid=old['pid'];row=snapshot.get(pid)
        if row and row['start']==old['start'] and str(campaign) in row['args']:
            lease=claim(root/'claims'/(old['task']+'.lock'))
            if lease is None:raise RuntimeError('Adopted task already claimed')
            children[pid]=dict(old,lease=lease,child=None)
        elif not (campaign/'runs'/(old['task']+'.tar.gz')).exists():
            atomic_json(root/'pause.json',dict(reason='Unexplained prior worker exit',worker=old,at=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env.pop('ML_QWEN_POOL_CONFIG',None);env.pop('PYTHONPATH',None)
    # Take a real interval sample before admissions; never assume the full
    # cgroup quota is free at startup.
    before=None;completed=[];peak_gib=0.
    while True:
        now=time.time();snapshot=processes()
        for pid,record in list(children.items()):
            child=record['child'];returncode=child.poll() if child is not None else None
            alive=pid in snapshot and snapshot[pid]['start']==record['start']
            if alive and returncode is None:continue
            archive=campaign/'runs'/(record['task']+'.tar.gz')
            if not stopping and (returncode not in (None,0) or not archive.exists()):
                if (not any(str(v).endswith('/rgpe_launch.py') for v in record['command'])
                        and record['task'] in plan.get('migration_tasks',[]) and returncode in (None,0)):
                    print(json.dumps(dict(event='cpu_event_drained_for_gpu',task=record['task'],pid=pid)),flush=True)
                elif recoverable_sobol_exit(record):
                    print(json.dumps(dict(event='legacy_sobol_retry',task=record['task'],pid=pid)),flush=True)
                elif record.get('upgrade_requested'):
                    print(json.dumps(dict(event='event_drained_for_exact_cuda_upgrade',task=record['task'],pid=pid)),flush=True)
                else:
                    atomic_json(root/'pause.json',dict(reason='Task exited without successful archive',
                        host=args.host,task=record['task'],pid=pid,returncode=returncode,log=record['log'],at=now))
            else:completed.append(record['task'])
            record['lease'].close();del children[pid]
        own_rows={pid:[v for v in descendants(snapshot,pid)] for pid in children}
        own_pids={v['pid'] for rows in own_rows.values() for v in rows}|{os.getpid()}
        worker_rss=[sum(v['rss'] for v in rows) for rows in own_rows.values()]
        peak_gib=max([peak_gib]+[sum(v['peak'] for v in rows)/2**30 for rows in own_rows.values()])
        reservation=max(pool['memory_per_worker_gib'],peak_gib*1.75)*2**30
        cgroup=Path('/sys/fs/cgroup')
        usage=int(dict(line.split() for line in (cgroup/'cpu.stat').read_text().splitlines())['usage_usec'])/1e6
        worker_time={pid:sum(v['seconds'] for v in rows) for pid,rows in own_rows.items()}
        quota_count,quota_period=(cgroup/'cpu.max').read_text().split()
        quota=float(quota_count)/float(quota_period)
        used=nonworker=None;limit=0
        if before is not None:
            dt=max(.001,now-before['at'])
            used=max(0,(usage-before['usage'])/dt)
            own_used=sum(max(0,t-before['workers'].get(pid,t))/dt for pid,t in worker_time.items())
            nonworker=max(0,used-own_used)
            limit=admission_limit(quota,pool['reserve_cores'],nonworker,pool['max_active'])
        blocked=blockers(snapshot,pool['cpus'],own_pids,plan)
        live_cpus={v['cpu'] for v in children.values()}
        ceiling=int((cgroup/'memory.max').read_text());memory=int((cgroup/'memory.current').read_text())
        paused=(root/'pause.json').exists() or (root/'stop.json').exists()
        completed_tasks={task for task in recipe['allowed_tasks']
                         if (campaign/'runs'/(task+'.tar.gz')).exists()}
        if not stopping and not paused and before is not None:
            # At most two new heavy fits per interval, avoiding synchronized
            # import/GP/ranking bursts on shared storage and memory.
            additions=0
            for cpu in pool['cpus']:
                if len(children)>=limit or additions>=2:break
                if cpu in live_cpus or cpu in blocked:continue
                if not memory_admissible(ceiling,memory,pool['memory_headroom_gib']*2**30,worker_rss,reservation):break
                for task in plan['recovery_priority']+[t for t in recipe['allowed_tasks'] if t not in plan['recovery_priority']]:
                    if task in completed_tasks or task in {v['task'] for v in children.values()}:continue
                    lease=claim(root/'claims'/(task+'.lock'))
                    if lease is None:continue
                    # Guard direct/manual task ownership as well as dispatchers.
                    task_lock=claim(campaign/'locks'/(task+'.lock'))
                    if task_lock is None:
                        lease.close();continue
                    task_lock.close()
                    port=pool['server_ports'][cpu%len(pool['server_ports'])]
                    launcher=plan['gpu_launcher']
                    command=[plan['python'],launcher,'--gpu-config',plan['gpu_config'],'--host',args.host,
                        '--amendment',plan['acquisition_amendment'],
                        'run-task','--campaign',str(campaign),
                        '--task',task,'--cpu',str(cpu),'--server','http://127.0.0.1:'+str(port)]
                    log=root/'logs'/(task+'.'+args.host+'.gpu-ranking.log');log.parent.mkdir(exist_ok=True)
                    with log.open('ab') as stream:
                        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,
                            stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    row=processes().get(child.pid)
                    if row is None:raise RuntimeError('Optimizer exited during launch; inspect '+str(log))
                    record=dict(pid=child.pid,start=row['start'],task=task,cpu=cpu,server_port=port,
                        log=str(log),command=command,started=now,lease=lease,child=child)
                    children[child.pid]=record;worker_rss.append(0);live_cpus.add(cpu);additions+=1
                    print(json.dumps(dict(event='launched',task=task,pid=child.pid,cpu=cpu)),flush=True)
                    break
        active=[{k:v for k,v in record.items() if k not in ('lease','child')} for record in children.values()]
        atomic_json(status_path,dict(host=args.host,pid=os.getpid(),at=now,plan_sha256=digest(args.plan),
            status='draining' if stopping else 'paused' if paused else 'complete' if len(completed_tasks)==200 else 'running',
            active=active,completed=len(completed_tasks),total=200,recent_completed=completed[-20:],
            available_cpu_slots=len(pool['cpus']),blocked_cpus=blocked,max_active=pool['max_active'],
            current_admission_limit=limit,cgroup_used_cores=used,other_used_cores=nonworker,
            quota_cores=quota,memory_gib=memory/2**30,worker_reservation_gib=reservation/2**30,
            observed_worker_peak_gib=peak_gib,launched_after_bora_drain=True))
        if (stopping and not children) or len(completed_tasks)==200:break
        before=dict(at=now,usage=usage,workers=worker_time)
        time.sleep(5)


if __name__=='__main__':main()
