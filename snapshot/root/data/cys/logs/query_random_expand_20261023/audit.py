"""Read-only fleet checks, with sanitized process and progress records."""

import json
import os
from pathlib import Path
import socket
import time

ROOT = Path(__file__).resolve().parent
QUERY = Path('/root/data/cys/logs/query_test200_baselines_20261012')
HOSTS = {'cs-94bac-0693e-server':'LLMOptimizer2', 'cs-0b2cd-18deb-server':'LLMOptimizer3',
         'cs-fa498-8793f-server':'LLMOptimizer4'}
HOST = HOSTS[socket.gethostname()]


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def argv(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        if fields[0] in ('Z','X'):
            return []
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except OSError:
        return []


def value(args, flag):
    return args[args.index(flag)+1] if flag in args else None


def physical(cpu):
    p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    return tuple((p/k).read_text().strip() for k in ('physical_package_id','core_id'))


def cpu_stats():
    return {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}


def main():
    workers, held, issues = [],{},[]
    for path in sorted((QUERY/'workers').glob(HOST+'.*.json')):
        w=load(path)
        args=argv(w.get('child_pid',-1))
        if not any(a.endswith('/run_query_ceb_baselines_v3.py') for a in args):
            continue
        if value(args,'--task-id') != w.get('task'):
            continue
        db=load(QUERY/'databases'/path.name)
        db_args=argv(db.get('pid',-1))
        assert '--method' in args
        cpus=sorted(os.sched_getaffinity(w['child_pid']))
        db_cpus=sorted(os.sched_getaffinity(db['pid'])) if db_args else []
        name=path.stem
        if cpus != [w['bo_cpu']] or db_cpus != sorted(w['db_cpus']):
            issues.append({'worker':name,'issue':'affinity mismatch'})
        if value(args,'--method') != 'random' or db.get('pgdata') not in db_args:
            issues.append({'worker':name,'issue':'method or private database mismatch'})
        for cpu in cpus+db_cpus:
            key=physical(cpu)
            if key in held:
                issues.append({'worker':name,'issue':'physical core shared','other':held[key]})
            held[key]=name
        s=w.get('summary',{})
        workers.append(dict(name=name,mode=w.get('mode'),task=w.get('task'),
            child_pid=w['child_pid'],cpus=cpus,db_pid=db['pid'],db_cpus=db_cpus,
            method=value(args,'--method'),budget=value(args,'--budget'),
            proposals=w.get('saved_proposals',0),database_calls=s.get('database_calls',0),
            cache_hits=s.get('cache_hits'),unique_additional_plans=s.get('unique_additional_plans'),
            unresolved_database_calls=s.get('unresolved_database_calls'),error=s.get('error'),
            updated_at=w.get('updated_at')))
    other=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        args=argv(path.name)
        for a in args:
            if a.endswith(('/run_query_ceb_baselines_v3.py','/ml_fixed_campaign.py',
                           '/ml_runtime_service.py','/train.py')):
                if a.endswith('/run_query_ceb_baselines_v3.py') and value(args,'--method')=='random':
                    continue
                other.append(dict(pid=int(path.name),program=Path(a).name,method=value(args,'--method')))
    before,start=cpu_stats(),time.monotonic()
    time.sleep(2)
    after=cpu_stats()
    cg=Path('/sys/fs/cgroup')
    ms={k:int(v) for k,v in (line.split() for line in (cg/'memory.stat').read_text().splitlines())}
    disk={}
    for directory in ('/dev/shm','/root/data','/root/blockdata'):
        s=os.statvfs(directory)
        disk[directory]=dict(free_gib=s.f_bavail*s.f_frsize/2**30,free_files=s.f_favail)
    report=dict(host=HOST,hostname=socket.gethostname(),epoch=time.time(),workers=workers,
        active_random_workers=len(workers),production_workers=sum(w['mode']=='production' for w in workers),
        smoke_workers=sum(w['mode']=='smoke' for w in workers),physical_worker_cores=len(held),issues=issues,
        other_workloads=other,cpu_quota=(cg/'cpu.max').read_text().strip(),
        used_cpu_cores=(after['usage_usec']-before['usage_usec'])/1e6/(time.monotonic()-start),
        throttle_events=after['nr_throttled']-before['nr_throttled'],
        working_memory_gib=(int((cg/'memory.current').read_text())-ms.get('inactive_file',0))/2**30,
        memory_limit_gib=int((cg/'memory.max').read_text())/2**30,disk=disk)
    target=ROOT/f'{HOST}.audit.json'
    temporary=target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(target)
    print(json.dumps({k:v for k,v in report.items() if k!='workers'}))


if __name__=='__main__':
    main()
