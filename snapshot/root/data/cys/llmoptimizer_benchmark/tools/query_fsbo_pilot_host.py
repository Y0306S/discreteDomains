#!/usr/bin/env python3
"""Provision only this run's query databases and FSBO readiness services."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    os.replace(str(temp), str(path))


def memory():
    root = Path('/sys/fs/cgroup')
    return dict(current=int((root/'memory.current').read_text()), maximum=int((root/'memory.max').read_text()))


def cpu():
    return {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--host', choices=['LLMOptimizer2','LLMOptimizer4'], required=True)
    parser.add_argument('--action', choices=['provision','pilot','status'], required=True)
    parser.add_argument('--worker-names', help='Optional comma-separated subset for staged admission')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    workers = [w for w in plan['workers'] if w['host'] == args.host]
    if args.worker_names:
        selected=set(args.worker_names.split(','))
        workers=[w for w in workers if w['name'] in selected]
        if {w['name'] for w in workers} != selected:
            raise ValueError('Unknown worker selection for this host')
    token = Path(plan['token_file']).read_text().strip()
    state_path = root / ('host_' + args.host + '.json')
    pg = '/root/data/cys/envs/pg16/bin/pg_ctl'
    prefix = ['setpriv','--reuid=10000','--regid=10000','--clear-groups']
    env = dict(os.environ, LD_LIBRARY_PATH='/root/data/cys/envs/pg16/lib:' + os.environ.get('LD_LIBRARY_PATH',''),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def health(w):
        with urllib.request.urlopen('http://127.0.0.1:%d/health'%w['service_port'],timeout=5) as response:
            value=json.load(response)
        if value['worker'] != w['name'] or value['checkpoint_sha256'] != plan['pinned_files'][plan['weights']]:
            raise ValueError('Service identity mismatch: '+w['name'])
        return value

    def request(w,mode):
        req=urllib.request.Request('http://127.0.0.1:%d/pilot'%w['service_port'],
            data=json.dumps(dict(mode=mode)).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=900) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(w['name']+': '+exc.read().decode()) from exc

    if args.action=='status':
        print(json.dumps([health(w) for w in workers]),flush=True)
        return
    if args.action=='provision':
        previous=json.loads(state_path.read_text()) if state_path.exists() else None
        if previous and previous['status']!='ready':
            raise ValueError('Existing provisioning is incomplete; inspect it before restarting')
        existing=previous['workers'] if previous else []
        existing_names={w['name'] for w in existing}
        workers=[w for w in workers if w['name'] not in existing_names]
        if not workers:
            print(json.dumps(dict(status='already_ready',host=args.host)),flush=True)
            return
        initial=memory()
        ram_count=sum(w['storage']=='tmpfs' for w in workers)
        disk_count=sum(w['storage']=='local_disk' for w in workers)
        # Disk data is reclaimable page cache, unlike the tmpfs copies. Budget
        # three GiB of hot cache for the one disk worker, then measure it.
        projected=initial['current'] + (ram_count*8.713+len(workers)*1.75+disk_count*3)*2**30
        reserve=plan['memory_headroom_gib'][args.host]*2**30
        if projected > initial['maximum']-reserve:
            raise RuntimeError('Memory admission denied: projected %.1f GiB; require %.1f GiB free'%
                               (projected/2**30,reserve/2**30))
        for w in workers:
            if Path(w['pgdata']).exists():
                raise ValueError('Database path already exists: '+w['pgdata'])
            for port in [w['db_port'],w['service_port']]:
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1',port))
        usage=shutil.disk_usage('/dev/shm')
        if usage.free < (ram_count*8.713+8)*2**30:
            raise RuntimeError('Insufficient tmpfs storage with eight GiB spare')
        if shutil.disk_usage('/root/blockdata').free < (disk_count*8.713+5)*2**30:
            raise RuntimeError('Insufficient local disk with five GiB spare')
        state=dict(host=args.host,status='copying_databases',started=time.time(),workers=list(existing),memory_before=initial)
        new_entries=[]
        atomic(state_path,state)
        source=None
        for w in workers:
            data=Path(w['pgdata']);data.parent.mkdir(parents=True,exist_ok=True)
            if source is None:
                data.mkdir(mode=0o700)
                subprocess.run(['tar','-xf',plan['database_archive'],'-C',str(data),'--strip-components=1'],check=True)
                subprocess.run(['chown','-R','10000:10000',str(data)],check=True)
                source=data
            else:
                subprocess.run(['cp','-a',str(source),str(data)],check=True)
            if w['storage']=='local_disk':
                # Flush and evict only the newly copied, stopped database files.
                # This avoids charging the entire disk copy as resident RAM.
                for file in data.rglob('*'):
                    if file.is_file() and not file.is_symlink():
                        with file.open('rb') as handle:
                            os.fsync(handle.fileno())
                            os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            if (data/'postmaster.pid').exists() or not (data/'PG_VERSION').is_file():
                raise ValueError('Restored database is not an offline PostgreSQL cluster')
            entry=dict(w,status='database_copied')
            state['workers'].append(entry)
            new_entries.append(entry)
            atomic(state_path,state)
            m=memory()
            if m['maximum']-m['current'] < reserve:
                raise RuntimeError('Memory reserve breached while copying databases')
            print(json.dumps(dict(event='database_copied',worker=w['name'],memory_gib=m['current']/2**30)),flush=True)
        state['status']='starting_services';atomic(state_path,state)
        for entry in new_entries:
            data=Path(entry['pgdata'])
            options=['-p',str(entry['db_port']),'-h','127.0.0.1','-k',str(data),
                     '-c','dynamic_library_path=/root/data/cys/envs/hpo_benchmark/lib:/root/data/cys/envs/pg16/lib',
                     '-c','max_connections=20']
            for key,value in plan['postgres_settings'].items():
                options+=['-c',key+'='+value]
            import shlex
            cmd=['taskset','-c',','.join(map(str,entry['cpus']))]+prefix+[pg,'-D',str(data),
                 '-l',str(data/'fsbo_pilot_postgres.log'),'-o',shlex.join(options),'-w','-t','60','start']
            subprocess.run(cmd,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,cwd='/tmp')
            entry['db_pid']=int((data/'postmaster.pid').read_text().splitlines()[0])
            if os.sched_getaffinity(entry['db_pid']) != set(entry['cpus']):
                raise ValueError('PostgreSQL CPU affinity differs from plan')
            folder=root/'workers'/entry['name'];folder.mkdir(parents=True,exist_ok=True)
            cmd=[plan['model_python'],'-u',str(Path(__file__).with_name('query_fsbo_pilot_worker.py')),
                 '--plan',str(args.plan),'--worker',entry['name']]
            child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=(folder/'service.log').open('a'),
                                   stderr=subprocess.STDOUT,start_new_session=True,
                                   env=dict(env,CUDA_VISIBLE_DEVICES=''))
            entry.update(service_pid=child.pid,status='starting',command=cmd)
            atomic(state_path,state)
        # Services import independently while we poll with a deadline.
        deadline=time.monotonic()+900
        pending={entry['name']:entry for entry in new_entries}
        while pending and time.monotonic()<deadline:
            for name,w in list(pending.items()):
                try:
                    result=health(w)
                except (OSError,ValueError):
                    if not Path('/proc/%d'%w['service_pid']).exists():
                        raise RuntimeError('Readiness worker exited: '+name)
                    continue
                w.update(status='ready',health=result)
                pending.pop(name)
                atomic(state_path,state)
                print(json.dumps(dict(event='worker_ready',worker=name)),flush=True)
            if pending:
                time.sleep(2)
        if pending:
            raise TimeoutError('Services did not become ready: '+','.join(pending))
        state.update(status='ready',ready=time.time(),memory_after=memory())
        atomic(state_path,state)
        print(json.dumps(dict(host=args.host,status='ready',workers=len(state['workers']))),flush=True)
        return

    # A bounded pilot only: fit known Bao observations and time known query plans.
    state=json.loads(state_path.read_text())
    if state['status']!='ready':
        raise ValueError('Host provisioning is not ready')
    ready_names={w['name'] for w in state['workers'] if w['status']=='ready'}
    workers=[w for w in workers if w['name'] in ready_names]
    report_path=root/('pilot_'+args.host+'.json')
    if report_path.exists():
        raise ValueError('Pilot report exists; refusing an untracked repeat')
    report=dict(host=args.host,status='warming_fsbo',started=time.time(),workers=len(workers),
                optimization_campaign_started=False,acquisition_wrapper=None,
                fits=[],serial_queries=[],concurrent=[],resource_samples=[])
    atomic(report_path,report)
    stop=__import__('threading').Event()
    def monitor():
        while not stop.is_set():
            report['resource_samples'].append(dict(at=time.time(),memory=memory(),cpu=cpu()))
            stop.wait(1)
    thread=__import__('threading').Thread(target=monitor,daemon=True);thread.start()
    try:
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            for value in pool.map(lambda w:request(w,'fit'),workers):
                report['fits'].append(value);atomic(report_path,report)
                print(json.dumps(dict(event='fsbo_bao_warmup_passed',worker=value['worker'],
                                     points=value['fit']['usable_distinct_uncensored'])),flush=True)
        report['status']='serial_query_baselines';atomic(report_path,report)
        for w in workers:
            value=request(w,'query');report['serial_queries'].append(value);atomic(report_path,report)
            print(json.dumps(dict(event='serial_baseline_passed',worker=w['name'])),flush=True)
        report['status']='concurrent_fit_and_query';atomic(report_path,report)
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            for value in pool.map(lambda w:request(w,'fit_query'),workers):
                report['concurrent'].append(value);atomic(report_path,report)
                print(json.dumps(dict(event='concurrent_fsbo_query_passed',worker=value['worker'])),flush=True)
        report['health']=[health(w) for w in workers]
        if any(h['errors'] for h in report['health']):
            raise ValueError('Worker reports pilot errors')
        report.update(status='complete',finished=time.time())
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),finished=time.time())
        raise
    finally:
        stop.set();thread.join(timeout=3)
        atomic(report_path,report)


if __name__=='__main__':
    main()
