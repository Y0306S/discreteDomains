"""Detached host-local supervision for BOLT, VAE processes and direct forwarding."""
import argparse
import fcntl
import signal
import sys
import time
from common import *

sys.path.insert(0,str(OPS))
import fleet


def guard(name):
    p=ROOT/'locks'/f'{name}.lock';p.parent.mkdir(exist_ok=True)
    handle=p.open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return handle


def validate(policy):
    if sha(BOLT/'recipe.json')!=policy['recipe_sha256']:
        raise RuntimeError('BOLT recipe hash mismatch')
    specs=policy['workers']
    if len({s['slot'] for s in specs})!=len(specs) or len({s['db_port'] for s in specs})!=len(specs):
        raise RuntimeError('Duplicate task slot/database')
    cpus=[c for s in specs for c in s['postgres_cpus']+[s['optimizer_cpu']]]
    if len(cpus)!=len(set(cpus)) or len(cpus)>60:
        raise RuntimeError('Overlapping affinities or CPU quota reserve violated')
    if any(s['host']!='LLMOptimizer4' for s in specs):
        raise RuntimeError('This admitted deployment is host-local to Optimizer4')


def worker(number):
    require_host('LLMOptimizer4')
    policy=load(ROOT/'placement.json');validate(policy)
    original_config=fleet.configuration
    original_command=fleet.task_command
    fleet.configuration=lambda kind:(policy['workers'],load(BOLT/'recipe.json')['tasks'],{}) if kind=='bolt' else original_config(kind)
    fleet.task_command=lambda kind,spec,task: ['taskset','-c',str(spec['optimizer_cpu']),PYTHON,'-u',str(ROOT/'task.py'),task] if kind=='bolt' else original_command(kind,spec,task)
    fleet.slot('bolt',number)


def stop(record):
    if alive(record):
        os.kill(record['pid'],signal.SIGTERM)


def supervise(role):
    require_host('LLMOptimizer3' if role=='vae' else 'LLMOptimizer4')
    own=guard(role)
    if role=='bolt':campaign_lock=fleet.lock(BOLT/'control/controller.lock')
    me=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),host='LLMOptimizer3' if role=='vae' else 'LLMOptimizer4',
            role=role,supervisor=str(ROOT/'runtime.py'),at=time.time())
    atomic(ROOT/'processes'/f'{role}_supervisor.json',me)
    if role=='bolt':atomic(BOLT/'control/isolated_controller.process.json',dict(me,kind='bolt'))
    started=time.time();failures={};restart_at={}
    while not (ROOT/f'STOP_{role}').exists():
        policy=load(ROOT/'placement.json');validate(policy)
        if role=='vae':
            specs=load(ROOT/'vae_services.json')['services']
            jobs=[(f'vae{s["slot"]:02}',s['command'],s) for s in specs]
        elif role=='forward':
            specs=[]
            jobs=[('vae_forward',load(ROOT/'forward.json')['command'],{})]
        else:
            specs=policy['workers'][:policy['admitted_workers']]
            jobs=[(f'sql{s["slot"]:02}',[PYTHON,'-u',str(ROOT/'runtime.py'),'worker','--slot',str(s['slot'])],s) for s in specs]
        live=[]
        for label,cmd,spec in jobs:
            record=load(ROOT/'processes'/f'{label}.json')
            if not alive(record):
                if role=='bolt' and record:
                    state=load(fleet.statusfile('bolt',spec['slot']),{})
                    if state.get('status') not in ('idle','paused'):
                        failures[label]=dict(slot=spec['slot'],status=state.get('status'),at=time.time())
                    # Never automatically replay a crashed slot or an uncertain SQL call.
                    continue
                if role=='bolt' and (BOLT/'PAUSE').exists():continue
                if time.time()<restart_at.get(label,0):continue
                record=launch(cmd,label,environment())
                restart_at[label]=time.time()+20
                if role=='vae':atomic(SEM/'query/services'/f'bolt_vae{spec["slot"]:02}.process.json',record)
            if alive(record):live.append((spec,record))
        if role=='bolt':
            # All process identities here are actual Optimizer4 processes, never local SSH launchers.
            children=[(s,fleet.AdoptedProcess(dict(r,slot=s['slot']),'bolt')) for s,r in live]
            status=fleet.summarize('bolt',load(BOLT/'recipe.json')['tasks'],{},children,started)
            remote=load(ROOT/'cpu5.status.json',{})
            remote_fresh=bool(remote and time.time()-remote.get('at',0)<60)
            remote_workers=remote.get('workers',[]) if remote_fresh else []
            remote_admitted=load(ROOT/'placement_cpu5.json',{}).get('admitted_workers',0)
            combined_failures=list(failures.values())+remote.get('failures',[])
            if remote_admitted and not remote_fresh:
                combined_failures.append(dict(host='LLMOptimizer5',reason='Host-local supervisor heartbeat stale; no automatic SQL replay'))
            status.update(supervisor=str(ROOT/'runtime.py'),controller_host='LLMOptimizer4',
                          allocation_amendment=str(ROOT/'placement.json'),vae_replicas=16,
                          admitted_worker_slots=policy['admitted_workers']+remote_admitted,
                          live_worker_slots=len(live)+len(remote_workers),worker_capacity=30,
                          worker_hosts={'LLMOptimizer4':len(live),'LLMOptimizer5':len(remote_workers)},
                          failures=combined_failures,optimizer2_required=False,at=time.time())
            if combined_failures and status['status']=='running':status['status']='degraded'
            atomic(BOLT/'status.json',status)
            atomic(BOLT/'control/workers.process.json',dict(host='LLMOptimizer4',supervisor=str(ROOT/'runtime.py'),
                workers=[dict(slot=s['slot'],host=s['host'],pid=r['pid'],start_ticks=r['start_ticks']) for s,r in live]+
                    [dict(slot=r['slot'],host=r['host'],pid=r['pid'],start_ticks=r['start_ticks']) for r in remote_workers],at=time.time()))
            if status['completed_tasks']==status['total_tasks']:return
        atomic(ROOT/f'{role}.status.json',dict(role=role,host=me['host'],live=len(live),
               failures=failures,processes=[dict(label=label,**(load(ROOT/'processes'/f'{label}.json') or {})) for label,_,_ in jobs],at=time.time()))
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['bolt','vae','forward','worker']);p.add_argument('--slot',type=int)
    a=p.parse_args()
    if a.role=='worker':worker(a.slot)
    else:supervise(a.role)
