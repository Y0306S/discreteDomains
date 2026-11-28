"""Independent second-host worker supervision; existing leases prevent duplication."""
import argparse
import fcntl
import sys
from common import *
sys.path.insert(0,str(OPS));import fleet

def validate(policy):
    assert sha(BOLT/'recipe.json')==policy['recipe_sha256']
    specs=policy['workers'];cpus=[c for s in specs for c in s['postgres_cpus']+[s['optimizer_cpu']]]
    assert len(specs)==15 and len({s['slot'] for s in specs})==15
    assert {s['slot'] for s in specs}==set(range(15,30))
    assert len({s['db_port'] for s in specs})==15
    assert len(cpus)==len(set(cpus))==60
    assert all(s['host']=='LLMOptimizer5' for s in specs)

def worker(slot):
    require_host('LLMOptimizer5');policy=load(ROOT/'placement_cpu5.json');validate(policy)
    fleet.configuration=lambda kind:(policy['workers'],load(BOLT/'recipe.json')['tasks'],{})
    fleet.task_command=lambda kind,spec,task:['taskset','-c',str(spec['optimizer_cpu']),PYTHON,'-u',str(ROOT/'task.py'),task]
    original_env=fleet.environment
    fleet.environment=lambda kind,spec:dict(original_env(kind,spec),BOLT_PLACEMENT=str(ROOT/'placement_cpu5.json'))
    fleet.slot('bolt',slot)

def supervise():
    require_host('LLMOptimizer5')
    handle=(ROOT/'locks/cpu5.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    me=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),host='LLMOptimizer5',at=time.time())
    atomic(ROOT/'processes/cpu5_supervisor.json',me)
    failures={};restart_at=0
    while not (ROOT/'STOP_cpu5').exists():
        policy=load(ROOT/'placement_cpu5.json');validate(policy)
        record=load(ROOT/'processes/cpu5_forward.json')
        if not alive(record) and time.time()>restart_at:
            record=launch(load(ROOT/'forward_cpu5.json')['command'],'cpu5_forward',environment());restart_at=time.time()+20
        live=[];states=[]
        for spec in policy['workers'][:policy['admitted_workers']]:
            label=f'sql{spec["slot"]:02}';record=load(ROOT/'processes'/f'{label}.json')
            if not record and not (BOLT/'PAUSE').exists():
                record=launch([PYTHON,'-u',str(ROOT/'cpu5_runtime.py'),'worker','--slot',str(spec['slot'])],label,environment())
            state=load(fleet.statusfile('bolt',spec['slot']),{})
            if alive(record):live.append(dict(slot=spec['slot'],host=spec['host'],**record))
            elif record and state.get('status') not in ('idle','paused'):
                failures[label]=dict(slot=spec['slot'],host=spec['host'],status=state.get('status'),at=time.time())
            states.append(state)
        atomic(ROOT/'cpu5.status.json',dict(host='LLMOptimizer5',live=len(live),workers=live,states=states,
            admitted_workers=policy['admitted_workers'],capacity=15,failures=list(failures.values()),at=time.time()))
        time.sleep(10)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('role',choices=['supervise','worker']);p.add_argument('--slot',type=int);a=p.parse_args()
    if a.role=='worker':worker(a.slot)
    else:supervise()
