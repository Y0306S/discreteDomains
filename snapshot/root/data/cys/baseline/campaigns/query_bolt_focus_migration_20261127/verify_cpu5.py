"""Verify second-host processes, exclusive ownership and saved production commits."""
from common import *
from cpu5_runtime import validate
require_host('LLMOptimizer5')
policy=load(ROOT/'placement_cpu5.json');validate(policy)
assert policy['admitted_workers']==15
assert alive(load(ROOT/'processes/cpu5_supervisor.json'))
assert alive(load(ROOT/'processes/cpu5_forward.json'))
assert not load(ROOT/'cpu5.status.json')['failures']
baseline=load(ROOT/'cpu5/baseline.json');workers=[]
all_tasks=[]
for slot in range(30):
    state=load(BOLT/'services'/f'worker{slot:02}.status.json')
    assert state['status']=='running',state
    all_tasks.append(state['task'])
    if slot<15:continue
    r=load(ROOT/'processes'/f'sql{slot:02}.json');assert alive(r)
    assert state['host']=='LLMOptimizer5' and state['port']==20500+slot-15
    assert alive(dict(pid=state['child_pid'],start_ticks=ticks(state['child_pid'])))
    spec=next(s for s in policy['workers'] if s['slot']==slot)
    assert os.sched_getaffinity(state['child_pid'])=={spec['optimizer_cpu']}
    task=state['task'];progress=load(BOLT/'runs'/task/'progress.json',{})
    before=baseline['progress'].get(task,{}).get('rows',30)
    workers.append(dict(slot=slot,task=task,pid=r['pid'],rows=progress.get('rows',30),
        evaluation_delta=progress.get('rows',30)-before))
assert len(set(all_tasks))==30
assert all(w['evaluation_delta']>0 for w in workers),workers
status=load(BOLT/'status.json')
assert status['live_worker_slots']==30 and status['worker_hosts']=={'LLMOptimizer4':15,'LLMOptimizer5':15}
assert not status['failures']
assert (SEM/'PAUSE').exists() and load(SEM/'status.json')['evaluations']==101438
for path,expected in load(ROOT/'before/bolt/identity.json')['files'].items():assert sha(path)==expected,path
result=dict(at=time.time(),host='LLMOptimizer5',live_workers=workers,
    saved_new_evaluations=sum(w['evaluation_delta'] for w in workers),
    global_status=status,unique_concurrent_tasks=len(set(all_tasks)),
    database_count=len(load(ROOT/'cpu5/databases_ready.json')['databases']),
    cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text(),
    memory_gib=int(Path('/sys/fs/cgroup/memory.current').read_text())/1024**3,
    memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),
    gpu_sample=load(ROOT/'gpu_usage_latest.json'),optimizer2_required=False)
atomic(ROOT/'cpu5/verified.json',result)
print(json.dumps(dict(workers=15,global_workers=30,saved_new_evaluations=result['saved_new_evaluations'],
    memory_gib=result['memory_gib'],completed_tasks=status['completed_tasks'],total_evaluations=status['evaluations'])))
