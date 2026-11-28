"""Read-only production verification on Optimizer4 after scaling."""
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from common import *

policy=load(ROOT/'placement.json');baseline=load(ROOT/'cutover_baseline.json')
status=load(BOLT/'status.json')
assert status['controller_host']=='LLMOptimizer4' and not status['optimizer2_required']
assert policy['admitted_workers']==15
assert status['worker_hosts']['LLMOptimizer4']==15
assert status['live_worker_slots']==sum(status['worker_hosts'].values())
assert not status['failures']
assert alive(load(ROOT/'processes/bolt_supervisor.json'))
assert alive(load(ROOT/'processes/forward_supervisor.json'))
assert alive(load(ROOT/'processes/vae_forward.json'))
workers=[]
for spec in policy['workers']:
    state=load(BOLT/'services'/f'worker{spec["slot"]:02}.status.json')
    record=load(ROOT/'processes'/f'sql{spec["slot"]:02}.json')
    assert alive(record) and state['status']=='running' and state['host']=='LLMOptimizer4'
    assert state['port']==spec['db_port']
    progress=load(BOLT/'runs'/state['task']/'progress.json',{})
    before=baseline['progress'].get(state['task'],{}).get('rows',30)
    workers.append(dict(slot=spec['slot'],task=state['task'],port=spec['db_port'],
        pid=record['pid'],child_pid=state['child_pid'],rows=progress.get('rows',30),
        evaluation_delta=progress.get('rows',30)-before))
assert len({w['task'] for w in workers})==15
def health(url):
    with urllib.request.urlopen(url+'/health',timeout=120) as r:return dict(endpoint=url,**json.load(r))
with ThreadPoolExecutor(max_workers=16) as pool:services=list(pool.map(health,policy['vae_endpoints']))
assert len({s['gpu'] for s in services})==16
assert all(s['checkpoint_sha256']=='eebbe1ee70209aec6b8236d3c01d43c0d3591e52f9bbc23e311c610712e0204d' for s in services)
delta=status['evaluations']-baseline['status']['evaluations']
assert delta>0
result=dict(at=time.time(),controller_host='LLMOptimizer4',vae_host='LLMOptimizer3',
    optimizer2_required=False,live_workers=workers,services=services,post_cutover_evaluations=delta,
    status=status,old_forward_cancellations=load(ROOT/'retired_optimizer2_forwards.json'),
    semanticopt_saved_state=load(ROOT/'saved_state_audit.json'),
    memory_gib=int(Path('/sys/fs/cgroup/memory.current').read_text())/1024**3,
    memory_events=Path('/sys/fs/cgroup/memory.events').read_text())
atomic(ROOT/'verified_migration.json',result)
print(json.dumps(dict(live_workers=len(workers),vae_services=len(services),post_cutover_evaluations=delta,
    workers_with_new_evaluations=sum(w['evaluation_delta']>0 for w in workers),
    completed_tasks=status['completed_tasks'],memory_gib=result['memory_gib'])))
