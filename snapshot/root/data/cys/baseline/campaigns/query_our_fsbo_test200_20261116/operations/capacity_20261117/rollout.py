"""Checkpointed resource expansion, independent of optimization policy."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'batch_code'), str(ROOT/'operations/recovery')]
import resources as resource
import recovery
from query_protocol import atomic


def probes():
    result = [recovery.probe('LLMOptimizer2'), recovery.host_call('LLMOptimizer4','probe')]
    atomic(HERE/'quiescence.json', result)
    return result


def quiescent():
    assert (ROOT/'PAUSE').exists(), 'Maintenance must remain paused'
    result = probes()
    assert all(not p['unrecorded_workers'] for p in result), 'Unknown worker exists'
    assert all(not w['alive'] for p in result for w in p['workers']), 'Workers still draining'
    assert all(d['ready'] and not d['active'] for p in result for d in p['databases']), 'SQL not quiescent'
    controller = recovery.load(ROOT/'services/batch_controller.process.json')
    assert not recovery.process_matches(controller), 'Controller still draining'
    return result


def stop(name):
    record = recovery.load(ROOT/'services'/f'{name}.process.json',{})
    if not recovery.process_matches(record):
        return
    os.kill(record['pid'], signal.SIGTERM)
    deadline = time.monotonic()+30
    while recovery.process_matches(record):
        assert time.monotonic()<deadline, f'{name} did not stop'
        time.sleep(.2)


def wait_health(timeout=240):
    deadline=time.monotonic()+timeout
    while True:
        try:
            return resource.health()
        except (OSError,ValueError,RuntimeError) as error:
            if time.monotonic()>deadline:
                raise
            print('Waiting for services:',type(error).__name__,str(error),flush=True)
            time.sleep(3)


def expand():
    current = quiescent()
    if not (HERE/'journal_audit.json').exists():
        print('Journal audit',recovery.audit_journals(HERE,current),flush=True)
    health=resource.json_get('http://127.0.0.1:21811/health')
    assert health['queued']==0 and not any(r.get('busy') for r in health['replica_health']), 'Broker still has work'
    cfg = json.loads((HERE/'before/resources.json').read_text())
    cfg['llm'] += json.loads((HERE/'candidate_llm.json').read_text())
    for i,gpu in enumerate([0,0,1,2,3,3],start=10):
        cfg['vae'].append(dict(name=f'vae{i}',gpu=gpu,cpu=30+i-10,port=21820+i))
    cfg['hosts']['LLMOptimizer2']['service_cpus']=list(range(24,38))+[39]
    cfg['hosts']['LLMOptimizer2']['reserve_cpus']=[38]
    cfg['service_cpu_policy']='Bounded inference/VAE services share service CPUs; evaluator and PostgreSQL CPU sets are unchanged. CPU 38 retains the monitor.'
    cfg['llm_stage_policy']='Eight verified replicas across four GPUs; shared ORPO queue without cache affinity; BOLT task affinity preserved.'
    cfg['capacity_expansion_at']=time.time()
    for row in cfg['llm']:
        h=resource.json_get(f"http://127.0.0.1:{row['port']}/health")
        assert h['ready'] and h['identities']==health['identities'], row
    atomic(ROOT/'resources.json',cfg)
    for role in ['orpo','warm_start']:
        path=ROOT/f'services/{role}.resources.broker.json'
        broker=json.loads(path.read_text())
        broker['replicas']=[f"http://127.0.0.1:{r['port']}" for r in cfg['llm']]
        broker['task_affinity']=role=='warm_start'
        atomic(path,broker)
    resource.vae_services()
    stop('orpo_resources_broker')
    resource.start(['/usr/bin/python3','-u',str(ROOT/'batch_code/qwen_inference_server.py'),'broker','--config',str(ROOT/'services/orpo.resources.broker.json')],'orpo_resources_broker')
    stop('resources_tunnel')
    resource.tunnel()
    atomic(ROOT/'services/active_inference_stage.json',dict(stage='orpo_batch10',role='orpo',replicas=len(cfg['llm']),placement_verified=True,at=time.time(),capacity_expansion=str(HERE)))
    h=wait_health()
    atomic(HERE/'expanded_health.json',h)
    atomic(HERE/'maintenance.json',dict(state='expanded_paused',at=time.time(),llm=len(cfg['llm']),vae=len(cfg['vae'])))
    print('EXPANDED',len(cfg['llm']),'LLM',len(cfg['vae']),'VAE',flush=True)


def select(count):
    quiescent()
    cfg=resource.config()
    if len(cfg['llm'])==count:
        return
    assert count in [4,8]
    all_llm=json.loads((HERE/'before/resources.json').read_text())['llm']+json.loads((HERE/'candidate_llm.json').read_text())
    stop('orpo_resources_broker')
    cfg['llm']=all_llm[:count]
    cfg['llm_stage_policy']=f'{count} verified replicas across four GPUs; shared ORPO queue without cache affinity; BOLT task affinity preserved.'
    atomic(ROOT/'resources.json',cfg)
    for role in ['orpo','warm_start']:
        path=ROOT/f'services/{role}.resources.broker.json'
        broker=json.loads(path.read_text());broker['replicas']=[f"http://127.0.0.1:{r['port']}" for r in cfg['llm']]
        atomic(path,broker)
    for row in all_llm[count:]:
        stop(row['name']+'_resources')
    resource.start(['/usr/bin/python3','-u',str(ROOT/'batch_code/qwen_inference_server.py'),'broker','--config',str(ROOT/'services/orpo.resources.broker.json')],'orpo_resources_broker')
    active=json.loads((ROOT/'services/active_inference_stage.json').read_text());active.update(replicas=count,at=time.time());atomic(ROOT/'services/active_inference_stage.json',active)
    atomic(HERE/'selected_health.json',wait_health())


def resume():
    quiescent()
    wait_health()
    command=[resource.PYTHON,str(ROOT/'batch_code/validate_launch.py')]
    print(recovery.command(command,timeout=240),flush=True)
    old=json.loads((HERE/'before/evidence/batch_code_pins.json').read_text())
    new=json.loads((ROOT/'evidence/batch_code_pins.json').read_text())
    changed=[p for p,h in old.items() if new.get(p)!=h]
    allowed={str(ROOT/p) for p in ['batch_code/resources.py','resources.json','services/vae_pool.json','services/orpo.resources.broker.json','services/warm_start.resources.broker.json']}
    assert set(changed)<=allowed, changed
    atomic(HERE/'resource_only_changes.json',dict(changed_pins=changed,new_pins=[p for p in new if p not in old],algorithm_pins_unchanged=True,at=time.time()))
    (ROOT/'PAUSE').unlink()
    pid=resource.start([resource.PYTHON,'-u',str(ROOT/'batch_code/run.py'),'controller'],'batch_controller')
    atomic(HERE/'maintenance.json',dict(state='resuming',at=time.time(),controller_pid=pid,llm=len(resource.config()['llm']),vae=len(resource.config()['vae'])))
    print('RESUMING controller',pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['probe','expand','select4','select8','resume']);args=parser.parse_args()
    if args.action=='probe':
        for p in probes():
            print(json.dumps(dict(host=p['host'],alive=sum(w['alive'] for w in p['workers']),ready=sum(d['ready'] for d in p['databases']),active=sum(bool(d['active']) for d in p['databases']),unrecorded=p['unrecorded_workers'])))
    elif args.action=='expand':expand()
    elif args.action.startswith('select'):select(int(args.action[6:]))
    else:resume()
