"""Apply tested replica isolation and resume existing Query histories."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'batch_code'))
sys.path.insert(0, str(ROOT/'operations/recovery'))
import resources
import recovery
from query_protocol import atomic, digest


def main():
    lock = (ROOT/'operations/recovery/tick.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert (ROOT/'PAUSE').exists()
    assert not recovery.process_matches(recovery.load(ROOT/'services/batch_controller.process.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        probes = list(pool.map(lambda h: recovery.probe(h) if h=='LLMOptimizer2'
            else recovery.host_call(h,'probe'), resources.config()['hosts']))
    atomic(OPS/'before_probes.json',probes)
    print('Host audit',json.dumps([dict(host=p['host'],workers=sum(w['alive'] for w in p['workers']),
        ready_databases=sum(d['ready'] for d in p['databases']),
        active_databases=sum(bool(d.get('active')) for d in p['databases'])) for p in probes]),flush=True)
    assert all(not p['unrecorded_workers'] and not any(w['alive'] for w in p['workers']) for p in probes)
    assert all(d['ready'] and not d['active'] for p in probes for d in p['databases'])
    folder=OPS/'incident';folder.mkdir(exist_ok=True)
    audit=recovery.audit_journals(folder,probes)
    print('Journal audit',audit,flush=True)
    old = resources.json_get('http://127.0.0.1:21811/health')
    assert old['queued']==0 and not any(r.get('busy') for r in old['replica_health'])
    atomic(OPS/'broker_before.json',old)
    resources.stop_service('orpo_resources_broker')
    record=recovery.load(ROOT/'services/orpo_resources_broker.process.json')
    deadline=time.monotonic()+30
    while resources.record_alive(record):
        assert time.monotonic()<deadline
        time.sleep(.2)
    journal=ROOT/'services/orpo.requests.sqlite'
    with sqlite3.connect(journal,timeout=120) as source, sqlite3.connect(OPS/'orpo.requests.before.sqlite') as backup:
        source.backup(backup)
    resources.start(['/usr/bin/python3','-u',str(ROOT/'batch_code/qwen_inference_server.py'),
        'broker','--config',str(ROOT/'services/orpo.resources.broker.json')],'orpo_resources_broker')
    deadline=time.monotonic()+60
    while True:
        try:
            h=resources.json_get('http://127.0.0.1:21811/health')
            if h.get('ready') and 'healthy_replicas' in h:break
        except OSError:pass
        assert time.monotonic()<deadline
        time.sleep(.5)
    atomic(OPS/'broker_after.json',h)
    print('Broker ready',h['healthy_replicas'],'/',h['replicas'],flush=True)
    health=resources.health()
    atomic(OPS/'resources_health.json',health)
    pins_path=ROOT/'evidence/batch_code_pins.json';pins=recovery.load(pins_path)
    reviewed=['batch_code/resources.py','batch_code/test_service_recovery.py',
              'batch_code/qwen_inference_server.py','batch_code/test_replica_isolation.py']
    changed={p for p,h in pins.items() if digest(p)!=h}
    assert changed <= {str(ROOT/n) for n in reviewed},changed
    for name in reviewed:pins[str(ROOT/name)]=digest(ROOT/name)
    atomic(pins_path,pins)
    recovery.verify_pins()
    errors=list((ROOT/'orpo_batch10/runs').glob('*/error.json'))
    assert len(errors)==1 and errors[0].parent.name=='CEB_2B362'
    error=recovery.load(errors[0]);assert 'GPU allocation exhausted' in error['error']
    errors[0].rename(folder/'CEB_2B362.error.json')
    assert not (ROOT/'services/resource_error.json').exists()
    (ROOT/'PAUSE').unlink()
    (ROOT/'operations/recovery/recovery_pause.json').unlink(missing_ok=True)
    pid=resources.start([resources.PYTHON,'-u',str(ROOT/'batch_code/run.py'),'controller'],'batch_controller')
    atomic(OPS/'resume.json',dict(at=time.time(),controller_pid=pid,measurements_before=audit['saved_rows'],
        audit=audit,healthy_llms=h['healthy_replicas'],configured_llms=h['replicas']))
    print('Resumed controller',pid,flush=True)


if __name__=='__main__':main()
