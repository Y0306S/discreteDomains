"""Apply the 12-replica resource change only after durable worker checkpoints."""
import fcntl
import json
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'batch_code'), str(ROOT / 'operations/recovery')]
import resources as resource
import recovery
from query_protocol import atomic, digest


def quiescent():
    assert (ROOT / 'PAUSE').exists()
    probes = [recovery.probe('LLMOptimizer2'), recovery.host_call('LLMOptimizer4', 'probe')]
    atomic(HERE / 'quiescence.json', probes)
    summary = [dict(host=p['host'], workers=sum(w['alive'] for w in p['workers']),
                    active_sql=sum(bool(d['active']) for d in p['databases'])) for p in probes]
    print(json.dumps(summary), flush=True)
    assert not any(p['unrecorded_workers'] for p in probes)
    assert all(d['ready'] for p in probes for d in p['databases'])
    return (not any(w['alive'] for p in probes for w in p['workers'])
            and not any(d['active'] for p in probes for d in p['databases'])
            and not recovery.process_matches(recovery.load(ROOT / 'services/batch_controller.process.json', {})))


def main():
    lock = (ROOT / 'operations/recovery/tick.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic() + 600
    while not quiescent():
        assert time.monotonic() < deadline, 'Workers have not reached checkpoints; leave paused for inspection'
        time.sleep(10)
    h = resource.json_get('http://127.0.0.1:21811/health')
    assert h['queued'] == 0 and not any(r['busy'] for r in h['replica_health'])
    cfg = resource.config()
    candidates = json.loads((HERE / 'candidate_llm.json').read_text())
    assert len(cfg['llm']) == 8
    cfg['llm'] += candidates
    cfg['llm_stage_policy'] = 'Twelve verified replicas, three per GPU. Shared ordinary requests; exclusive long-context admission above 16384 tokens. ORPO shared queue; BOLT affinity preserved.'
    cfg['capacity_expansion_at'] = time.time()
    atomic(ROOT / 'resources.json', cfg)
    for role in ['orpo', 'warm_start']:
        p = ROOT / f'services/{role}.resources.broker.json'
        broker = json.loads(p.read_text())
        broker['replicas'] = [f"http://127.0.0.1:{r['port']}" for r in cfg['llm']]
        atomic(p, broker)
    # All old processes must reload the shared gate, not just the new four.
    record = recovery.load(ROOT / 'services/orpo_resources_broker.process.json')
    if recovery.process_matches(record):
        import os
        os.kill(record['pid'], signal.SIGTERM)
    resource.inference_stage('orpo_batch10')
    atomic(HERE / 'expanded_health.json', resource.health())
    atomic(HERE / 'maintenance.json', dict(state='expanded_paused', at=time.time(), llm=12))
    print('TWELVE REPLICAS READY; still paused for context/memory tests', flush=True)


if __name__ == '__main__':
    main()
