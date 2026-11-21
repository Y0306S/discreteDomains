#!/usr/bin/env python3
"""Real query inference and reproducibility checks; no database calls."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
from query_vae_client import QueryVAEPool, QueryVAEBackend


def main():
    pool = QueryVAEPool(ROOT / 'pool.json')
    health = pool.health()
    assert len(health) == 10 and len({h['pid'] for h in health}) == 10
    assert all(h['status'] == 'ready' and h['domain'] == 'ceb_query_runtime' for h in health)
    task = 'CEB_1A1019'
    history = json.loads((Path('/root/data/cys/logs/query_train800_author_fixed_20261004/outputs') /
                          task / 'lolbo/seed0/history.json').read_text())
    initial = history['observations'][:49]
    plans = [r['extra_info']['plan'] for r in initial if not r['extra_info']['censored']][:3]
    assert plans
    backend = QueryVAEBackend(ROOT / 'pool.json', task, plans[0])
    def worker(index):
        client = QueryVAEPool(ROOT / 'pool.json')
        client.endpoints = [pool.endpoints[index]]
        t = time.monotonic()
        enc = client.encode_mean(plans, task=task)
        assert len(enc) == len(plans) and all(len(z) == 64 and all(map(math.isfinite,z)) for z in enc)
        rows = client.decode(enc[:2], task, plans[0], 987)
        assert len(rows) == min(2,len(plans)) and all(r['decode_retries'] <= 51 for r in rows)
        client.encode_mean([r['plan'] for r in rows], task=task)
        assert rows == client.decode(enc[:2], task, plans[0], 987)
        return dict(index=index,z=enc,rows=rows,seconds=time.monotonic()-t)
    t = time.monotonic()
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker,range(10)))
    reference = results[0]
    maxdiff = max(abs(a-b) for r in results for za,zb in zip(r['z'],reference['z']) for a,b in zip(za,zb))
    assert maxdiff < 1e-5, maxdiff
    assert all(r['rows'] == reference['rows'] for r in results)
    z = backend.encode(plans)
    first = backend.decode(z[:1], seed=427)
    backend.decode(z[:1], seed=428)
    assert first == backend.decode(z[:1], seed=427)
    after = pool.health()
    assert all(h['errors'] == 0 and h['oracle_calls'] == 0 for h in after)
    result = dict(status='passed',domain='ceb_query_runtime',task=task,workers=10,
                  concurrent_seconds=time.monotonic()-t,max_encode_difference=maxdiff,
                  checks=['finite_64d_posterior_mean','all_workers_concurrent_query_decode',
                          'canonical_reencode','cross_worker_seed_agreement','interleaved_seed_reproducibility',
                          'task_backend_adapter','zero_oracle_calls'],
                  workers_health=after,results=results)
    (ROOT / 'pool_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['workers_health','results']}),flush=True)


if __name__ == '__main__':
    main()
