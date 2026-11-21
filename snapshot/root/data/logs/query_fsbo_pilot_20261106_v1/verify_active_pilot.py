#!/usr/bin/env python3
"""Verify the sixteen admitted workers after parking memory-constrained slots."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from finish_pilot import PLAN, ROOT, atomic, call, median_query, remote_resource


def main():
    active=[w for w in PLAN['workers'] if w.get('admitted')]
    assert len(active)==16 and all(w['host']=='LLMOptimizer4' for w in active)
    assert len({cpu for w in active for cpu in w['cpus']})==48
    baseline_report=json.loads((ROOT/'pilot_LLMOptimizer4.json').read_text())
    assert baseline_report['status']=='complete'
    baselines={r['worker']:median_query(r) for r in baseline_report['serial_queries']}
    for name in ['w00','w05','w06']:
        report=json.loads((ROOT/('pilot_extra_'+name+'.json')).read_text())
        assert report['status']=='complete'
        baselines[name]=median_query(report['serial_query'])
    path=ROOT/'active_16_concurrency_pilot.json'
    if path.exists():raise ValueError('An active pilot has already been recorded')
    before=remote_resource();started=time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as pool:
        results=list(pool.map(lambda w:call(w,'query'),active))
    seconds=time.monotonic()-started;after=remote_resource()
    health=[call(w) for w in active]
    ratios={r['worker']:median_query(r)/baselines[r['worker']] for r in results}
    expected=PLAN['pinned_files'][PLAN['weights']]
    checks=dict(all_16_active_workers=True,
                query_fsbo_checkpoint=all(h['checkpoint_sha256']==expected and h['method']=='fsbo' and h['domain']=='ceb_query_runtime' for h in health),
                bao_warmup=all(h['fits']>=1 for h in health),
                shared_three_core_affinity=all(h['optimizer_cpu'] in h['postgres_cpus'] and len(h['postgres_cpus'])==3 for h in health),
                zero_active_worker_errors=all(h['errors']==0 for h in health),
                no_thompson_or_optimization_campaign=all(h['acquisition_wrapper'] is None and not h['optimization_campaign_started'] for h in health),
                no_remote_cpu_throttling=after['cpu']['nr_throttled']==before['cpu']['nr_throttled'],
                remote_memory_headroom=min((s['memory_maximum']-s['memory_current'])/2**30 for s in [before,after])>=32,
                median_timing_stability=statistics.median(ratios.values())<=1.25,
                individual_timing_stability=max(ratios.values())<=1.5)
    local=Path('/sys/fs/cgroup')
    report=dict(status='complete',passed=all(checks.values()),checks=checks,
                prepared_slots=20,active_workers=16,parked_workers=4,
                allocation={'LLMOptimizer2':0,'LLMOptimizer4':16},
                method='fsbo',domain='ceb_query_runtime',acquisition_wrapper=None,
                optimization_campaign_started=False,results=results,health=health,
                runtime_ratios=ratios,median_runtime_ratio=statistics.median(ratios.values()),
                maximum_runtime_ratio=max(ratios.values()),seconds=seconds,
                remote_before=before,remote_after=after,
                local_memory_headroom_gib=(int((local/'memory.max').read_text())-int((local/'memory.current').read_text()))/2**30,
                limitation='Four local slots are parked after a disk-backed query timeout, then a query-helper startup timeout under local memory pressure. Full campaign remains stopped.')
    atomic(path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['results','health','remote_before','remote_after']}),flush=True)


if __name__=='__main__':main()
