#!/usr/bin/env python3
"""Verify live scheduling, immutable results and post-migration throughput."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import statistics
import time

from accelerate_peptide_20261031 import OUT, CAMPAIGN, TRAIN, SERVICES, history, prefix, read, write, health, digest


def audit(require_progress=False):
    config = read(CAMPAIGN/'fsbo/gpu_runtime.json')
    assert all(digest(Path(path)) == expected for path,expected in config['code_sha256'].items())
    services = read(SERVICES/'services.json')['services']
    healthy = [health(row['endpoint']) for row in services]
    for row, observed in zip(services, healthy):
        observed['actual_cpu_affinity'] = sorted(os.sched_getaffinity(row['pid']))
        assert observed['actual_cpu_affinity'] == [row['cpu']]
    assert {int(h['gpu']) for h in healthy} == {0,1,2,3}
    assert all(h['errors'] == 0 and h['recipe_sha256'] == config['recipe_sha256'] for h in healthy)
    validation = [read(OUT/('fsbo_gpu%d_validation'%i)/'validation.json') for i in range(4)]
    assert all(v['passed'] for v in validation)
    new_fits, verified = [], []
    queue_summary = {}
    for side in ('local','remote'):
        record = read(OUT/('fsbo_'+side+'_drain.json'))
        assert record['status'] == 'drained'
        for task, before in record['baseline'].items():
            h = history(task)
            count = before['drained_rows']
            assert prefix(h, count) == before['drained_prefix_sha256'], 'Changed saved prefix: '+task
            assert h['proposals'] >= before['drained_proposals']
            verified.append(task)
            new_fits.extend(r['fit'] for r in h['observations'][count:] if r.get('fit'))
        queue = read(CAMPAIGN/'workers'/('queue_'+side+'.json'))
        assert time.time()-queue['updated'] < 90
        assert not queue['halted_cpus'], queue['halted_cpus']
        queue_summary[side] = dict(pid=queue['pid'], active=len(queue['active']),
                                   completed=queue['complete'], halted=queue['halted_cpus'])
    for fit in new_fits:
        assert fit['device'] == 'cuda'
    gpu_progress = {str(fit['service_gpu']) for fit in new_fits}
    if require_progress:
        assert gpu_progress == {'0','1','2','3'}, 'Awaiting committed production batches on every GPU'
    old = read(OUT/'rgpe_bank_before.json')
    with sqlite3.connect('file:'+str(TRAIN/'rgpe/models.sqlite')+'?mode=ro',uri=True,timeout=120) as db:
        current = dict(db.execute('SELECT task,sha256 FROM models').fetchall())
    assert all(current.get(task) == value for task,value in old.items())
    rgpe = {}
    active = []
    owners = read(OUT/'rgpe_assignments.json')['owners']
    for name in ('remote','machine3'):
        queue = read(TRAIN/(name+'_queue.json'))
        assert time.time()-queue['updated'] < 30
        for row in queue['rgpe_active'].values():
            assert owners.get(row['task'],'remote') == name
            active.append(row['task'])
        rgpe[name] = dict(workers=queue['compact_workers'], pid=queue['pid'],
                          remaining_assigned=queue['assigned_pending'],
                          available_after_reservations_gib=queue['available_after_reservations_gib'])
    assert len(active) == len(set(active)), 'Duplicate RGPE source fits'
    result = dict(passed=True, checked_at=time.time(),
        fsbo=dict(saved_task_prefixes_verified=len(verified), topology=healthy,
                  full_size_validations_passed=len(validation), queues=queue_summary,
                  post_migration_fits=len(new_fits),
                  post_migration_gpus=sorted(gpu_progress),
                  production_progress_verified=gpu_progress == {'0','1','2','3'}),
        rgpe=dict(queues=rgpe, distinct_live_fits=len(active),
                  completed_models=len(current), immutable_bank_hashes_verified=len(old)),
        gpu_rgpe_experiment=read(OUT/'rgpe_cuda_full_fit/validation.json'))
    if new_fits:
        result['fsbo']['median_queue_seconds'] = statistics.median(f['queue_seconds'] for f in new_fits)
        result['fsbo']['median_service_seconds'] = statistics.median(f['service_seconds'] for f in new_fits)
    write(OUT/'audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'gpu_rgpe_experiment'},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-production-progress', action='store_true')
    audit(parser.parse_args().require_production_progress)
