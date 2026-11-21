"""Frozen representation RGPE worker with GPU ranking and target-only screening."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
TOOLS = '/root/data/cys/llmoptimizer_benchmark/tools'
ACCEL = '/root/data/cys/rgpe_exact_acceleration_20261104/code'
REPAIR = '/root/reports/peptide_rgpe_sobol_recovery_20261102/code'
sys.path[:0] = [str(ROOT/'code'), TOOLS, ACCEL, REPAIR,
               '/root/data/cys/baseline/repos/transfer-hpo-framework']


def check_pins():
    pins=json.loads((ROOT/'pins.json').read_text())
    for name, expected in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Pinned dependency changed: '+name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--endpoint',required=True)
    p.add_argument('--task',required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU-only model worker required')
    check_pins()
    wire=sys.stdout.buffer
    sys.stdout=sys.stderr
    from sobol_repair import install as sobol_install, verify as sobol_verify
    sobol_verify('/root/reports/peptide_rgpe_sobol_recovery_20261102/manifest.json')
    sobol_install()
    from rgpe_cuda_client import Client, install
    import peptide_transfer_models as models
    import peptide_rgpe_merged_worker as merged
    import numpy as np
    from screening import shortlist
    client=Client(a.endpoint,ROOT/'services/token')
    directory=ROOT/'telemetry'/a.task
    directory.mkdir(parents=True,exist_ok=True)
    client.telemetry=directory/'rgpe_gpu_stage.json'
    install(client)
    recipe=json.loads((ROOT/'rgpe/recipe.json').read_text())
    backend=merged.MergedRGPEBackend(recipe)
    fit_seed=None
    while True:
        req=models.receive(sys.stdin.buffer)
        if req is None:break
        try:
            op=req['op']
            if op=='close':
                models.send(wire,dict(ok=True));break
            if op=='fit':
                fit_seed=req['seed']
                mic=-np.asarray(req['score'])
                client.y=(mic-mic.mean())/(mic.std() or 1.)
                client.source_seconds=client.loo_seconds=0.;client.source_count=0
                client.write_stage('target_fit_cpu',observations=len(mic))
                result=backend.fit(req['x'],req['score'],req['seed'])
                result.update(source_rank_seconds=client.source_seconds,loo_seconds=client.loo_seconds,
                              endpoint=a.endpoint)
            elif op=='screen':
                result=shortlist(backend.bank.target,req['candidates'],req['best_mic'],
                                 backend.ensemble.Y_mean_,backend.ensemble.Y_std_,200)
            elif op=='sample':
                result=backend.sample(req['candidates'],req['count'],req['seed'])
            elif op=='snapshot':result=backend.snapshot()
            elif op=='restore':result=backend.restore(req['payload'])
            else:raise ValueError('Unknown operation: '+op)
            if op in ('fit','screen','sample'):
                info=result.get('info',result)
                merged.atomic_json(directory/('rgpe_'+op+'.json'),dict(at=time.time(),info=info))
                merged.atomic_json(directory/('%s_seed%s.json'%(op,fit_seed)),dict(at=time.time(),info=info))
            models.send(wire,dict(result=result))
        except Exception:
            models.send(wire,dict(error=traceback.format_exc()))


if __name__=='__main__':main()
