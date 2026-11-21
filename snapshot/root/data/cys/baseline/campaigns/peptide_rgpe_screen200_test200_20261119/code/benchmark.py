"""Matched acquisition timing; no oracle calls and no task-state changes."""
import json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),'/root/data/cys/llmoptimizer_benchmark/tools',
             '/root/data/cys/rgpe_exact_acceleration_20261104/code',
             '/root/data/cys/baseline/repos/transfer-hpo-framework']
from worker import check_pins
from screening import shortlist
from rgpe_cuda_client import Client,install
import peptide_rgpe_merged_worker as merged

def main():
    check_pins();os.sched_setaffinity(0,{62})
    client=Client('http://127.0.0.1:22443',ROOT/'services/token');install(client)
    recipe=json.loads((ROOT/'rgpe/recipe.json').read_text())
    backend=merged.MergedRGPEBackend(recipe)
    backend.restore((ROOT/'evidence/canary_target.snapshot').read_bytes())
    pool=np.load(ROOT/'evidence/canary_pool.npz');x=pool['candidates']
    report=dict(at=time.time(),source_models=len(recipe['rgpe_sources']),
        nonzero_weights=int(np.count_nonzero(backend.ensemble.weights_)),
        observations=len(backend.bank.target.gp.X_train_),rows=[],
        scope='same fitted target, source weights, pool and sampling seed; acquisition only; excludes fit and constraint GP')
    for repeat in range(2):
        # Alternate order to expose source-cache/queue sensitivity.
        for screened in ([False,True] if repeat==0 else [True,False]):
            started=time.monotonic()
            screening=shortlist(backend.bank.target,x,float(pool['best_mic']),
                backend.ensemble.Y_mean_,backend.ensemble.Y_std_) if screened else None
            candidates=x[screening['indices']] if screened else x
            result=backend.sample(candidates,50,771)
            report['rows'].append(dict(repeat=repeat,screened=screened,
                candidates=len(candidates),wall_seconds=time.monotonic()-started,
                screen_seconds=screening['info']['seconds'] if screened else 0.,**result['info']))
            (ROOT/'evidence/matched_speed.json').write_text(json.dumps(report,indent=2))
    report['median_full_seconds']=float(np.median([r['wall_seconds'] for r in report['rows'] if not r['screened']]))
    report['median_screened_seconds']=float(np.median([r['wall_seconds'] for r in report['rows'] if r['screened']]))
    report['acquisition_speedup']=report['median_full_seconds']/report['median_screened_seconds']
    report['complete']=True
    (ROOT/'evidence/matched_speed.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
