"""Reuse the audited task state machine with a distinct screened RGPE recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),'/root/data/cys/llmoptimizer_benchmark/tools']
from worker import check_pins


def acquisition(state,recipe,constraint,objective_gp,client):
    import peptide_local_bo as runner
    import numpy as np
    from lolbo.utils.bo_utils.constrained_max_posterior_sampling import MaxPosteriorSampling
    torch=runner.torch
    started=time.monotonic()
    center=torch.as_tensor(state['constraint_z'][runner.policy.incumbent(state['rows'])])
    candidates=runner.policy.candidate_pool(center,state['trust_region'],recipe['n_candidates'],
                                            recipe['local']['perturb_dimensions'])
    fit=client.request('fit',x=runner.transfer.normalized(state['features'],recipe),
        score=np.asarray([r['score'] for r in state['rows']]),
        seed=recipe['seed']+state['progress']['events'])
    best=min(r['mic'] for r in state['rows'] if r['feasible'])
    screen=client.request('screen',candidates=runner.transfer.normalized(candidates.numpy(),recipe),
                          best_mic=best)
    if state['progress']['events']==0 and state['identity']['task']==recipe['allowed_tasks'][0]:
        (ROOT/'evidence/canary_target.snapshot').write_bytes(client.request('snapshot'))
        np.savez(ROOT/'evidence/canary_pool.npz',
                 candidates=runner.transfer.normalized(candidates.numpy(),recipe),best_mic=best)
    indices=np.asarray(screen['indices'])
    if indices.shape!=(200,) or len(set(indices.tolist()))!=200 or np.any(indices<0) or np.any(indices>=1500):
        raise ValueError('Invalid shortlist mapping')
    count=min(recipe['batch_size'],recipe['proposals']-state['progress']['proposals'],
              recipe['fresh_budget']-state['progress']['fresh_evaluations'])
    selected=candidates[torch.as_tensor(indices,dtype=torch.long)]
    model=runner.transfer.BridgeModel(client,recipe)
    started_sample=time.monotonic()
    with torch.no_grad():
        z=MaxPosteriorSampling(model=model,constraint_models=[constraint],constrained=True,
                              replacement=False)(selected,num_samples=count)
    if len(z)!=count or len(torch.unique(z,dim=0))!=count:
        raise ValueError('Joint batch sampler returned duplicate latent candidates')
    info=dict(fit=fit,screen=screen['info'],sample=client.last_sample,
              constrained_sampling_seconds=time.monotonic()-started_sample,
              acquisition_seconds=time.monotonic()-started)
    return z.numpy(),info


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--endpoint',required=True)
    a,remaining=p.parse_known_args()
    check_pins()
    import peptide_local_bo as runner
    task=remaining[remaining.index('--task')+1]
    campaign=Path(remaining[remaining.index('--campaign')+1]).resolve()
    if campaign!=ROOT/'rgpe':raise ValueError('Wrong campaign')
    recipe=runner.verify(campaign)
    if (recipe['n_candidates'],recipe['screen_size'],recipe['batch_size'],recipe['e2e_every'])!=(1500,200,50,0):
        raise ValueError('Incorrect frozen screening protocol')
    original=runner.transfer.ModelClient
    class ScreenClient(original):
        def __init__(self,recipe_path,parent):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            self.child=subprocess.Popen([parent['model_python'],str(ROOT/'code/worker.py'),
                '--endpoint',a.endpoint,'--task',task],stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env)
            self.last_sample=None
    runner.transfer.ModelClient=ScreenClient
    runner.acquisition=acquisition
    sys.argv=[runner.__file__]+remaining
    runner.main()


if __name__=='__main__':main()
