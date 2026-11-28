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


def canonical_initialize(original, recipe, task, client, identity, turbo_class):
    import numpy as np
    from source_screen import atomic
    state = original(recipe, task, client, identity, turbo_class)
    initial = np.load(str(ROOT/'prefilter'/task/'initialization.npz'))
    if ([r['sequence'] for r in state['rows']] != initial['sequences'].tolist()
            or not np.array_equal([r['score'] for r in state['rows']], initial['scores'])
            or not np.allclose(state['features'], initial['features'], rtol=1e-5, atol=1e-5)):
        raise ValueError('Initialization does not reproduce the original experiment')
    # Reuse its exact canonical coordinates, avoiding cross-GPU rounding differences.
    atomic(ROOT/'evidence'/('initialization_'+task+'.json'), dict(passed=True,
        observations=100, sequences_and_scores_identical=True,
        reencoding_max_abs_error=float(np.max(np.abs(state['features']-initial['features']))),
        coordinates='reused_verified_original_frozen_VAE_features'))
    state['features'] = initial['features'].copy()
    state['constraint_z'] = initial['features'].copy()
    return state


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
    from source_screen import atomic
    atomic(ROOT/'telemetry'/state['identity']['task']/('round%d.json'%(state['progress']['rounds']+1)),
           dict(round=state['progress']['rounds']+1, target_observations=len(state['rows']), **info))
    return z.numpy(),info


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--sampling-mode', choices=['bootstrap'], default='bootstrap')
    p.add_argument('--num-posterior-samples', type=int, default=100)
    a,remaining=p.parse_known_args()
    check_pins()
    import peptide_local_bo as runner
    task=remaining[remaining.index('--task')+1]
    campaign=Path(remaining[remaining.index('--campaign')+1]).resolve()
    if campaign!=ROOT/'rgpe':raise ValueError('Wrong campaign')
    recipe=runner.verify(campaign)
    if (a.sampling_mode,a.num_posterior_samples)!=(recipe['rgpe_sampling_mode'],recipe['rgpe_rank_samples']):
        raise ValueError('Sampling flags must match the pinned pilot recipe')
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
    original_initialize=runner.initialize
    runner.initialize=lambda *args: canonical_initialize(original_initialize,*args)
    sys.argv=[runner.__file__]+remaining
    runner.main()


if __name__=='__main__':main()
