"""Inference-only component check: frozen VAE, no oracle or LOLBO loop."""
import copy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
import peptide_local_bo as runner
from peptide_local_bo import bo,torch,transfer,policy

torch.set_num_threads(1)
recipe=runner.train800_settings()
task=recipe['train'][0]
recipe=copy.deepcopy(recipe)
recipe.update(method='tpe',local=dict(policy.DEFAULTS))
bo.restore_rng(dict(python=__import__('random').Random(1).getstate(),
    numpy=np.random.RandomState(1).get_state(),torch=torch.manual_seed(1).get_state(),cuda=[]))
objective,client=transfer.create_objective(recipe,task,'http://127.0.0.1:18871',
    dict(recipe_sha256='frozen-inference-check-20261101',task=task))
ops=[]
original=client.request
def request(value):
    if value['op']=='oracle':
        raise AssertionError('Inference check must make zero oracle calls')
    ops.append(value['op'])
    return original(value)
client.request=request
try:
    objective.vae.eval()
    for p in objective.vae.parameters():
        p.requires_grad_(False)
    weights=bo.cpu_copy(objective.vae.state_dict())
    from lolbo.utils.bo_utils.turbo import TurboState
    state=runner.initialize(recipe,task,client,{'task':task},TurboState)
    z,extra=runner.propose_tpe(state,recipe)
    assert z.shape==(50,256) and len(extra['tpe_pending'])==50
    plan=policy.recenter_plan(state['rows'],50,recipe['local'])
    seeds=plan['inputs'][:8]
    with torch.no_grad():
        local_sequences=list(objective.vae_decode(torch.as_tensor(z)))
        recenter_z,_=objective.vae_forward(seeds)
        recenter_sequences=list(objective.vae_decode(recenter_z))
    for sequence in local_sequences+recenter_sequences:
        bo.validate_sequence(sequence)
    canonical=transfer.encode_mean(client,recenter_sequences)
    assert canonical.shape==(8,256) and torch.isfinite(recenter_z).all()
    assert all(torch.equal(v,objective.vae.state_dict()[k]) for k,v in weights.items())
    report=dict(passed=True,task=task,initialization=len(state['rows']),
        acquisition_batch=len(local_sequences),recenter_examples=len(recenter_sequences),
        acquisition_feasible=sum(bo.similarity(s,recipe['references'][task])>=.75 for s in local_sequences),
        recenter_feasible=sum(bo.similarity(s,recipe['references'][task])>=.75 for s in recenter_sequences),
        vae_weights_unchanged=True,vae_gradients_disabled=True,new_oracle_calls=0,
        executed_lolbo=False,rpc_operations=ops,
        limitation='Single inference-only diagnostic, not a feasibility performance estimate')
    bo.atomic_json(Path(__file__).with_name('frozen_inference_check.json'),report)
    print(json.dumps(report))
finally:
    client.close()
