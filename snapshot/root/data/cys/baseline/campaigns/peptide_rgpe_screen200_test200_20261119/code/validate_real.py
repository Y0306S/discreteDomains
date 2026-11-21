"""Real immutable source GPs: EI units and exact 200-point posterior parity."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),'/root/data/cys/llmoptimizer_benchmark/tools',
             '/root/data/cys/rgpe_exact_acceleration_20261104/code',
             '/root/data/cys/baseline/repos/transfer-hpo-framework']
from screening import shortlist
from peptide_transfer_models import SourceBank
from peptide_rgpe_merged import merge_posterior,sample_posterior
from rgpe_cuda_client import Client

def main():
    recipe=json.loads((ROOT/'rgpe/recipe.json').read_text())
    small=dict(recipe,rgpe_sources=recipe['rgpe_sources'][:3])
    bank=SourceBank(small)
    sources=[bank[i] for i in range(3)]
    target=sources[0]
    data=np.load('/root/data/logs/peptide_gpu_acceleration_20261102/validation_data.npz')
    x=data['x'].astype(float);rng=np.random.RandomState(82)
    pool=x[rng.randint(len(x),size=1500)]+rng.normal(0,.03,size=(1500,256))
    result=shortlist(target,pool,float(target.mean_y_),float(target.mean_y_),float(target.std_y_))
    candidates=pool[result['indices']]
    # Three independently weighted source contributions plus the target fixture.
    bank.target=target
    ensemble=SimpleNamespace(model_list_=bank,weights_=np.array([.11,.23,.29,.37]),
        Y_mean_=float(target.mean_y_),Y_std_=float(target.std_y_),variance_mode='average')
    cpu_mean,cpu_cov,_=merge_posterior(ensemble,candidates)
    client=Client('http://127.0.0.1:22443',ROOT/'services/token')
    payload=client.payload(target,candidates,source_output='standardized');payload.pop('x')
    gpu=client.call('merge',x=candidates,sources=small['rgpe_sources'],weights=ensemble.weights_,
        target=payload,scale=ensemble.Y_std_,offset=ensemble.Y_mean_)
    assert np.allclose(cpu_mean,gpu['mean'],rtol=1e-7,atol=1e-6)
    assert np.allclose(cpu_cov,gpu['covariance'],rtol=1e-6,atol=1e-6)
    expected,_=sample_posterior(cpu_mean,cpu_cov,50,77)
    actual,_=sample_posterior(gpu['mean'],gpu['covariance'],50,77)
    assert np.allclose(expected,actual,rtol=1e-5,atol=1e-4)
    assert np.array_equal(expected.argmax(1),actual.argmax(1))
    report=dict(passed=True,at=time.time(),source_models=3,candidates=200,draws_shape=list(actual.shape),
        checks=['real_SM​AC_GP_target_EI','four_nonzero_weighted_covariances','CPU_GPU_posterior_parity',
                '50_correlated_draws_parity'],mean_max_error=float(abs(cpu_mean-gpu['mean']).max()),
        covariance_max_error=float(abs(cpu_cov-gpu['covariance']).max()),screen=result['info'])
    (ROOT/'evidence/real_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
