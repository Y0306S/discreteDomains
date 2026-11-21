#!/usr/bin/env python3
"""Independent signed-covariance, scaling and row-wise Gaussian sampling checks."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.sched_setaffinity(0,{18})
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0,'/root/data/cys/baseline/repos/transfer-hpo-framework')
import numpy as np
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.epm.util_funcs import get_types
from rgpe.utils import get_gaussian_process
from peptide_rgpe_posterior import full_posterior,merge_posterior,sample_posterior,sample_rank_posterior
import peptide_transfer as transfer


def main():
    cs=ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('z000',0,1))
    types,bounds=get_types(cs,None)
    model=get_gaussian_process(cs,types,bounds,np.random.RandomState(123),None)
    x=np.array([[0.3],[0.5],[0.7]])
    model._train(x,np.array([10.,20.,15.]),do_optimize=False)
    query=np.linspace(0,1,21)[:,None]
    unit_mean,unit_cov=model.gp.predict(query,return_cov=True)
    expected_mean,expected_cov=model._untransform_y(unit_mean,unit_cov)
    assert np.any(expected_cov < -1e-8)
    mean,cov=full_posterior(model,query)
    assert np.allclose(mean,expected_mean) and np.allclose(cov,expected_cov)
    _,clipped=model._predict(query,cov_return_type='full_cov')
    assert not np.allclose(clipped,expected_cov)
    draws=sample_rank_posterior(model,query,16384,123)
    error=np.linalg.norm(np.cov(draws,rowvar=False)-cov)/np.linalg.norm(cov)
    assert error < 0.025,error
    assert np.array_equal(draws,sample_rank_posterior(model,query,16384,123))
    ensemble=SimpleNamespace(variance_mode='average',weights_=np.array([1.]),model_list_=[model],Y_mean_=42.,Y_std_=3.)
    merged,merged_cov,_=merge_posterior(ensemble,query,source_output='standardized')
    assert np.allclose(merged,unit_mean*3+42)
    assert np.allclose(merged_cov,unit_cov*9)
    batch,_=sample_posterior(merged,merged_cov,50,73)
    raw,_=sample_posterior(merged,merged_cov,50,73,maximize_negative_mic=False)
    assert batch.shape==(50,21) and np.array_equal(batch,-raw)
    report=dict(passed=True,signed_covariance_preserved=True,upstream_clipping_bug_reproduced=True,
        row_draws_use_lower_factor_transpose=True,qmc_empirical_covariance_relative_error=float(error),
        standardized_bank_scale_verified=True,target_scale_applied_once=True,
        batch50_shape=list(batch.shape),negative_mic_sign_verified=True,oracle_calls=0,
        implementation_sha256=transfer.digest('/root/data/cys/llmoptimizer_benchmark/tools/peptide_rgpe_posterior.py'))
    transfer.atomic_json(Path(__file__).with_name('posterior_validation.json'),report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
