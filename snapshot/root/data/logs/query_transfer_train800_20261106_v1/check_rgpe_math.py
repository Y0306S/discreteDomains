import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/data/cys/baseline/repos/transfer-hpo-framework')
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
from ConfigSpace import ConfigurationSpace,UniformFloatHyperparameter
from smac.epm.util_funcs import get_types
from rgpe.utils import get_gaussian_process
from peptide_rgpe_exact import marginal_likelihood
cs=ConfigurationSpace(); cs.add_hyperparameters([UniformFloatHyperparameter('z%03d'%i,0,1) for i in range(64)])
types,bounds=get_types(cs,None)
gp=get_gaussian_process(cs,types,bounds,np.random.RandomState(123),None)
rng=np.random.RandomState(42);x=rng.uniform(0,0.2,(20,64));y=2+np.sin(x[:,0]*3)
gp._train(x,y,do_optimize=False)
results=[]
for offset in [-0.1,0,0.1]:
 theta=gp.gp.kernel_.theta.copy()+offset;theta[-1]=np.log(0.01)
 expected,gradient=gp.gp.log_marginal_likelihood(theta,eval_gradient=True)
 actual,bounded=marginal_likelihood(gp.gp,theta)
 np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-8)
 np.testing.assert_allclose(bounded,gradient,rtol=1e-7,atol=2e-8)
 results.append(dict(value_error=float(abs(actual-expected)),max_gradient_error=float(abs(gradient-bounded).max())))
report=dict(passed=True,dimensions=64,checks=results)
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
