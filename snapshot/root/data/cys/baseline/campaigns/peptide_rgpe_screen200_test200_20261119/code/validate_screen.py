"""Focused validation of new screening semantics and index mapping."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.stats import norm
from screening import shortlist

class GP:
    def predict(self,x,return_std):
        assert return_std
        return x[:,0],x[:,1]

def main():
    target=SimpleNamespace(gp=GP())
    rng=np.random.RandomState(12)
    x=rng.normal(size=(1500,256));x[:,1]=abs(x[:,1])
    x[0,1]=0
    mu,std=x[:,0],x[:,1]
    best,mean,scale=2.,4.,3.
    delta=(best-mean)/scale-mu
    with np.errstate(divide='ignore',invalid='ignore'):
        expected=np.where(std>0,delta*norm.cdf(delta/std)+std*norm.pdf(delta/std),np.maximum(delta,0))
    before=np.random.get_state()
    result=shortlist(target,x,best,mean,scale)
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    assert np.array_equal(result['indices'],np.lexsort((np.arange(1500),-expected))[:200])
    assert len(np.unique(result['indices']))==200
    # A lower deterministic MIC is preferred. Zero variance cannot create NaNs.
    x[:,:2]=[-1.,0.];x[110,0]=-2.;x[111,0]=2.
    tied=shortlist(target,x,best,mean,scale)
    assert tied['indices'][0]==110 and tied['indices'][1]==0 and 111 not in tied['indices']
    assert np.array_equal(tied['indices'],shortlist(target,x,best,mean,scale)['indices'])
    # A positive affine change of raw MIC units preserves ordering.
    assert np.array_equal(tied['indices'],shortlist(target,x,best*7+9,mean*7+9,scale*7)['indices'])
    bad=x.copy();bad[0,0]=np.nan
    try:shortlist(target,bad,best,mean,scale)
    except ValueError:pass
    else:raise AssertionError('Nonfinite input accepted')
    report=dict(passed=True,checks=['EI_matches_independent_formula','MIC_minimization',
        'raw_to_standardized_output_units','deterministic_ties','200_unique_indices',
        'no_screen_RNG_consumption','zero_variance','finite_guard'],source_models_evaluated=0)
    root=Path(__file__).resolve().parents[1]
    (root/'evidence/screen_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()
