#!/usr/bin/env python3
"""Reconstruct a saved TPE study offline; do not ask, decode or score candidates."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import collections,inspect,json,pickle,pickletools,sqlite3,tarfile,warnings
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers._tpe.sampler import _split_trials

OUT=Path(__file__).resolve().parent
SOURCE=Path('/root/data/logs/peptide_test200_baselines_20261027_v2/tpe/runs/pep_925.tar.gz')
def main():
    with tarfile.open(SOURCE,'r|gz') as tar:
        for member in tar:
            if member.name=='runtime.sqlite':data=tar.extractfile(member).read();break
    with sqlite3.connect(':memory:') as c:
        c.deserialize(data)
        records=[pickle.loads(r[0]) for r in c.execute('select value from trials order by number')]
        state=c.execute("select value from meta where key='state'").fetchone()[0]
    # The saved whole state includes Torch RNG tensors. Inspect sampler flags via
    # pickle opcodes; do not import Torch or deserialize tensor state.
    ops=list(pickletools.genops(state));fields={}
    for i,(op,arg,pos) in enumerate(ops):
        if op.name in ('SHORT_BINUNICODE','BINUNICODE') and arg in ['_multivariate','_group','_n_startup_trials','_n_ei_candidates','_constraints_func']:
            fields[arg]=[(o.name,a) for o,a,_ in ops[i+1:i+4]]
    z=np.array([list(r['params'].values()) for r in records])
    checks=dict(trials=len(records)==5100,all_256_parameters=z.shape==(5100,256),
        parameter_bounds=bool(np.all((z>=-4)&(z<=4))),
        scores_match_history=all(r['score']==r['row']['score'] for r in records),
        constraint_sign_matches_feasibility=all((r['constraints'][0]<=0)==r['row']['feasible'] for r in records))
    assert all(checks.values())
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=1,constraints_func=lambda t:t.user_attrs['constraints']))
    for r in records:
        study.add_trial(optuna.trial.create_trial(value=r['score'],user_attrs={'constraints':r['constraints']},system_attrs={'constraints':r['constraints']}))
    n_below=study.sampler._gamma(len(records))
    below,above=_split_trials(study,study.get_trials(deepcopy=False),n_below,True)
    good=dict(total=len(below),initialization=sum(t.number<100 for t in below),feasible=sum(t.system_attrs['constraints'][0]<=0 for t in below))
    stages={}
    for label,a in [('initial100',z[:100]),('first1000',z[100:1100]),('last1000',z[-1000:])]:
        stages[label]=dict(mean_latent_l2=float(np.mean(np.linalg.norm(a,axis=1))),median_max_abs_coordinate=float(np.median(np.max(np.abs(a),axis=1))),fraction_coordinates_abs_gt2=float(np.mean(np.abs(a)>2)))
    best=min(records[100:],key=lambda r:r['row']['mic'])
    report=dict(task='pep_925',source=str(SOURCE),checks=checks,optuna_version=optuna.__version__,saved_sampler_fields=fields,
        latent_stages=stages,tpe_good_set_at_finish=good,
        best_unconstrained_proposal=dict(best['row'],constraint=best['constraints'][0],similarity=.75-best['constraints'][0]),
        scope='One archived task: all 5100 saved records inspected; actual default-gamma good-set reconstruction. No inference or oracle calls.')
    (OUT/'sqlite_sample_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
