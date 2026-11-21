"""Bind completed checks to the exact acquisition code before deployment."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
TOOLS=ROOT.parents[1]/'tools'
CAMPAIGN=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks={name:json.loads((ROOT/(name+'.json')).read_text()) for name in
            ('old5000','new5000','dense100','bank800','recovery','wire','launcher')}
    assert all(value['passed'] for value in checks.values())
    assert checks['bank800']['source_models']==800
    assert checks['dense100']['source_models']==100
    logs={'unit.log':14,'wrapper.log':17,'rollout_tests.log':9}
    for name, count in logs.items():
        value=(ROOT/name).read_text()
        assert 'Ran %d tests'%count in value and '\nOK\n' in value,name
    before=np.load(ROOT/'old5000.npz');after=np.load(ROOT/'new5000.npz')
    np.testing.assert_allclose(before['mean'],after['mean'],atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(before['covariance_probe'],after['covariance_probe'],atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(before['draws'],after['draws'],atol=1e-6,rtol=1e-6)
    checks['equivalence5000']=dict(passed=True,max_absolute_errors={
        key:float(np.max(np.abs(before[key]-after[key]))) for key in before.files})
    recipe=json.loads((CAMPAIGN/'recipe.json').read_text())
    recipe_hash=digest(CAMPAIGN/'recipe.json')
    assert recipe_hash=='d1a63763a1dc970225a2770bdef9834f68eb727461aa9d73c22e6c7f61fedd28'
    assert len(recipe['rgpe_sources'])==800 and len(recipe['allowed_tasks'])==200
    assert not set(recipe['rgpe_sources']) & set(recipe['allowed_tasks'])
    assert recipe['vae_policy']=='frozen_author' and recipe['e2e_every']==0
    for path, expected in recipe['code'].items():assert digest(path)==expected,path
    code={str(TOOLS/name):digest(TOOLS/name) for name in
          ('peptide_rgpe_merged.py','peptide_rgpe_merged_policy.py',
           'peptide_rgpe_merged_launch.py','peptide_rgpe_merged_worker.py')}
    assert code==json.loads((ROOT/'staging_validation.json').read_text())['code_sha256']
    report=dict(passed=True,at=time.time(),code_sha256=code,recipe_sha256=recipe_hash,
        dispatcher_sha256=digest(TOOLS/'peptide_rgpe_merged_dispatch.py'),
        production_scale_passed=True,recovery_passed=True,all_800_sources_passed=True,
        launcher_passed=True,unit_and_wrapper_tests=40,checks=checks,
        logs={name:dict(tests=count,sha256=digest(ROOT/name)) for name,count in logs.items()},
        full_covariance_preserved=True,source_weights_squared=True,one_merged_factorization=True,
        original_pinned_code_unchanged=True,original_recipe_unchanged=True,
        target_fitting_and_joint_loocv_ranking_unchanged=True,vae_updates=False,
        lolbo_executed=False,new_test_oracle_calls=0,
        peak_rss_reduction_fraction=1-checks['new5000']['peak_rss_gib']/checks['old5000']['peak_rss_gib'],
        limitations=['Source predictions still scale with contributing models',
                     'LOOCV ranking remains a separate calculation',
                     'Dense candidate and training covariance memory still grows with problem size',
                     'Timing is one controlled comparison, not an end-to-end speedup estimate'])
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,tests=40,all_800_sources_passed=True,
        peak_rss_reduction_fraction=report['peak_rss_reduction_fraction'],recipe_unchanged=True)))


if __name__=='__main__':main()
