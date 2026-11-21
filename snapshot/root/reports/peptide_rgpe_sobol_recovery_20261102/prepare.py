import hashlib,json,time
from pathlib import Path
r=Path('/root/reports/peptide_rgpe_sobol_recovery_20261102')
c=Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
v=json.loads((r/'validation.json').read_text());assert v['passed'] and v['tests']==6
assert v['code']=={str(p):sha(p) for p in (r/'code').glob('*.py')}
before=json.loads((r/'before.json').read_text());assert len(before['tasks'])==10
plan=json.loads((c/'dispatch/plan.json').read_text())
deps=[c/'recipe.json',Path(plan['acquisition_amendment']),Path('/root/data/cys/llmoptimizer_benchmark/tools/peptide_rgpe_posterior.py'),Path('/root/data/cys/llmoptimizer_benchmark/tools/peptide_transfer_models.py')]
m=dict(format='rgpe-extended-sobol-v1',at=time.time(),campaign=str(c),
       code=v['code'],dependencies={str(p):sha(p) for p in deps},
       validation=dict(path=str(r/'validation.json'),sha256=sha(r/'validation.json')),
       policy='Legacy NormalQMCEngine unchanged through 1110 dimensions; higher dimensions use seeded scrambled SciPy Sobol uniforms in the existing modern Python environment with the legacy Box-Muller transform and unchanged full posterior Cholesky; independent scrambled uniform blocks only above SciPy MAXDIM 21201. No GP, ranking formula, observation rows or budget changes.',
       recovery=dict(before=str(r/'before.json'),sha256=sha(r/'before.json'),tasks=[x['task'] for x in before['tasks']]))
save(r/'manifest.json',m)
new=dict(plan,previous_plan_sha256=sha(c/'dispatch/plan.json'),at=time.time(),
         dispatcher_sha256=sha(r/'code/repair_dispatch.py'),sobol_repair=str(r/'manifest.json'),
         recovery_priority=m['recovery']['tasks'])
save(c/'dispatch/sobol_repair_plan.json',new)
save(r/'plan_transition.json',dict(previous_path=str(c/'dispatch/plan.json'),previous_sha256=sha(c/'dispatch/plan.json'),new_path=str(c/'dispatch/sobol_repair_plan.json'),new_sha256=sha(c/'dispatch/sobol_repair_plan.json'),recipe_sha256=sha(c/'recipe.json'),adopt_active=True,initial_restart_count=10))
print(json.dumps({'tests':v['tests'],'validation_passed':v['passed'],'tasks':m['recovery']['tasks'],'plan':str(c/'dispatch/sobol_repair_plan.json')},indent=2))
