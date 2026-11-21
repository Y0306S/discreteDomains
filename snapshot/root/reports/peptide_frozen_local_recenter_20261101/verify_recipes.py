"""Verify prepared wrapper recipes and train800 setting parity, no optimizer runs."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
import peptide_local_bo as r

campaign=Path('/root/data/logs/peptide_frozen_local_recenter_20261101_v1')
source=r.train800_settings()
reports={}
for method in ('fsbo','rgpe','tpe','gp','hebo','smac'):
    recipe=r.verify(campaign/method)
    keys=[key for key in r.SHARED_SETTINGS if key not in ('proposals','fresh_budget')]
    checks=dict(shared_settings=all(recipe[key]==source[key] for key in keys),
        budgets=all(recipe[key]==source[key] for key in ('proposals','fresh_budget')),
        train800_scope=recipe['allowed_tasks']==source['train'] and len(recipe['allowed_tasks'])==800,
        train800_pool=recipe['prepared_pool']==source['prepared_pool'],
        frozen_vae=recipe['e2e_every']==0 and recipe['vae_policy']=='frozen_author',
        native_local_settings=recipe['local']==r.policy.DEFAULTS,
        recenter_schedule=recipe['local']['recenter_every']==source['e2e_every']
            and recipe['local']['recenter_epochs']==source['gp_update_epochs'],
        batch_size=recipe['batch_size']==(source['batch_size'] if method in ('fsbo','rgpe','tpe','gp') else 1),
        no_jobs_launched=not (campaign/method/'runs').exists())
    assert all(checks.values()),(method,checks)
    reports[method]=dict(passed=True,checks=checks,recipe_sha256=r.bo.digest(campaign/method/'recipe.json'))
r.bo.atomic_json(Path(__file__).with_name('recipe_verification.json'),dict(passed=True,methods=reports))
print(json.dumps(reports,indent=2))
