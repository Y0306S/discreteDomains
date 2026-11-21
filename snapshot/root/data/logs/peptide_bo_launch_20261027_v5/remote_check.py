import json
from pathlib import Path
import sys

ROOT = Path('/root/data/logs/peptide_bo_release_20261027_v5')
sys.path.insert(0, str(ROOT / 'tools'))
import peptide_bo_v5 as runner

source = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')
target = Path('/root/data/logs/peptide_bo_validation_20261027_v5b_remote4')
if '--verify' in sys.argv:
    import torch
    recipe = runner.verify(target)
    sys.path.insert(0, recipe['cpu_upstream'])
    saved = torch.load(target / 'runs/pep_000/checkpoint.pt', map_location='cpu', weights_only=False)
    runner.protocol.check_accounting(recipe, saved['progress'], saved['rows'])
    original = torch.load(recipe['vae'], map_location='cpu', weights_only=False)
    changed = sum(not torch.equal(value, original[key]) for key, value in saved['vae'].items())
    assert saved['progress']['rounds'] == 2 and saved['state']['tot_num_e2e_updates'] == 1
    assert saved['progress']['recenter_fresh'] > 0 and changed > 0
    report = dict(passed=True, host='LLMOptimizer4', recipe_sha256=runner.bo.digest(target / 'recipe.json'),
        vae_changed_tensors=changed, progress={key: value for key, value in saved['progress'].items()
            if key not in ('recenter_plan', 'recenter_optimizer')})
    runner.bo.atomic_json(target / 'report.json', report)
    print(json.dumps(report))
    sys.exit(0)
target.mkdir(exist_ok=False)
recipe = dict(runner.verify(source), purpose='component_check', proposals=150,
              batch_size=50, e2e_every=2, fresh_budget=400)
runner.bo.atomic_json(target / 'recipe.json', recipe)
print(json.dumps(dict(campaign=str(target))))
