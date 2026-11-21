"""Recipe-preserving, validated amendments for peptide target refitting."""

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effective_recipe(path):
    path = Path(path)
    recipe = json.loads(path.read_text())
    amendment_path = path.with_name('target_fit_amendment.json')
    if not amendment_path.exists():
        return recipe
    amendment = json.loads(amendment_path.read_text())
    if (recipe['method'] != 'fsbo' or amendment['format'] != 'fsbo-target-fit-amendment-v1'
            or amendment['recipe_sha256'] != digest(path)):
        raise ValueError('Target-fit amendment has the wrong recipe identity')
    allowed = {'peptide_transfer_eval.py', 'peptide_transfer_models.py',
               'peptide_transfer_policy.py', 'fsbo_target_refit.py'}
    for name, entry in amendment['code'].items():
        if Path(name).name not in allowed or entry['previous_sha256'] != recipe['code'].get(name):
            raise ValueError('Unrelated code change in target-fit amendment')
        if digest(name) != entry['sha256']:
            raise ValueError('Amended target-fit code changed: ' + name)
        recipe['code'][name] = entry['sha256']
    required = {str(Path(__file__).resolve()), str(Path(__file__).with_name('fsbo_target_refit.py').resolve())}
    if not required <= set(amendment['code']):
        raise ValueError('Target-fit amendment must pin its new dependencies')
    report = amendment['validation']
    if digest(report['path']) != report['sha256']:
        raise ValueError('Target-fit validation report changed')
    validation = json.loads(Path(report['path']).read_text())
    if (not validation['passed'] or validation['code_sha256'] !=
            {name: entry['sha256'] for name, entry in amendment['code'].items()}):
        raise ValueError('Target-fit validation did not pass against the amended code')
    settings = amendment['settings']
    if (not 1 <= settings['maximum_updates'] <= recipe['fsbo_target_epochs']
            or settings['patience'] != recipe['fsbo_target_patience']
            or settings['min_delta'] <= 0 or settings['likelihood'] != 'deterministic_seeded_likelihood'):
        raise ValueError('Unexpected target-fit amendment settings')
    if validation.get('settings') != settings:
        raise ValueError('Target-fit settings differ from the validated settings')
    recipe['fsbo_original_target_epochs'] = recipe['fsbo_target_epochs']
    recipe['fsbo_target_epochs'] = settings['maximum_updates']
    recipe['fsbo_target_stopping'] = settings
    recipe['target_fit_amendment_sha256'] = digest(amendment_path)
    return recipe
