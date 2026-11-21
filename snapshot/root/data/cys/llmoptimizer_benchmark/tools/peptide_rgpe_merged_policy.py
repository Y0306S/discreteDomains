"""Validate an acquisition-only amendment without changing checkpoint identities."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path):
    path = Path(path)
    amendment = json.loads(path.read_text())
    campaign = Path(amendment['campaign'])
    recipe = json.loads((campaign/'recipe.json').read_text())
    if (amendment['format'] != 'rgpe-merged-acquisition-v1'
            or digest(campaign/'recipe.json') != amendment['recipe_sha256']
            or recipe['method'] != 'rgpe' or recipe['vae_policy'] != 'frozen_author'
            or recipe['e2e_every'] != 0):
        raise ValueError('Invalid frozen RGPE acquisition amendment')
    required = {str(Path(__file__).with_name(name).resolve()) for name in
                ('peptide_rgpe_merged.py', 'peptide_rgpe_merged_policy.py',
                 'peptide_rgpe_merged_launch.py', 'peptide_rgpe_merged_worker.py')}
    if set(amendment['code']) != required:
        raise ValueError('Unexpected acquisition amendment dependencies')
    for name, expected in amendment['code'].items():
        if digest(name) != expected:
            raise ValueError('Acquisition code changed: '+name)
    validation = amendment['validation']
    if digest(validation['path']) != validation['sha256']:
        raise ValueError('Acquisition validation changed')
    report = json.loads(Path(validation['path']).read_text())
    if (not report['passed'] or report['code_sha256'] != amendment['code']
            or not report['production_scale_passed'] or not report['recovery_passed']):
        raise ValueError('Acquisition amendment is not validated')
    return amendment
