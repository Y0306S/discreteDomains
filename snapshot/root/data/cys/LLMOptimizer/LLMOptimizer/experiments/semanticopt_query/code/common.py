import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path('/root/data/cys/baseline/campaigns/semanticopt_discrete_20261109')
DEPENDENCY = Path('/root/data/cys/baseline/campaigns/ml_orpo_fsbo120_20261109')
QUERY_SOURCE = Path('/root/data/cys/logs/query_train800_author_fixed_20261004')
TOOLS = Path('/root/data/cys/llmoptimizer_benchmark/tools')
PYTHON = '/root/data/cys/envs/bolt_clean_venv/bin/python'
INFER_PYTHON = '/root/data/cys/baseline/envs/taskset_qwen35_fast_20261030/bin/python'
sys.path.insert(0, str(ROOT/'services'))
from protocol import atomic, canonical, digest
from qwen_inference_transport import http_json, model_identity

def load(path):
    return json.loads(Path(path).read_text())

def apply_budget_policy(recipe, policy, policy_sha):
    """Keep the frozen recipe while applying an explicitly audited round cap."""
    if policy['base_recipe_sha256'] != digest(ROOT/'recipe.json'):
        raise ValueError('Budget amendment belongs to a different base recipe')
    size = recipe.get('batch_policy', {}).get('size')
    if size != policy['batch_size'] or size != 50:
        raise ValueError('Budget amendment requires the existing batch size of 50')
    if set(policy['rounds']) != {'peptide', 'query'}:
        raise ValueError('Budget amendment must specify the two SemanticOPT domains')
    for domain, rounds in policy['rounds'].items():
        if type(rounds) is not int or rounds <= 0 or rounds * size > recipe[domain]['proposals']:
            raise ValueError('Round cap must be positive and cannot increase the original budget')
        recipe[domain] = dict(recipe[domain], original_proposals=recipe[domain]['proposals'],
            proposals=rounds * size, rounds=rounds)
    recipe['budget_policy_sha256'] = policy_sha
    return recipe

def domain_recipe(recipe, domain):
    """Resolve the query-only amendment without changing peptide policy hashes."""
    if domain != 'query' or not recipe.get('query_policy'):
        return recipe
    policy = recipe['query_policy']; sha = recipe['query_policy_sha256']
    resolved = dict(recipe, batch_policy=dict(recipe['batch_policy'], size=policy['batch_size']),
        batch_policy_sha256=sha, budget_policy_sha256=sha,
        query=dict(recipe['query'], proposals=policy['proposals'], rounds=policy['rounds'],
                   batch_size=policy['batch_size']))
    if recipe.get('query_round_policy'):
        cap = recipe['query_round_policy']
        resolved['query'] = dict(resolved['query'], rounds=cap['rounds'],
                                 proposals=cap['rounds'] * policy['batch_size'])
        resolved['budget_policy_sha256'] = recipe['query_round_policy_sha256']
    return resolved

def verify():
    identity = load(ROOT/'identity.json')
    for name, expected in identity['files'].items():
        if digest(name) != expected:
            raise ValueError('Pinned file changed: '+name)
    recipe = load(ROOT/'recipe.json')
    amendment = ROOT/'runtime_policy.json'
    if amendment.exists():
        policy = load(amendment)
        if policy['base_recipe_sha256'] != digest(ROOT/'recipe.json'):
            raise ValueError('Runtime amendment belongs to a different base recipe')
        recipe['stages'] = policy['stages']
        recipe['fallback'] = dict(policy['fallback'], sha256=digest(amendment))
        recipe['runtime_policy_sha256'] = digest(amendment)
    batch_file = ROOT/'batch_policy.json'
    if batch_file.exists():
        policy=load(batch_file)
        if policy['base_recipe_sha256']!=digest(ROOT/'recipe.json'):
            raise ValueError('Batch policy belongs to a different recipe')
        recipe['batch_policy']=policy
        recipe['batch_policy_sha256']=digest(batch_file)
    budget_file = ROOT/'budget_policy.json'
    if budget_file.exists():
        apply_budget_policy(recipe, load(budget_file), digest(budget_file))
    oracle_file = ROOT/'peptide_oracle_policy.json'
    if oracle_file.exists():
        policy=load(oracle_file)
        if policy['base_recipe_sha256']!=digest(ROOT/'recipe.json'):
            raise ValueError('Peptide oracle policy belongs to a different recipe')
        recipe['peptide_oracle_policy']=policy
        recipe['peptide_oracle_policy_sha256']=digest(oracle_file)
    deployment = ROOT/'deployment.json'
    if deployment.exists():
        settings=load(deployment)
        if settings['base_recipe_sha256']!=digest(ROOT/'recipe.json'):
            raise ValueError('Deployment belongs to a different base recipe')
        recipe['replicas']=len(settings['gpu_assignments'])
        recipe['gpu_assignments']=settings['gpu_assignments']
    query_file = ROOT/'query_policy.json'
    if query_file.exists():
        policy = load(query_file)
        if (policy['base_recipe_sha256'] != digest(ROOT/'recipe.json') or
                policy['domain'] != 'query' or
                type(policy['batch_size']) is not int or policy['batch_size'] <= 0 or
                type(policy['rounds']) is not int or policy['rounds'] <= 0 or
                policy['batch_size'] * policy['rounds'] != policy['proposals'] or
                policy['proposals'] != recipe['query']['proposals']):
            raise ValueError('Invalid query-only batch and round amendment')
        recipe['query_policy'] = policy
        recipe['query_policy_sha256'] = digest(query_file)
        recipe['query'] = dict(recipe['query'], rounds=policy['rounds'], batch_size=policy['batch_size'])
    cache_file = ROOT/'query_cache_policy.json'
    if cache_file.exists():
        policy = load(cache_file)
        if policy['base_recipe_sha256'] != digest(ROOT/'recipe.json'):
            raise ValueError('Query cache policy belongs to a different recipe')
        recipe['query_cache_policy'] = policy
        recipe['query_cache_policy_sha256'] = digest(cache_file)
    round_file = ROOT/'query_round_policy.json'
    if round_file.exists():
        policy = load(round_file)
        if (policy['base_recipe_sha256'] != digest(ROOT/'recipe.json') or
                policy['query_policy_sha256'] != recipe.get('query_policy_sha256') or
                type(policy['rounds']) is not int or
                not 0 < policy['rounds'] <= recipe['query_policy']['rounds']):
            raise ValueError('Invalid query round cap')
        recipe['query_round_policy'] = policy
        recipe['query_round_policy_sha256'] = digest(round_file)
    return recipe

def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        stream.close(); raise
    return stream

def status(path, **values):
    atomic(path, dict(at=time.time(), pid=os.getpid(), **values))

def task_summary(domain, task):
    path=ROOT/domain/'runs'/task/'status.json'
    return load(path) if path.exists() else {}

def all_complete(domain, recipe):
    recipe = domain_recipe(recipe, domain)
    return all(task_summary(domain,t).get('status')=='complete' and
               task_summary(domain,t).get('evaluations')==recipe[domain]['proposals'] and
               task_summary(domain,t).get('recipe_sha256')==digest(ROOT/'recipe.json') and
               task_summary(domain,t).get('batch_policy_sha256')==recipe.get('batch_policy_sha256') and
               task_summary(domain,t).get('budget_policy_sha256')==recipe.get('budget_policy_sha256') and
               task_summary(domain,t).get('runtime_policy_sha256')==recipe.get('runtime_policy_sha256')
               for t in recipe[domain]['test'])
