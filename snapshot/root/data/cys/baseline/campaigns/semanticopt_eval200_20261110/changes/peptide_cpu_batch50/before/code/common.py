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
    deployment = ROOT/'deployment.json'
    if deployment.exists():
        settings=load(deployment)
        if settings['base_recipe_sha256']!=digest(ROOT/'recipe.json'):
            raise ValueError('Deployment belongs to a different base recipe')
        recipe['replicas']=len(settings['gpu_assignments'])
        recipe['gpu_assignments']=settings['gpu_assignments']
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
    return all(task_summary(domain,t).get('status')=='complete' and
               task_summary(domain,t).get('evaluations')==recipe[domain]['proposals'] and
               task_summary(domain,t).get('recipe_sha256')==digest(ROOT/'recipe.json') and
               task_summary(domain,t).get('batch_policy_sha256')==recipe.get('batch_policy_sha256') and
               task_summary(domain,t).get('runtime_policy_sha256')==recipe.get('runtime_policy_sha256')
               for t in recipe[domain]['test'])
