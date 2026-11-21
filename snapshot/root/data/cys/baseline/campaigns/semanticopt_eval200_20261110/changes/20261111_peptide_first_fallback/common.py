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
    return load(ROOT/'recipe.json')

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
               task_summary(domain,t).get('recipe_sha256')==digest(ROOT/'recipe.json')
               for t in recipe[domain]['test'])
