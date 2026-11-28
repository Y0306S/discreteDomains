"""Explicit live CPU-slot admission on Optimizer4; preserve SQL settings."""
import argparse
from common import *
from runtime import validate,fleet

p=argparse.ArgumentParser();p.add_argument('count',type=int);a=p.parse_args()
require_host('LLMOptimizer4')
policy=load(ROOT/'placement.json');validate(policy)
assert policy['admitted_workers'] < a.count <= len(policy['workers'])
assert load(ROOT/'bolt.status.json')['live']==policy['admitted_workers']
assert not load(ROOT/'bolt.status.json')['failures']
memory=int(Path('/sys/fs/cgroup/memory.current').read_text())
limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
assert limit-memory>48*1024**3, 'Insufficient measured host memory reserve'
checks=[]
for spec in policy['workers'][policy['admitted_workers']:a.count]:
    assert not (ROOT/'processes'/f'sql{spec["slot"]:02}.json').exists()
    checks.append(dict(slot=spec['slot'],**fleet.database_check(spec)))
old=policy['admitted_workers'];policy['admitted_workers']=a.count
atomic(ROOT/'placement.json',policy)
atomic(ROOT/'admissions'/f'{a.count:02}.json',dict(previous=old,admitted=a.count,
    memory_current=memory,memory_limit=limit,database_checks=checks,at=time.time()))
print(json.dumps(dict(previous=old,admitted=a.count,memory_gib=memory/1024**3)))
