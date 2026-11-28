import argparse
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from common import *
from cpu5_runtime import validate,fleet

require_host('LLMOptimizer5')
p=argparse.ArgumentParser();p.add_argument('count',type=int);a=p.parse_args()
policy=load(ROOT/'placement_cpu5.json');validate(policy)
assert policy['admitted_workers']<a.count<=15
assert alive(load(ROOT/'processes/cpu5_supervisor.json')) and alive(load(ROOT/'processes/cpu5_forward.json'))
free=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
assert free>48*1024**3
def health(url):
    with urllib.request.urlopen(url+'/health',timeout=120) as response:d=json.load(response)
    assert d['checkpoint_sha256']=='eebbe1ee70209aec6b8236d3c01d43c0d3591e52f9bbc23e311c610712e0204d'
    return d
with ThreadPoolExecutor(max_workers=16) as pool:services=list(pool.map(health,policy['vae_endpoints']))
checks=[]
for spec in policy['workers'][policy['admitted_workers']:a.count]:
    assert load(ROOT/'cpu5'/f'slot{spec["slot"]}'/'ready.json')['ready']
    assert not (ROOT/'processes'/f'sql{spec["slot"]:02}.json').exists()
    checks.append(dict(slot=spec['slot'],**fleet.database_check(spec)))
if policy['admitted_workers']==0:
    atomic(ROOT/'cpu5/baseline.json',dict(at=time.time(),status=load(BOLT/'status.json'),
        progress={t:load(BOLT/'runs'/t/'progress.json',{}) for t in load(BOLT/'recipe.json')['tasks']}))
prior=policy['admitted_workers'];policy['admitted_workers']=a.count;atomic(ROOT/'placement_cpu5.json',policy)
atomic(ROOT/'cpu5'/f'admitted_{a.count:02}.json',dict(at=time.time(),previous=prior,admitted=a.count,
    memory_headroom_gib=free/1024**3,checks=checks,healthy_vae=len(services)))
print(json.dumps(dict(previous=prior,admitted=a.count,healthy_vae=len(services),memory_headroom_gib=free/1024**3)))
