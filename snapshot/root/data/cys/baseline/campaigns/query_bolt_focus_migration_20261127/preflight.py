"""Database admission, direct endpoint health, deterministic encode/decode probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import sys
from common import *
import runtime

p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');args=p.parse_args()
policy=load(ROOT/'placement.json');runtime.validate(policy)
dbs=[dict(slot=s['slot'],**runtime.fleet.database_check(s)) for s in policy['workers']]
expected='eebbe1ee70209aec6b8236d3c01d43c0d3591e52f9bbc23e311c610712e0204d'
def health(endpoint):
    with urllib.request.urlopen(endpoint+'/health',timeout=60) as r:d=json.load(r)
    assert d['checkpoint_sha256']==expected,d
    return dict(endpoint=endpoint,**d)
with ThreadPoolExecutor(max_workers=16) as pool:services=list(pool.map(health,policy['vae_endpoints']))
assert len({s['gpu'] for s in services})==16
result=dict(at=time.time(),databases=dbs,services=services)
if args.probe:
    sys.path.insert(0,str(BOLT));import run_lolbo
    task=load(BOLT/'recipe.json')['tasks'][0]
    rows=run_lolbo.load_pool(task);fallback=run_lolbo.valid_fallback(task,rows)
    from query_vae_client import QueryVAEBackend
    backend=QueryVAEBackend(load(BOLT/'recipe.json')['vae_pool'],task,fallback)
    backend.pool.endpoints=policy['vae_endpoints']
    z=backend.encode([fallback])
    decoded=backend.decode(z,seed=123)
    assert len(z)==1 and len(z[0])==64 and len(decoded)==1
    result['probe']=dict(task=task,latent_dimension=len(z[0]),decoded=decoded,sql_calls=0)
atomic(ROOT/'preflight.json',result)
print(json.dumps(dict(databases=len(dbs),healthy_vae=len(services),probe=result.get('probe'))))
