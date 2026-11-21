"""Training-only VAE pool throughput and checkpoint equivalence checks."""
import concurrent.futures
import json
from pathlib import Path
import pickle
import queue
import sqlite3
import sys
import time
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'batch_code'))
from query_data import canonicalize
from query_protocol import atomic
TOKEN=(ROOT/'services/token').read_text().strip()


def call(port,body):
    request=urllib.request.Request(f'http://127.0.0.1:{port}/inference',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+TOKEN})
    with urllib.request.urlopen(request,timeout=240) as response:
        return json.load(response)


def run(name,ports,bodies):
    work=queue.Queue()
    for i,b in enumerate(bodies):work.put((i,b))
    def worker(port):
        replies=[]
        while True:
            try:i,body=work.get_nowait()
            except queue.Empty:return replies
            result=call(port,body)
            assert len(result['rows'])==len(body['z'])
            for row in result['rows']:
                canonicalize(body['task'],canonicalize(body['task'],row['plan']),strict=True)
            replies.append(dict(index=i,port=port,timing=result['timing'],decoded=len(result['rows']),fallbacks=sum(x['fallback'] for x in result['rows'])))
    started=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as executor:
        values=[v for group in executor.map(worker,ports) for v in group]
    elapsed=time.time()-started
    result=dict(name=name,started=started,finished=time.time(),workers=len(ports),requests=len(values),wall_seconds=elapsed,requests_per_hour=len(values)*3600/elapsed,responses=values,training_only=True,SQL_calls=0)
    atomic(HERE/(name+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='responses'}),flush=True)
    return result


def main():
    import numpy as np
    assert (ROOT/'PAUSE').exists()
    with sqlite3.connect('file:'+str(ROOT/'evidence/batch_preflight.sqlite')+'?mode=ro',uri=True) as db:
        rows=pickle.loads(db.execute('SELECT value FROM kv WHERE key=?',('warm_state',)).fetchone()[0])['rows']
    task='CEB_10A19';cfg=json.loads((ROOT/'batch_recipe.json').read_text())
    assert task in cfg['train_tasks'] and task not in cfg['tasks']
    plan=canonicalize(task,rows[0]['plan'],strict=True)
    def encode(port):return call(port,dict(op='encode_mean',plans=[plan],task=task,seed=123))['z']
    ports=list(range(21820,21836))
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:features=list(executor.map(encode,ports))
    assert all(np.allclose(features[0],z,atol=1e-6,rtol=1e-5) for z in features)
    request=dict(op='decode',z=features[0]*10,task=task,fallback=plan)
    bodies=[dict(request,seed=287000+i) for i in range(32)]
    run('vae_warmup_sixteen',ports,[dict(request,z=features[0],seed=123) for _ in ports])
    ten=run('vae_ten',ports[:10],bodies)
    sixteen=run('vae_sixteen',ports,bodies)
    ten2=run('vae_ten_repeat',ports[:10],bodies[:16])
    rate10=(ten['requests']+ten2['requests'])*3600/(ten['wall_seconds']+ten2['wall_seconds'])
    ratio=sixteen['requests_per_hour']/rate10
    result=dict(at=time.time(),ten_workers_per_hour=rate10,sixteen_workers_per_hour=sixteen['requests_per_hour'],sixteen_over_ten=ratio,selected_vae=16 if ratio>1.10 else 10,encodings_equivalent=True,strict_plan_validation=True,SQL_calls=0)
    atomic(HERE/'vae_benchmark.json',result);print('DECISION',json.dumps(result),flush=True)


if __name__=='__main__':main()
