from common import *

cfg=load(SEM/'allocation.json');recipe=load(BOLT/'recipe.json')
old=[s for s in recipe['workers'] if s['host']=='LLMOptimizer4']
extra=[s for s in cfg['workers'] if s['host']=='LLMOptimizer4']
# RAM databases first. Keep their proven per-cluster CPU assignments.
extra.sort(key=lambda s:(not s['pgdata'].startswith('/dev/shm'),s['index']))
workers=[dict(s,slot=i,method='bolt') for i,s in enumerate(old+extra)]
atomic(ROOT/'placement.json',dict(recipe_sha256=sha(BOLT/'recipe.json'),workers=workers,
    admitted_workers=12,vae_endpoints=[f'http://127.0.0.1:{24300+i}' for i in range(16)],
    cpu_quota=64,cpu_reserve=4,host='LLMOptimizer4',second_cpu_host='unidentified',
    scientific_recipe_unchanged=True,at=time.time()))
services=[]
for i,gpu in enumerate([15,4,6,3,8,10,12,5,0,1,2,7,9,11,13,14]):
    if i<8:
        record=load(SEM/'query/services'/f'bolt_vae{i:02}.process.json')
        cmd=record['command']
    else:
        cmd=[PYTHON,'-u',str(HYB/'batch_code/query_vae_service.py'),'--gpu',str(gpu),
             '--cpu',str(i-8),'--port',str(23300+i),'--token-file',str(HYB/'services/token'),
             '--native','/root/data/cys/logs/query_train800_author_fixed_20261004/native',
             '--checkpoint',str(HYB/'training/vae/best.ckpt'),'--checkpoint-sha256',
             'eebbe1ee70209aec6b8236d3c01d43c0d3591e52f9bbc23e311c610712e0204d',
             '--memory-fraction','.02']
    services.append(dict(slot=i,gpu=gpu,port=23300+i,command=cmd))
atomic(ROOT/'vae_services.json',dict(services=services))
cmd=['ssh','-N','-T','-i','/root/.ssh/bolt_focus_vae','-o','IdentitiesOnly=yes',
     '-o','UserKnownHostsFile=/root/.ssh/bolt_focus_known_hosts','-o','StrictHostKeyChecking=yes',
     '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
     '-p','36449']
for i in range(16):cmd+=['-L',f'127.0.0.1:{24300+i}:127.0.0.1:{23300+i}']
cmd+=['root@ssh-cn-huabei1.ebcloud.com']
atomic(ROOT/'forward.json',dict(command=cmd))
print(json.dumps(dict(workers=len(workers),admitted=12,vae_services=len(services))))
