from common import *
for i in range(8):
    r=load(SEM/'query/services'/f'bolt_vae{i:02}.process.json')
    if not alive(r) or str(HYB/'batch_code/query_vae_service.py') not in command(r['pid']):
        raise RuntimeError(f'Cannot safely adopt VAE {i}')
    atomic(ROOT/'processes'/f'vae{i:02}.json',dict(r,host='LLMOptimizer3',adopted=True))
print('Adopted eight existing VAE services')
