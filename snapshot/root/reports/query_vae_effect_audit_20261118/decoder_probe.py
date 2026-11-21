"""Bounded offline probe; no SQL, training, serving changes or checkpoint selection."""
import collections, hashlib, json, os, random, re, sys, time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.cuda.set_per_process_memory_fraction(.025)
torch.set_float32_matmul_precision('highest')
OUT=Path(__file__).resolve().parent
BASE=Path('/root/data/cys/baseline/campaigns')
NEW=BASE/'query_our_fsbo_vae80_test200_20261117'
OLD=BASE/'query_our_fsbo_test200_20261116'
NATIVE=Path('/root/data/cys/logs/query_train800_author_fixed_20261004/native')
sys.path[:0]=[str(NEW/'batch_code'),str(NATIVE)]
from query_data import canonicalize, definition, plan_key
from vae.model import VAEModule, encode_batch
from workload.workloads import WorkloadSpec,OracleCodec
from oracle.oracle import plan_has_crossjoin

recipe=json.loads((NEW/'batch_recipe.json').read_text())
families={}
for task in recipe['tasks']:
    family=re.match(r'CEB_(\d+[A-Z])',task)[1]
    families.setdefault(family,task)
tasks=list(families.values())
examples=[]
pools=json.loads((NEW/'pools/batch10_frozen_pool.json').read_text())['tasks']
for task in tasks:
    pool=pools[task]
    plans=[r['plan'] for r in pool]
    for plan in plans[:2]:examples.append((task,plan,{plan_key(p) for p in plans}))
result=dict(at=time.time(),tasks=tasks,selection='First query per template in fixed test ordering, first two BOLT plans in generation order; independent of objective results.',examples=len(examples),oracle_calls=0,training_updates=0,models={})
specs={t:WorkloadSpec.from_definition(definition(t)[0],OracleCodec.Aliases) for t in tasks}
for name,root,checkpoint in [('original',OLD,NATIVE/'vae/CEB_64_author.ckpt'),('retrained',NEW,NEW/'training/vae/best.ckpt')]:
    started=time.time();model=VAEModule.load_from_checkpoint(str(checkpoint),map_location='cuda:0').eval().requires_grad_(False)
    cfg=json.loads((root/'fsbo_recipe.json').read_text())
    span=torch.tensor(np.subtract(cfg['upper'],cfg['lower']),device='cuda:0',dtype=torch.float32)
    lower=torch.tensor(cfg['lower'],device='cuda:0',dtype=torch.float32)
    bounds=[]
    counts=collections.defaultdict(collections.Counter);details=[]
    with torch.inference_mode():
        for task in tasks:
            plans=[r['plan'] for r in pools[task]]
            mu,_=model.model.encode(encode_batch(plans,model.vocab).cuda())
            normalized=(mu.reshape(-1,64)-lower)/span
            outside=(normalized<0)|(normalized>1)
            bounds.append(dict(task=task,plans=len(plans),coordinates=outside.numel(),outside_coordinates=int(outside.sum()),plans_with_outside_coordinates=int(outside.any(dim=1).sum()),normalized_min=float(normalized.min()),normalized_max=float(normalized.max())))
        for i,(task,plan,seen) in enumerate(examples):
            mu,_=model.model.encode(encode_batch([plan],model.vocab).cuda());mu=mu.reshape(1,64)
            for regime,sigma in [('encoded_mean',0.),('local_005_span',.05),('local_020_span',.2),('native_initial_candidate',None)]:
                torch.manual_seed(12300+i)
                if sigma is None:
                    # Same unranked candidate geometry as native TuRBO at length .8.
                    offsets=(torch.quasirandom.SobolEngine(64,scramble=True,seed=12300+i).draw(2).cuda()*2-1)*3.2
                    mask=torch.rand(2,64,device='cuda:0')<=20/64
                    for row in range(2):
                        if not mask[row].any():mask[row,0]=True
                    offsets=offsets*mask
                else:
                    offsets=torch.randn(2,64,device='cuda:0')*span*sigma
                z=mu.repeat(2,1)+offsets
                torch.manual_seed(45600+i)
                samples=model.sample(z)
                for raw in samples:
                    c=counts[regime];c['draws']+=1
                    d=dict(task=task,regime=regime,raw=raw)
                    try:
                        canonicalize(task,raw,strict=True);c['strict_raw_valid']+=1;d['strict_raw_valid']=True
                    except (ValueError,TypeError,KeyError,IndexError):d['strict_raw_valid']=False
                    try:
                        cross=plan_has_crossjoin(specs[task],list(raw));d['crossjoin']=cross
                        c['non_crossjoin']+=not cross
                        cp=canonicalize(task,raw);canonicalize(task,cp,strict=True)
                        c['canonical_valid']+=1
                        if not cross:
                            c['production_valid_before_retries']+=1
                            novel=plan_key(cp) not in seen
                            c['novel_vs_bolt30']+=novel;d['novel_vs_bolt30']=novel
                            c['same_as_input']+=plan_key(cp)==plan_key(plan)
                        d['canonical']=cp
                    except Exception as exc:
                        c['codec_errors']+=1;d['codec_error']=repr(exc)
                    details.append(d)
            if (i+1)%10==0:print(name,'examples',i+1,'/',len(examples),flush=True)
    result['models'][name]=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),regimes={k:dict(v) for k,v in counts.items()},warm_start_bounds=bounds,details=details,seconds=time.time()-started)
    del model
    torch.cuda.empty_cache()
    (OUT/'decoder_probe.json').write_text(json.dumps(result,indent=2))
print(json.dumps({**result,'models':{k:{a:b for a,b in v.items() if a!='details'} for k,v in result['models'].items()}},indent=2))
