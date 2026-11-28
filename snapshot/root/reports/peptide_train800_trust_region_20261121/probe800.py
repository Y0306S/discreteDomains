"""All800 source decoder calibration, bounded shared GPU, resumable per task."""
import csv,gzip,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
SHARD=int(os.environ['PROBE_SHARD']);SHARDS=4;OUT=Path(__file__).resolve().parent
os.sched_setaffinity(0,{34+2*SHARD,35+2*SHARD})
ROOT=Path('/root/data/logs/peptide_target_loss_gate_20round_max2_test10_20261119')
UP=Path('/root/data/cys/BOLT/optimization/peptides')
sys.path[:0]=[str(ROOT/'deps'),str(UP)]
import torch,numpy as np,pandas as pd
from Levenshtein import distance
from uniref_vae import data as native_data
sys.modules['data']=native_data
torch.set_num_threads(2);torch.set_num_interop_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.cuda.set_per_process_memory_fraction(.15)
spec=importlib.util.spec_from_file_location('probe_vae',ROOT/'reference/APEXGo/generation/vae.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=native_data.DatasetKmers(k=1,load_data=False,vocab=pd.read_csv(UP/'uniref_vae/1mer_vocab.csv',header=None).values.squeeze().tolist())
model=module.InfoTransformerVAE(dataset=data,d_model=128,kl_factor=.0001,encoder_dim_feedforward=256,decoder_dim_feedforward=256,encoder_num_layers=6,decoder_num_layers=6).cuda()
model.max_string_length=30
cfg=json.loads((ROOT/'config.json').read_text())
pool_path=Path('/root/data/logs/peptide_bo_train800_20261027_v5b/initialization.json')
pool=json.loads(pool_path.read_text())
tasks=sorted(cfg['train_tasks']);assert len(tasks)==800 and len(set(tasks))==800
test=set(json.loads(Path('/root/reports/peptide_train800_test200_audit_20261027/test200_revalidated_initialization.json').read_text()))
assert not set(tasks)&test
split=sorted(tasks,key=lambda t:hashlib.sha256(('trust_region_split_20261121'+t).encode()).hexdigest())
confirm=set(split[640:])
SEEDS=[31001,31002,31003];N=32
def token(seqs):return native_data.collate_fn([data.encode(s).unsqueeze(0) for s in data.tokenize_sequence(seqs)]).cuda()
def write(path,d):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False));t.replace(path)
models=[('adapted',ROOT/'vae/weights.pt',[0.,.1,.125,.15,.175,.2,.3,.8]),('author',Path(cfg['author_vae']),[.15,.2,.8])]
started=time.monotonic();completed=0
for name,path,radii in models:
 model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False),strict=True);model.eval()
 identity=dict(model=name,weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),radii=radii,seeds=SEEDS,draws_per_seed=N,precision='cuda_fp32_tf32_disabled',shard=SHARD)
 for task in tasks[SHARD::SHARDS]:
  result_path=OUT/'tasks'/name/(task+'.json');result_path.parent.mkdir(parents=True,exist_ok=True)
  if result_path.exists():
   old=json.loads(result_path.read_text());assert {k:v for k,v in old['identity'].items() if k!='shard'}=={k:v for k,v in identity.items() if k!='shard'} and old['status']=='complete';completed+=1;continue
  item=pool[task];ref=item['reference'];known=set(item['sequences'])|{ref}
  assert all(distance(s,ref)<=len(ref)//4 for s in item['sequences'])
  anchors=list(dict.fromkeys([ref]+item['sequences'][:3]));rows=[];summaries=[]
  with torch.inference_mode():
   mu,_=model.encode(token(anchors));mu=mu.reshape(-1,256)
   for radius in radii:
    draws=[]
    for seed in SEEDS:
     ids=torch.arange(N,device='cuda')%len(anchors)
     gen=torch.Generator().manual_seed(seed)
     mask=torch.rand((N,256),generator=gen)<=20/256
     empty=torch.where(mask.sum(1)==0)[0];mask[empty,torch.randint(0,255,(len(empty),),generator=gen)]=True
     u=torch.quasirandom.SobolEngine(256,scramble=True,seed=seed).draw(N)
     z=mu[ids].clone()+((u*2-1)*4*radius*mask).cuda()
     torch.manual_seed(seed+1000)
     samples=model.sample(z=z.reshape(-1,2,128))
     for i in range(N):
      seq=data.decode(samples[i]);center=anchors[int(ids[i])]
      er=distance(seq,ref);ec=distance(seq,center)
      syntax=bool(seq) and len(seq)<=30 and seq.count('C')<=1 and not set(seq)-set('ACDEFGHIKLMNPQRSTVWY')
      row=dict(radius=radius,seed=seed,anchor=int(ids[i]),center=center,sequence=seq,edits_reference=er,edits_center=ec,valid=bool(syntax and er<=len(ref)//4),novel=seq not in known)
      draws.append(row);rows.append(row)
    counts={}
    for edits in [1,2,3,100]:
     good=[r for r in draws if r['valid'] and r['edits_center']<=edits]
     novel=[r for r in good if r['novel']]
     per_center={str(i):len({r['sequence'] for r in novel if r['anchor']==i}) for i in range(len(anchors))}
     counts[str(edits)]=dict(valid_draws=len(good),novel_valid_draws=len(novel),unique_novel_valid=len({r['sequence'] for r in novel}),unique_novel_per_center=per_center)
    summaries.append(dict(radius=radius,draws=len(draws),counts=counts,median_reference_edits=float(np.median([r['edits_reference'] for r in draws])),median_center_edits=float(np.median([r['edits_center'] for r in draws]))))
  drawpath=result_path.with_suffix('.jsonl.gz')
  with gzip.open(drawpath,'wt') as f:
   for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
  write(result_path,dict(status='complete',task=task,reference=ref,reference_length=len(ref),global_edit_budget=len(ref)//4,development_split='confirmation160' if task in confirm else 'calibration640',identity=identity,summary=summaries))
  completed+=1
  write(OUT/f'status_{SHARD}.json',dict(status='running',model=name,task=task,completed_model_tasks=completed,target_model_tasks=400,elapsed_seconds=time.monotonic()-started,oracle_calls=0))
  if completed%10==0:print(name,completed,'model tasks',round(time.monotonic()-started,1),'seconds',flush=True)
write(OUT/f'status_{SHARD}.json',dict(status='complete',completed_model_tasks=completed,elapsed_seconds=time.monotonic()-started,oracle_calls=0))
print('COMPLETE',SHARD,completed,flush=True)
