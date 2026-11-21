import json,sys,os,time
from urllib.error import URLError
from pathlib import Path
import numpy as np
from query_data import build_prompt,definition,canonicalize,plan_key
TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path.insert(0,str(TOOLS))
from query_resource_recovery import evaluate_query
class LLM:
 def request(self,url,body=None,timeout=30):
  # Generation requests carry a durable request_id. A transport retry reuses
  # that exact request and can replay a saved response without starting SQL.
  for attempt in range(4):
   try:return self.http(url,body,timeout=timeout)
   except (URLError,OSError,TimeoutError) as exc:
    if attempt==3:raise
    print(f'Inference transport interrupted; retrying identical request {attempt+2}/4: {type(exc).__name__}',file=sys.stderr,flush=True)
    time.sleep(2**(attempt+1))
 def __init__(self,recipe,stage,url):
  from qwen_inference_transport import http_json
  self.http=http_json;self.recipe=recipe;self.stage=stage;self.url=url;self.identity=recipe['models'][stage]['identity']
  health=self.request(url+'/health',timeout=30)
  if not health.get('ready') or self.identity not in health['identities']:raise ValueError('Wrong inference identity')
  if self.identity['adapter'] and not any(v.get('all_tensors_exact') and v['sha256']==self.identity['adapter_sha256'] for v in health['adapter_verifications']):raise ValueError('Unverified loaded adapter')
 def __call__(self,request):
  from qwen_inference_transport import digest
  messages=build_prompt(request['task'],[],sft=True)[0] if self.stage=='warm_start' else request['messages']
  body=dict(identity=self.identity,messages=messages,settings=dict(self.recipe['sampling_by_stage'][self.stage],enable_thinking=False),task=self.recipe['name']+'/'+self.stage+'/'+request['task'],rng=dict(seed=request['seed'],cuda_rng=None),use_s1b_template=True,
   namespace=dict(campaign=self.recipe['name'],stage=self.stage,task=request['task'],attempt=request['attempt'],retry=request.get('retry',0)))
  body['request_id']=digest(body);response=self.request(self.url+'/generate',body,timeout=3600)
  if response.get('identity')!=self.identity or response.get('request_id')!=body['request_id']:raise ValueError('Mismatched inference response')
  return response

def oracle(task,timeout):
 sys.path.insert(0,'/root/data/cys/logs/query_train800_author_fixed_20261004/code')
 from run_query_ceb_baselines import RuntimeOracle
 from workload.workloads import WorkloadSpec,OracleCodec
 d,_,_,_=definition(task)
 instance=RuntimeOracle(WorkloadSpec.from_definition(d,OracleCodec.Aliases),timeout)
 return lambda plan:evaluate_query(instance,plan,timeout)

class FSBO:
 def __init__(self,recipe,task,pool):
  import torch
  torch.set_num_threads(1)
  import query_fsbo_eval as q
  from query_vae_client import QueryVAEBackend
  self.q=q;self.task=task;self.path=recipe['fsbo_recipe'];self.cfg=json.loads(Path(self.path).read_text());self.limit=recipe['optimization']['fsbo_decode_limit']
  _,_,cross=q.canonical_context(self.cfg['native'],task)
  fallbacks=[r for r in pool if not cross(r['plan'])]
  if not fallbacks:raise ValueError('Warm pool lacks a non-crossjoin VAE fallback')
  # Constrained batch decoding can queue behind another task and perform up
  # to 51 samples per latent vector. This is inference time, not the SQL cap.
  self.vae=QueryVAEBackend(self.cfg['vae_pool'],task,min(fallbacks,key=lambda r:r['runtime'])['plan'],timeout=1800)
  self.model=q.ModelClient(Path(self.path),self.cfg);self.encodings={}
 def __call__(self,rows,seed,count=10,excluded=()):
  import torch
  q=self.q;cfg=self.cfg;seen={plan_key(r['plan']) for r in rows} | {plan_key(p) for p in excluded}
  observed=[r for r in rows if not r['censored']]
  if not observed:raise ValueError('No exact targets for FSBO')
  missing=[r['plan'] for r in observed if plan_key(r['plan']) not in self.encodings]
  if missing:
   z=np.asarray(self.vae.encode(missing),dtype=np.float32)
   if z.shape!=(len(missing),64) or not np.isfinite(z).all():raise ValueError('Bad VAE encoding')
   self.encodings.update((plan_key(p),v.tolist()) for p,v in zip(missing,z))
  features=[self.encodings[plan_key(r['plan'])] for r in observed]
  fit=self.model.request('fit',x=q.normalized(features,cfg),score=np.asarray([-r['runtime'] for r in observed]),seed=seed%(2**31-1))
  turbo=q.peptide_turbo(cfg)
  state=turbo.TurboState(dim=64,batch_size=count,best_value=max(-r['runtime'] for r in observed))
  rejected=[];accepted=[];accepted_decodes=[];posterior=[]
  with torch.random.fork_rng(devices=[]),torch.no_grad():
   torch.manual_seed(seed%(2**63-1));x=torch.tensor(features,dtype=torch.float32);y=torch.tensor([-r['runtime'] for r in observed],dtype=torch.float32).view(-1,1)
   for retry in range(self.limit):
    needed=count-len(accepted)
    z=turbo.generate_batch(state,q.BridgeModel(self.model,cfg),x,y,needed,n_candidates=cfg['n_candidates'],device=torch.device('cpu'))
    posterior.append(self.model.last_sample)
    decoded_batch=self.vae.decode(z.tolist(),seed=(seed+retry)%(2**31-1))
    for decoded in decoded_batch:
     try:
      if decoded['fallback']:raise ValueError('duplicate_or_decoder_fallback')
      # Match the original native FSBO: turn the raw VAE token sequence into
      # its executed join tree, then validate the final canonical plan strictly.
      plan=canonicalize(self.task,decoded['plan'])
      plan=canonicalize(self.task,plan,strict=True)
      if plan_key(plan) in seen:raise ValueError('duplicate_or_decoder_fallback')
     except (ValueError,TypeError,KeyError,IndexError) as exc:rejected.append(dict(decoded=decoded,error=str(exc)));continue
     accepted.append(plan);accepted_decodes.append(dict(raw_vae_tokens=decoded['plan'],plan=plan,native_canonicalized=decoded['plan']!=plan));seen.add(plan_key(plan))
    if len(accepted)==count:break
  return dict(plans=accepted,diagnostics=dict(fit=fit,rejected_decodes=rejected,requested=count,accepted=len(accepted),accepted_decodes=accepted_decodes,posterior=posterior))
 def close(self):self.model.close()
