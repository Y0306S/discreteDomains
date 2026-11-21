"""Keep SMAC fitting, priors and RGPE weights; delegate dense arithmetic to CUDA."""
import hashlib,json,pickle,time,urllib.request,uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
class Client:
 def __init__(self,endpoint,token_file):
  self.endpoint=endpoint;self.token=Path(token_file).read_text().strip();self.datasets={};self.y=None;self.calls=0;self.source_seconds=0;self.loo_seconds=0;self.source_count=0;self.telemetry=None
 def call(self,op,**values):
  req=urllib.request.Request(self.endpoint+'/rpc',data=pickle.dumps(dict(op=op,**values),protocol=4),headers={'Content-Type':'application/octet-stream','Authorization':'Bearer '+self.token})
  with urllib.request.urlopen(req,timeout=3600) as r:reply=pickle.loads(r.read())
  self.calls+=1
  if 'error' in reply:raise RuntimeError('CUDA math: '+reply['error'])
  return reply['result']
 def marginal(self,gp,theta):
  from peptide_rgpe_exact import kernel_parts
  identity=id(gp)
  if identity not in self.datasets or self.datasets[identity][0] is not gp:
   kernel_parts(gp.kernel_,gp.X_train_.shape[1]);key=uuid.uuid4().hex;self.datasets[identity]=(gp,key)
   self.call('dataset',key=key,x=gp.X_train_,y=gp.y_train_)
  key=self.datasets[identity][1];reply=self.call('nll',key=key,theta=theta)
  if reply.get('missing'):
   self.call('dataset',key=key,x=gp.X_train_,y=gp.y_train_);reply=self.call('nll',key=key,theta=theta)
  return reply['value'],reply['gradient']
 def normal(self,n,count,seed):
  from botorch.sampling.qmc import NormalQMCEngine
  return NormalQMCEngine(n,seed=int(seed)).draw(count).numpy()
 def payload(self,model,x,source_output='ensemble'):
  from peptide_rgpe_exact import kernel_parts
  gp=model.gp;kernel_parts(gp.kernel_,gp.X_train_.shape[1])
  if gp.normalize_y or not model.normalize_y:raise ValueError('Expected pinned scalar normalized SMAC GP')
  scale,offset=(float(model.std_y_),float(model.mean_y_)) if source_output=='ensemble' else (1.,0.)
  return dict(x=np.asarray(x,dtype=np.float64),train=gp.X_train_,theta=gp.kernel_.theta,L=gp.L_,alpha=gp.alpha_,scale=scale,offset=offset)
 def source_draw(self,model,x,count,seed):
  started=time.monotonic()
  if isinstance(model,LazySource):
   result=self.call('source_rank_id',source=model.task,x=x,normal=self.normal(len(x),count,seed),y=self.y)
  else:
   p=self.payload(model,x);p.update(normal=self.normal(len(x),count,seed),y=self.y);result=self.call('source_rank',**p)
  self.source_seconds+=time.monotonic()-started;self.source_count+=1
  if self.telemetry and (self.source_count==1 or self.source_count%25==0):
   self.write_stage('source_ranking',source_models_completed=self.source_count,source_rank_seconds=self.source_seconds)
  return RankResult(result['loss'])
 def write_stage(self,stage,**values):
  if not self.telemetry:return
  p=Path(self.telemetry);t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(at=time.time(),stage=stage,endpoint=self.endpoint,**values),indent=2));t.replace(p)
 def loo(self,x,y,count,model,seed):
  # get_gaussian_process consumes one seed per fold, even with no optimization.
  for _ in range(len(y)):model.rng.randint(0,2**20)
  self.write_stage('joint_LOOCV',observations=len(y),source_models_completed=self.source_count)
  started=time.monotonic();r=self.call('loo',x=x,y=y,theta=model.kernel.theta,normal=self.normal(len(y),count,seed));self.loo_seconds+=time.monotonic()-started
  return r['loss']
class LazySource:
 def __init__(self,task,load):self.task=task;self.load=load;self.model=None
 def __getattr__(self,name):
  if self.model is None:self.model=self.load()
  return getattr(self.model,name)
class RankResult:
 def __init__(self,loss):self.loss=loss

def install(client):
 import peptide_rgpe_exact as exact
 import peptide_transfer_models as models
 from rgpe.methods import rgpe as upstream
 original_bank=models.SourceBank
 class GPUSourceBank(original_bank):
  def __getitem__(self,index):
   if index==len(self.tasks) and self.target is not None:return self.target
   return LazySource(self.tasks[index],lambda:original_bank.__getitem__(self,index))
 models.SourceBank=GPUSourceBank
 # Preserve the existing CPU likelihood optimizer: all starts and gradients remain unchanged.
 models.sample_rank_posterior=client.source_draw
 # RGPEBackend imports this callable inside fit, so patch the imported module too.
 import peptide_rgpe_posterior as posterior
 posterior.sample_rank_posterior=client.source_draw
 models.target_ranking_loss=client.loo
 original=upstream.compute_ranking_loss
 def ranking(draws,y,target_model):
  if isinstance(draws,RankResult):
   if target_model:raise ValueError('Unexpected target ranking request')
   return draws.loss
  return original(draws,y,target_model)
 upstream.compute_ranking_loss=ranking
