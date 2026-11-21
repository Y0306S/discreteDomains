"""FP64 CUDA evaluation of the existing Matern-5/2 GP mathematics."""
import time
import os,subprocess,pickle,struct
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch

class Math:
 def __init__(self):
  torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.cuda.set_per_process_memory_fraction(.10)
  self.data={};self.draws=None;self.sources=OrderedDict();self.loader=None
 def tensor(self,x):return torch.as_tensor(x,dtype=torch.float64,device='cuda')
 def source(self,task):
  if task in self.sources:
   self.sources.move_to_end(task);return self.sources[task]
  if self.loader is None:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
   self.loader=subprocess.Popen(['/root/data/cys/baseline/envs/bl_rgpe/bin/python',str(Path(__file__).with_name('source_loader.py')),'/root/data/logs/peptide_transfer_test200_20261028_v1/rgpe/recipe.json'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env)
  payload=pickle.dumps(dict(task=task),protocol=4);self.loader.stdin.write(struct.pack('!Q',len(payload))+payload);self.loader.stdin.flush()
  header=self.loader.stdout.read(8)
  if len(header)!=8:raise RuntimeError('Source loader exited')
  size=struct.unpack('!Q',header)[0];buffer=bytearray()
  while len(buffer)<size:
   chunk=self.loader.stdout.read(min(size-len(buffer),2**20))
   if not chunk:raise RuntimeError('Source loader truncated its reply')
   buffer.extend(chunk)
  reply=pickle.loads(buffer)
  if 'error' in reply:raise RuntimeError(reply['error'])
  d=reply['result']
  while len(self.sources)>=12:self.sources.popitem(last=False)
  state={k:self.tensor(v) if k in ('train','L','alpha','theta') else v for k,v in d.items()}
  self.sources[task]=state;return state
 def kernel(self,x,y,theta,diagonal=False):
  p=theta.exp();a,l,n=p[0],p[1:-1],p[-1]
  r=torch.cdist(x/l,y/l,compute_mode='donot_use_mm_for_euclid_dist')*np.sqrt(5.)
  k=a*(1+r+r.square()/3)*(-r).exp()
  if diagonal:k.diagonal().add_(n)
  return k
 def factor(self,c):
  mag=max(1.,float(c.diagonal().abs().max()))
  for jitter in (0.,1e-12,1e-10,1e-8):
   a=c.clone();a.diagonal().add_(jitter*mag);l,info=torch.linalg.cholesky_ex(a)
   if int(info)==0:return l,jitter*mag
  raise ValueError('Posterior not positive semidefinite within existing jitter policy')
 def posterior(self,d):
  x=self.tensor(d['x']);train=self.tensor(d['train']);theta=self.tensor(d['theta'])
  cross=self.kernel(x,train,theta);l=self.tensor(d['L']);alpha=self.tensor(d['alpha'])
  mean=cross@alpha;v=torch.cholesky_solve(cross.T,l)
  covariance=self.kernel(x,x,theta,True)-cross@v
  return mean*d.get('scale',1.)+d.get('offset',0.),covariance*d.get('scale',1.)**2
 def ranking(self,draws,y):
  # Same ordered-pair loss as the upstream circular-shift implementation.
  total=torch.zeros(draws.shape[0],dtype=torch.int64,device='cuda')
  for start in range(0,len(y),32):
   stop=min(start+32,len(y));a=draws[:,start:stop,None]<draws[:,None,:]
   b=y[start:stop,None]<y[None,:]
   total+=(a^b).sum(dim=(1,2))
  return total
 def run(self,d):
  op=d['op'];started=time.monotonic()
  if op=='health':return dict(device=torch.cuda.get_device_name(0),dtype='float64',peak_gib=torch.cuda.max_memory_allocated()/2**30)
  if op=='dataset':
   key=d['key'];self.data[key]=(self.tensor(d['x']),self.tensor(d['y']))
   while len(self.data)>64:self.data.pop(next(iter(self.data)))
   return True
  if op=='nll':
   if d['key'] not in self.data:return {'missing':True}
   x,y=self.data[d['key']];theta=self.tensor(d['theta']);p=theta.exp();amplitude,scales,noise=p[0],p[1:-1],p[-1]
   radial=torch.cdist(x/scales,x/scales,compute_mode='donot_use_mm_for_euclid_dist')*np.sqrt(5.)
   exponential=(-radial).exp();signal=(1+radial+radial.square()/3)*exponential
   covariance=amplitude*signal;covariance.diagonal().add_(noise)
   factor,info=torch.linalg.cholesky_ex(covariance)
   if int(info):return dict(value=-np.inf,gradient=np.zeros_like(d['theta']))
   identity=torch.eye(len(x),dtype=torch.float64,device='cuda');alpha=torch.cholesky_solve(y[:,None],factor)[:,0]
   value=-.5*y.dot(alpha)-factor.diagonal().log().sum()-len(x)/2*np.log(2*np.pi)
   sensitivity=alpha[:,None]*alpha[None,:]-torch.cholesky_solve(identity,factor)
   gradient=torch.empty_like(p);gradient[0]=.5*amplitude*(sensitivity*signal).sum();gradient[-1]=.5*noise*sensitivity.diagonal().sum()
   sensitivity*=(5./3)*amplitude*(1+radial)*exponential;sensitivity.fill_diagonal_(0)
   edges=sensitivity.sum(0)+sensitivity.sum(1)
   gradient[1:-1]=(.5*edges.matmul(x.square())-(x*sensitivity.matmul(x)).sum(0))/scales.square()
   return dict(value=value.item(),gradient=gradient.cpu().numpy(),seconds=time.monotonic()-started)
  if op=='source_rank_id':
   original=d;d=dict(self.source(d['source']),x=d['x'],normal=d['normal'],y=d['y']);op='source_rank'
  if op in ('posterior','source_rank'):
   mean,cov=self.posterior(d)
   if op=='posterior':return dict(mean=mean.cpu().numpy(),covariance=cov.cpu().numpy())
   factor,jitter=self.factor(cov);draws=mean+self.tensor(d['normal'])@factor.T
   loss=self.ranking(draws,self.tensor(d['y']))
   return dict(loss=loss.cpu().numpy(),jitter=jitter,seconds=time.monotonic()-started)
  if op=='loo':
   x=self.tensor(d['x']);y=self.tensor(d['y']);theta=self.tensor(d['theta']);normal=self.tensor(d['normal'])
   kernel=self.kernel(x,x,theta,True);signal=kernel.clone();signal.diagonal().sub_(theta[-1].exp())
   total=torch.zeros(normal.shape[0],dtype=torch.int64,device='cuda');indices=torch.arange(len(y),device='cuda')
   # Each fold retains the upstream default kernel and independently normalizes labels.
   for i in d.get('fold_indices',range(len(y))):
    keep=indices!=i;raw=y[keep];offset=raw.mean();scale=raw.std(unbiased=False)
    if float(scale)==0:scale=torch.ones_like(scale)
    folded=kernel[keep][:,keep];l=torch.linalg.cholesky(folded)
    cross=signal[:,keep];alpha=torch.cholesky_solve(((raw-offset)/scale)[:,None],l)[:,0]
    mean=(cross@alpha)*scale+offset
    cov=(kernel-cross@torch.cholesky_solve(cross.T,l))*scale.square()
    factor,jitter=self.factor(cov);draws=mean+normal@factor.T
    total+=((draws[:,i:i+1]<draws)^(y[i]<y)[None,:]).sum(1)
   return dict(loss=total.cpu().numpy(),seconds=time.monotonic()-started)
  raise ValueError('Unsupported CUDA operation '+op)
