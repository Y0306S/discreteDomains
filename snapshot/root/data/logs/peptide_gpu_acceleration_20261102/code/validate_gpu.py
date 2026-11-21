"""Actual saved-coordinate, source-GP and target-fit equivalence checks."""
import hashlib,json,os,pickle,sqlite3,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path[:0]=[str(TOOLS),'/root/reports/peptide_rgpe_sobol_recovery_20261102/code','/root/data/cys/baseline/repos/transfer-hpo-framework']
os.sched_setaffinity(0,{72})
from sobol_repair import install as install_sobol
install_sobol()
from rgpe_cuda_client import Client
from peptide_rgpe_exact import marginal_likelihood,exact_objective
from peptide_rgpe_posterior import full_posterior,sample_rank_posterior
from peptide_transfer_models import target_ranking_loss
from rgpe.methods.rgpe import compute_ranking_loss
from ConfigSpace import ConfigurationSpace,UniformFloatHyperparameter
from smac.epm.util_funcs import get_types
from rgpe.utils import get_gaussian_process
client=Client('http://127.0.0.1:18941',OUT/'service_token')
data=np.load(str(OUT/'validation_data.npz'));x=data['x'].astype(np.float64);raw=-data['score'];y=(raw-raw.mean())/raw.std();report={'checks':[],'timings':[],'at':time.time()}
def record(name,passed,**details):
 d=dict(name=name,passed=bool(passed),**details);report['checks'].append(d);print(json.dumps(d),flush=True)
def make(seed=81):
 cs=ConfigurationSpace();cs.add_hyperparameters([UniformFloatHyperparameter('z%03d'%i,0.,1.) for i in range(256)]);types,bounds=get_types(cs,None)
 return get_gaussian_process(cs,types,bounds,np.random.RandomState(seed),None)
model=make();model._train(x,y,do_optimize=False)
for index,theta in enumerate([model.gp.kernel_.theta,model.gp.kernel_.theta+np.random.RandomState(9).uniform(-.03,.03,258)]):
 t=time.monotonic();cpu=marginal_likelihood(model.gp,theta);ct=time.monotonic()-t
 t=time.monotonic();gpu=client.marginal(model.gp,theta);gt=time.monotonic()-t
 ok=np.allclose(cpu[0],gpu[0],rtol=1e-8,atol=1e-6) and np.allclose(cpu[1],gpu[1],rtol=1e-5,atol=1e-5)
 record('target_marginal_'+str(index),ok,n=len(x),value_error=float(abs(cpu[0]-gpu[0])),gradient_error=float(np.max(np.abs(cpu[1]-gpu[1]))),cpu_seconds=ct,gpu_seconds=gt)
report['timings'].append(dict(stage='target_marginal',cpu=ct,gpu=gt))
recipe=json.loads(Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe/recipe.json').read_text())
with sqlite3.connect('file:'+recipe['rgpe_bank']+'?mode=ro',uri=True) as c:source=pickle.loads(c.execute('select payload from models where task=?',(recipe['rgpe_sources'][0],)).fetchone()[0])
for n in [64,min(1150,len(x))]:
 test=x[:n];normal=client.normal(n,256,19)
 t=time.monotonic();cm,cc=full_posterior(source,test);cd=sample_rank_posterior(source,test,256,19);cl=compute_ranking_loss(cd,y[:n],False);ct=time.monotonic()-t
 t=time.monotonic();gp=client.call('posterior',**client.payload(source,test));gl=client.call('source_rank',**dict(client.payload(source,test),normal=normal,y=y[:n]));gt=time.monotonic()-t
 record('source_posterior_'+str(n),np.allclose(cm,gp['mean'],rtol=1e-7,atol=1e-6) and np.allclose(cc,gp['covariance'],rtol=1e-6,atol=1e-6),mean_max_error=float(np.max(np.abs(cm-gp['mean']))),covariance_max_error=float(np.max(np.abs(cc-gp['covariance']))))
 record('source_ranking_'+str(n),np.array_equal(cl,gl['loss']),loss_max_error=float(np.max(np.abs(cl-gl['loss']))),cpu_seconds=ct,gpu_seconds=gt)
 report['timings'].append(dict(stage='source_rank_'+str(n),cpu=ct,gpu=gt))
for n in [24,64]:
 test=x[:n];labels=y[:n];m=make();m._train(test,labels,do_optimize=False)
 t=time.monotonic();cpu=target_ranking_loss(test,labels,256,m,29);ct=time.monotonic()-t
 t=time.monotonic();gpu=client.loo(test,labels,256,m,29);gt=time.monotonic()-t
 record('joint_LOOCV_'+str(n),np.array_equal(cpu,gpu),loss_max_error=int(np.max(np.abs(cpu-gpu))),cpu_seconds=ct,gpu_seconds=gt)
# Exercise all eleven optimizer starts on actual observations with both objectives.
a=make(91);b=make(91)
t=time.monotonic()
with exact_objective(a):a.train(x[:24],raw[:24])
ct=time.monotonic()-t
import peptide_rgpe_exact as exact
original=exact.marginal_likelihood;exact.marginal_likelihood=client.marginal
t=time.monotonic()
try:
 with exact_objective(b):b.train(x[:24],raw[:24])
finally:exact.marginal_likelihood=original
gt=time.monotonic()-t
am,ac=full_posterior(a,x[:40]);bm,bc=full_posterior(b,x[:40])
record('full_eleven_start_optimizer',np.allclose(am,bm,rtol=1e-5,atol=1e-4) and np.allclose(ac,bc,rtol=1e-5,atol=1e-4),cpu_seconds=ct,gpu_seconds=gt,mean_max_error=float(np.max(np.abs(am-bm))),covariance_max_error=float(np.max(np.abs(ac-bc))),cpu_nll_evaluations=a._n_ll_evals,gpu_nll_evaluations=b._n_ll_evals)
report['passed']=all(v['passed'] for v in report['checks']);report['code']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')};report['completed_at']=time.time();(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'])),flush=True)
raise SystemExit(0 if report['passed'] else 1)
