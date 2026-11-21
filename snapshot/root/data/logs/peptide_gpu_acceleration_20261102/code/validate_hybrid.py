import hashlib,json,os,pickle,sqlite3,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent;TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path[:0]=[str(TOOLS),'/root/reports/peptide_rgpe_sobol_recovery_20261102/code','/root/data/cys/baseline/repos/transfer-hpo-framework'];os.sched_setaffinity(0,{72})
from sobol_repair import install as install_sobol
install_sobol()
from rgpe_cuda_client import Client,install
from peptide_rgpe_posterior import sample_rank_posterior,full_posterior
from peptide_transfer_models import target_ranking_loss,RGPEBackend
from rgpe.methods import rgpe as upstream
from rgpe.utils import get_gaussian_process
from ConfigSpace import ConfigurationSpace,UniformFloatHyperparameter
from smac.epm.util_funcs import get_types
client=Client('http://127.0.0.1:18941',OUT/'service_token')
data=np.load(str(OUT/'validation_data.npz'));x=data['x'].astype(np.float64);raw=-data['score'];y=(raw-raw.mean())/raw.std();report=dict(at=time.time(),checks=[],implementation='CPU target optimizer + exact FP64 CUDA posterior/ranking',rejected_full_gpu_optimizer='validation.json')
recipe=json.loads(Path('/root/data/logs/peptide_frozen_local_recenter_test200_20261101_v1/rgpe/recipe.json').read_text())
def record(name,ok,**details):
 d=dict(name=name,passed=bool(ok),**details);report['checks'].append(d);print(json.dumps(d),flush=True)
def make():
 cs=ConfigurationSpace();cs.add_hyperparameters([UniformFloatHyperparameter('z%03d'%i,0.,1.) for i in range(256)]);types,bounds=get_types(cs,None)
 return get_gaussian_process(cs,types,bounds,np.random.RandomState(81),None)
with sqlite3.connect('file:'+recipe['rgpe_bank']+'?mode=ro',uri=True) as c:
 for task in recipe['rgpe_sources'][:3]:
  source=pickle.loads(c.execute('select payload from models where task=?',(task,)).fetchone()[0]);test=x[:1150];labels=y[:1150]
  t=time.monotonic();draws=sample_rank_posterior(source,test,256,19);loss=upstream.compute_ranking_loss(draws,labels,False);ct=time.monotonic()-t
  normal=client.normal(len(test),256,19);t=time.monotonic();gpu=client.call('source_rank_id',source=task,x=test,normal=normal,y=labels);gt=time.monotonic()-t
  t=time.monotonic();cached=client.call('source_rank_id',source=task,x=test,normal=normal,y=labels);cachedt=time.monotonic()-t
  cm,cc=full_posterior(source,test);posterior=client.call('posterior',**client.payload(source,test))
  posterior_ok=np.allclose(cm,posterior['mean'],rtol=1e-7,atol=1e-6) and np.allclose(cc,posterior['covariance'],rtol=1e-6,atol=1e-6)
  loss_error=int(np.max(np.abs(loss-gpu['loss'])));pair_count=len(test)*(len(test)-1)
  record('real_source_'+task,posterior_ok and loss_error<=max(2,int(pair_count*1e-5)) and np.array_equal(gpu['loss'],cached['loss']),posterior_passed=bool(posterior_ok),loss_max_error=loss_error,ordered_pair_count=pair_count,maximum_allowed_pair_fraction=1e-5,posterior_mean_error=float(np.max(np.abs(cm-posterior['mean']))),posterior_covariance_error=float(np.max(np.abs(cc-posterior['covariance']))),cpu_seconds=ct,gpu_cold_seconds=gt,gpu_cached_seconds=cachedt)
  del source
m=make();normal=client.normal(len(y),256,29);indices=[0,len(y)//2,len(y)-1];loss=np.zeros(256,dtype=np.int64);t=time.monotonic()
for i in indices:
 keep=np.arange(len(y))!=i;fold=get_gaussian_process(m.configspace,m.types,m.bounds,m.rng,m.kernel);fold._train(x[keep],y[keep],do_optimize=False);draws=sample_rank_posterior(fold,x,256,29);loss+=np.sum((draws[:,i:i+1]<draws)^(y[i]<y)[None,:],axis=1)
ct=time.monotonic()-t;t=time.monotonic();gpu=client.call('loo',x=x,y=y,theta=m.kernel.theta,normal=normal,fold_indices=indices);gt=time.monotonic()-t
record('large_joint_LOOCV_folds',np.array_equal(loss,gpu['loss']),observations=len(y),fold_indices=indices,cpu_seconds=ct,gpu_seconds=gt,loss_max_error=int(np.max(np.abs(loss-gpu['loss']))))
# Complete a real three-source target fit, full LOOCV, dilution and final weights.
small=dict(recipe);small['rgpe_sources']=small['rgpe_sources'][:3]
a=RGPEBackend(small);t=time.monotonic();cpu=a.fit(x[:32],data['score'][:32],91);ct=time.monotonic()-t
import peptide_rgpe_exact as exact
cpu_objective=exact.marginal_likelihood;install(client);client.y=(raw[:32]-raw[:32].mean())/raw[:32].std()
b=RGPEBackend(small);t=time.monotonic();gpu=b.fit(x[:32],data['score'][:32],91);gt=time.monotonic()-t
record('target_optimizer_unchanged',exact.marginal_likelihood is cpu_objective and np.array_equal(a.bank.target.hypers,b.bank.target.hypers),optimizer_restarts=a.bank.target.n_opt_restarts,theta_max_error=float(np.max(np.abs(a.bank.target.hypers-b.bank.target.hypers))))
record('complete_weight_pipeline',np.array_equal(cpu['weights'],gpu['weights']),cpu_seconds=ct,gpu_seconds=gt,cpu_weights=cpu['weights'],gpu_weights=gpu['weights'])
from peptide_rgpe_merged import sample_ensemble
saved_weights_a=a.ensemble.weights_.copy();saved_weights_b=b.ensemble.weights_.copy()
a.ensemble.weights_=np.asarray([.5,0.,0.,.5]);b.ensemble.weights_=a.ensemble.weights_.copy()
sa=sample_ensemble(a.ensemble,x[:64],50,77);sb=sample_ensemble(b.ensemble,x[:64],50,77)
record('joint_batch_50_and_lazy_source',np.array_equal(sa['draws'],sb['draws']) and sb['draws'].shape==(50,64),shape=list(sb['draws'].shape),maximum_draw_error=float(np.max(np.abs(sa['draws']-sb['draws']))))
b.ensemble.weights_=saved_weights_b;payload=b.snapshot();b.restore(payload)
record('target_snapshot_restore',np.array_equal(b.ensemble.weights_,saved_weights_b) and np.array_equal(a.bank.target.hypers,b.bank.target.hypers),snapshot_bytes=len(payload))
report['passed']=all(v['passed'] for v in report['checks']);report['code']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')};report['completed_at']=time.time();(OUT/'hybrid_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'])),flush=True);raise SystemExit(0 if report['passed'] else 1)
