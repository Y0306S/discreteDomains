"""Isolated diagnostic: original/retrained features, identical80 source tasks and measured labels."""
import argparse,hashlib,json,os,sqlite3,sys,time
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
NEW=Path('/root/data/cys/baseline/campaigns/query_our_fsbo_vae80_test200_20261117')
OLD=Path('/root/data/logs/query_transfer_train800_20261106_v1')
TOOLS=Path('/root/data/cys/llmoptimizer_benchmark/tools')
sys.path[:0]=[str(TOOLS),'/root/data/cys/llmoptimizer_benchmark/examples/benchmark/domains']
import query_transfer as qt,peptide_transfer as shared

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def prepare():
 cfg=json.loads((NEW/'training/config.json').read_text());plans=json.loads((NEW/'training/plans.json').read_text())
 manifests={k:json.loads((root/'corpus/manifest.json').read_text()) for k,root in [('original',OLD),('retrained',NEW/'training')]}
 dbs={k:sqlite3.connect('file:'+str(root/'corpus/corpus.sqlite')+'?mode=ro',uri=True) for k,root in [('original',OLD),('retrained',NEW/'training')]}
 data={k:{} for k in dbs};checks=[]
 for task in cfg['source_tasks']:
  expected=plans['tasks'][task]
  for name,db in dbs.items():
   saved=qt.read_task(db,task);assert saved['payload_sha256']==manifests[name]['task_hashes'][task]
   assert saved['source_sha256']==expected['source_sha256']
   index={int(v):i for i,v in enumerate(saved['source_rows'])}
   order=[index[int(v)] for v in expected['source_rows']]
   np.testing.assert_array_equal(saved['y'][order],expected['y'])
   data[name][task]={'X':saved['X'][order],'y':np.asarray(expected['y'],dtype=np.float64)}
  checks.append({'task':task,'plans':len(expected['y']),'source_sha256':expected['source_sha256']})
 for name,tasks in data.items():
  folder=OUT/name/'corpus';folder.mkdir(parents=True,exist_ok=True)
  xs=np.concatenate([tasks[t]['X'] for t in cfg['source_tasks']]);lower=xs.min(0);upper=xs.max(0)
  assert np.all(upper>lower)
  for task,r in tasks.items():np.savez(folder/(task+'.npz'),X=((r['X']-lower)/(upper-lower)).astype(np.float32),y=r['y'])
  manifest={'source_tasks':cfg['source_tasks'],'fsbo_train_tasks':cfg['fsbo_train_tasks'],'fsbo_validation_tasks':cfg['fsbo_validation_tasks'],'representation':name,'lower':lower.tolist(),'upper':upper.tolist(),'dimension':64,'source_manifest_sha256':sha((OLD if name=='original' else NEW/'training')/'corpus/manifest.json'),'training_config_sha256':sha(NEW/'training/config.json'),'normalization':'same80_coordinate_minmax_including_training_side_validation_as_current_production','seed':123,'steps':100000,'label_checks':checks,'oracle_calls':0,'purpose':'isolated_diagnostic_no_promotion'}
  write(folder/'manifest.json',manifest)
  if name=='retrained':
   np.testing.assert_allclose(lower,manifests[name]['lower'],rtol=0,atol=0);np.testing.assert_allclose(upper,manifests[name]['upper'],rtol=0,atol=0)
 write(OUT/'matched80_preparation.json',{'passed':True,'tasks':80,'training_tasks':64,'validation_tasks':16,'plans':sum(x['plans'] for x in checks),'same_source_rows_and_labels':True,'retrained_bounds_identical_to_production':True,'oracle_calls':0})
 print('PREPARED',flush=True)
def train(name):
 import argparse
 import train_taskset_transfer as trainer
 root=OUT/name;folder=root/'corpus';m=json.loads((folder/'manifest.json').read_text());tasks={}
 for t in m['source_tasks']:
  with np.load(folder/(t+'.npz')) as a:tasks[t]={'X':a['X'].copy(),'y_maximize':-a['y'].copy(),'source_sha256':next(c['source_sha256'] for c in m['label_checks'] if c['task']==t)}
 def identity(args,manifest):return {'domain':'query_matched80_diagnostic','representation':name,'corpus_sha256':sha(folder/'manifest.json'),'seed':123,'device':'cpu','threads':2,'trainer_sha256':sha(trainer.__file__),'diagnostic_sha256':sha(__file__),'production_promoted':False}
 trainer.identity=identity
 output=root/'fsbo';output.mkdir(exist_ok=True)
 args=argparse.Namespace(corpus=folder,output=output,seed=123,threads=2,steps=100000,purpose='component_check',method='fsbo',resume=True)
 trainer.fsbo(args,m,tasks)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','original','retrained']);a=p.parse_args()
 os.sched_setaffinity(0,{34,35})
 prepare() if a.action=='prepare' else train(a.action)
