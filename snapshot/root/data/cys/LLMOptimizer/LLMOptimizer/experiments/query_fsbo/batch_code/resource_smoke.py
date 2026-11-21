"""Cross-host locking, native VAE equivalence, train-only remote FSBO smoke."""
import argparse,fcntl,json,os,shlex,subprocess,time,urllib.request
from pathlib import Path
import numpy as np
from query_protocol import atomic,digest
ROOT=Path(__file__).resolve().parents[1]
PYTHON='/root/data/cys/envs/bolt_clean_venv/bin/python'
def remote(*args):
 cmd=[PYTHON,str(Path(__file__).resolve()),*args]
 return subprocess.run(['ssh','-o','BatchMode=yes','LLMOptimizer4',shlex.join(['env','PYTHONHASHSEED=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1']+cmd)],capture_output=True,text=True,check=True,timeout=180)
def probe(expect):
 with (ROOT/'evidence/resource_flock_probe.lock').open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locked=True
  except BlockingIOError:locked=False
 assert locked==(expect=='acquire'),(locked,expect)
 print(json.dumps(dict(shared_flock=expect,passed=True)))
def train_rows():
 from single_method import Journal
 from warm import warm_start
 # Read only an already committed train-only preflight state; no SQL is issued.
 import sqlite3,pickle
 with sqlite3.connect(f"file:{ROOT/'evidence/batch_preflight.sqlite'}?mode=ro",uri=True) as db:
  print('journal schema',db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'").fetchall(),flush=True) if False else None
  rows=db.execute('SELECT value FROM kv WHERE key=?',('warm_state',)).fetchone()
  if rows is None:raise ValueError('Missing train-only warm state')
  return pickle.loads(rows[0])['rows']
def smoke():
 from query_data import canonicalize
 task='CEB_10A19';cfg=json.loads((ROOT/'batch_recipe.json').read_text());assert task in cfg['train_tasks'] and task not in cfg['tasks']
 rows=train_rows();plan=canonicalize(task,rows[0]['plan'],strict=True)
 token=(ROOT/'services/token').read_text().strip();encodings=[];reports=[]
 for w in json.loads((ROOT/'resources.json').read_text())['vae']:
  endpoint=f"http://127.0.0.1:{w['port']}"
  def call(body):
   req=urllib.request.Request(endpoint+'/inference',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
   with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
  z=call(dict(op='encode_mean',plans=[plan],task=task,seed=123))['z'];encodings.append(z)
  decoded=call(dict(op='decode',z=z,task=task,fallback=plan,seed=123))['rows']
  assert len(decoded)==1
  executed=canonicalize(task,canonicalize(task,decoded[0]['plan']),strict=True)
  reports.append(dict(port=w['port'],dimension=len(z[0]),strict_executed_plan_valid=True,used_decoder_fallback=decoded[0]['fallback']))
 assert all(np.allclose(encodings[0],z,atol=1e-6,rtol=1e-5) for z in encodings)
 # Compare actual GPU service outputs with the GPU-exported training corpus.
 import io,sqlite3
 manifest=json.loads((ROOT/'training/corpus/manifest.json').read_text())
 source_task=manifest['source_tasks'][0]
 data=json.loads((ROOT/'training/plans.json').read_text())['tasks'][source_task]
 with sqlite3.connect('file:'+str(ROOT/'training/corpus/corpus.sqlite')+'?mode=ro',uri=True) as db:
  payload=db.execute('SELECT payload FROM tasks WHERE id=?',(source_task,)).fetchone()[0]
 with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:expected=arrays['X'][:8].copy()
 actual=call(dict(op='encode_mean',plans=data['plans'][:8],task=source_task,seed=123))['z']
 assert np.allclose(expected,np.asarray(actual),atol=1e-6,rtol=1e-5),'FSBO corpus/serving embeddings differ'
 original=Path('/root/data/logs/query_transfer_train800_20261106_v1/corpus')
 om=json.loads((original/'manifest.json').read_text());ot=om['source_tasks'][0]
 with sqlite3.connect('file:'+str(original/'corpus.sqlite')+'?mode=ro',uri=True) as db:
  payload=db.execute('SELECT payload FROM tasks WHERE id=?',(ot,)).fetchone()[0]
 with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:indices=arrays['source_rows'][:8].copy();expected_original=arrays['X'][:8].copy()
 h=json.loads((Path('/root/data/cys/logs/query_train800_author_fixed_20261004/outputs')/ot/'lolbo/seed0/history.json').read_text())
 op=[canonicalize(ot,h['observations'][int(i)]['extra_info']['plan']) for i in indices]
 actual_original=call(dict(op='encode_retrieval_mean',plans=op,task=ot,seed=123))['z']
 assert np.allclose(expected_original,np.asarray(actual_original),atol=1e-6,rtol=1e-5),'Frozen retrieval GP/serving embeddings differ'
 from backends import FSBO
 fsbo=FSBO(cfg,task,rows)
 try:result=fsbo(rows,seed=123,count=10)
 finally:fsbo.close()
 assert result['plans'],'Remote FSBO should produce a valid native proposal'
 for p in result['plans']:canonicalize(task,p,strict=True)
 report=dict(passed=True,host='LLMOptimizer4',heldout_SQL_calls=0,new_SQL_calls=0,training_task=task,vae=reports,encodings_equivalent=True,gpu_training_and_service_embeddings_equivalent=True,retrieval_embeddings_match_original_corpus=True,fsbo_accepted=len(result['plans']),fsbo_fit=result['diagnostics']['fit'],at=time.time())
 atomic(ROOT/'evidence/resources_remote_smoke.json',report);print(json.dumps(report),flush=True)
def main():
 with (ROOT/'evidence/resource_flock_probe.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);remote('probe','blocked')
 remote('probe','acquire');print('Cross-host lock exclusion and release passed',flush=True)
 result=remote('smoke');print(result.stdout,flush=True)
if __name__=='__main__':
 import sys
 if len(sys.argv)>1 and sys.argv[1]=='probe':probe(sys.argv[2])
 elif len(sys.argv)>1 and sys.argv[1]=='smoke':smoke()
 else:main()
