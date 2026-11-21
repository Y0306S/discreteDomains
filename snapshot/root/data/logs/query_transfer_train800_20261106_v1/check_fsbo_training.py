import json,os,sqlite3,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/data/cys/llmoptimizer_benchmark/tools')
import query_transfer as query
sys.path.insert(0,str(query.TRAINER.parent))
import train_taskset_transfer as trainer
db=sqlite3.connect('file:'+str(ROOT/'corpus/corpus.sqlite')+'?mode=ro',uri=True,timeout=120)
ids=[r[0] for r in db.execute('select id from tasks order by id limit 8')]
data={task:query.read_task(db,task) for task in ids};db.close()
features=np.concatenate([d['X'] for d in data.values()])
manifest=dict(fsbo_train_tasks=ids[:6],fsbo_validation_tasks=ids[6:],lower=features.min(0).tolist(),upper=features.max(0).tolist())
tasks={task:dict(X=query.shared.normalize(d['X'],manifest),y_maximize=-d['y']) for task,d in data.items()}
output=ROOT/'component_fsbo_100';output.mkdir(exist_ok=True)
args=SimpleNamespace(threads=1,seed=123,output=output,steps=100,resume=False,purpose='component_check')
trainer.identity=lambda args,manifest:dict(domain='ceb_query_runtime',purpose='component_check',production=False,source_tasks=ids)
trainer.fsbo(args,manifest,tasks)
status=json.loads((output/'status.json').read_text())
assert status['status']=='complete' and status['reload_verified']
print(json.dumps(dict(passed=True,source_tasks=8,input_dimensions=64,updates=100,reload_verified=True)))
