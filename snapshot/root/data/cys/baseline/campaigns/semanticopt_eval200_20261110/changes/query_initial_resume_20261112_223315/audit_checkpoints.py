import pathlib,sys,json,sqlite3,hashlib
root=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'code'))
from common import QUERY_SOURCE
from evaluate import query_initializations
from protocol import validate_query_plan
sys.path.insert(0,str(QUERY_SOURCE/'code'))
import run_query_ceb_baselines as query
query.native_path(QUERY_SOURCE/'native')
from workload.workloads import get_workload_set
from codec.codec import AliasesCodec
results=[]
for p in sorted((root/'query/runs').glob('*/observations.sqlite')):
 db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 saved=dict(db.execute('select key,value from state'));initial=json.loads(saved.get('initial','null'))
 if not initial:continue
 task=p.parent.name;native=QUERY_SOURCE/'native'
 definition=get_workload_set('CEB_3K').extract_query(query.sql_path(native,task).read_text(),task);codec=AliasesCodec(definition.all_tables)
 raw=query.load_initializations(native/'initialization_data/ceb_3k_bao_initialization.csv',[task])[task]
 assert query_initializations(raw,definition,codec,initial)==initial
 plans=[json.loads(x[0])['configuration']['plan'] for x in db.execute('select value from records')]
 pending=json.loads(saved.get('pending','null'))
 assert not pending or pending['status']!='in_flight',task
 if pending:plans.append(pending['answer']['next_configuration']['plan'])
 for key in ['proposal_batch']+[k for k in saved if k.startswith('batch_build_')]:
  batch=json.loads(saved.get(key,'null'))
  if batch:plans.extend(x['answer']['next_configuration']['plan'] for x in batch['items'])
 for plan in plans:validate_query_plan(plan,definition,codec)
 results.append({'task':task,'bao':len(initial),'validated_saved_plans':len(plans)})
print(json.dumps({'passed':True,'results':results}))
