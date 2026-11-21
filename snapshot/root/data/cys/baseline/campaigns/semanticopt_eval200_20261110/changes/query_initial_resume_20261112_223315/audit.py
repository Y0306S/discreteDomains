import pathlib,sys,json,sqlite3,os
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'code'))
from common import QUERY_SOURCE
sys.path.insert(0,str(QUERY_SOURCE/'code'))
import run_query_ceb_baselines as query
query.native_path(QUERY_SOURCE/'native')
from workload.workloads import get_workload_set
from codec.codec import AliasesCodec
from protocol import validate_query_plan
native=QUERY_SOURCE/'native';results=[]
for p in sorted((root/'query/runs').glob('*/observations.sqlite')):
 db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True)
 initial=db.execute("select value from state where key='initial'").fetchone()
 if not initial:continue
 old=json.loads(initial[0]);task=p.parent.name
 definition=get_workload_set('CEB_3K').extract_query(query.sql_path(native,task).read_text(),task)
 codec=AliasesCodec(definition.all_tables)
 raw=query.load_initializations(native/'initialization_data/ceb_3k_bao_initialization.csv',[task])[task]
 mismatches=[];bad=[]
 for i,(saved,row) in enumerate(zip(old,raw)):
  tree=codec.decode(definition.query_tables,row['plan'].copy())
  plan=codec.encode(tree)
  restored=codec.decode(definition.query_tables,saved['configuration']['plan'].copy())
  if saved['configuration']['plan']!=plan:mismatches.append(i)
  if not tree.equal(restored) or saved['objective']!=row['runtime'] or saved['censored']!=bool(row['censored']):bad.append(i)
  validate_query_plan(saved['configuration']['plan'],definition,codec,require_canonical=False)
 results.append({'task':task,'count':len(old),'raw_count':len(raw),'encoding_differences':mismatches,'semantic_or_measurement_changes':bad})
print(json.dumps({'hash_seed':os.environ.get('PYTHONHASHSEED'),'results':results}))
