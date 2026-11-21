import json,time
from pathlib import Path
from warm import legacy
from query_protocol import atomic,digest
from query_data import task_description,build_prompt
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'batch_recipe.json').read_text());report={}
for task in cfg['tasks']:
 if not (ROOT/'warm_start/runs'/task/'runtime.sqlite').exists():continue
 data=legacy(task);report[task]=dict(reuse=len(data['rows']),archive=len(data['archive']),generation_calls=data['attempts'],rejected=data['rejected'],legacy_sha256=data['legacy_sha256'])
 assert not data['rejected'],(task,data['rejected'])
 # Exact training SFT constraints are already explicit, no prompt modification.
 messages,_=build_prompt(task,[],sft=True);ctx=json.loads(messages[1]['content']);desc=ctx['current_task']
 assert desc['constraints']['no_implicit_codec_repairs'] and desc['constraints']['preserve_SQL_and_predicates'] and desc['query_text'].strip()
 assert set(desc['constraints']['operators'])=={'1','2','3'}
atomic(ROOT/'evidence/batch_reuse_audit.json',dict(passed=True,tasks=report,reuse=sum(x['reuse'] for x in report.values()),archive=sum(x['archive'] for x in report.values()),rejected=sum(len(x['rejected']) for x in report.values()),constraints_explicit=True,at=time.time()))
print(json.dumps(report,indent=2));print('Reuse',sum(x['reuse'] for x in report.values()),'archive',sum(x['archive'] for x in report.values()))
