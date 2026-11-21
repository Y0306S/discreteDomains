"""Import already audited legacy rows without inference or SQL calls."""
import json,time
from pathlib import Path
from run import ROOT,recipe,identity
from warm import legacy,warm_progress
from single_method import Journal,best
from query_protocol import atomic
cfg=recipe();summary={}
for task in cfg['tasks']:
 if not (ROOT/'warm_start/runs'/task/'runtime.sqlite').exists():continue
 folder=ROOT/'warm30/runs'/task;folder.mkdir(parents=True,exist_ok=True);j=Journal(folder/'runtime.sqlite',identity('warm30',task))
 try:
  state=j.get('warm_state')
  if state is None:
   old=legacy(task)
   if old['rejected']:raise ValueError('Unexpected invalid legacy plan')
   state=dict(rows=old['rows'],attempts=old['attempts'],rejected=old['rejected'],legacy_calls=old['attempts'],legacy_sha256=old['legacy_sha256'])
   atomic(folder/'legacy_archive.json',dict(task=task,excluded_from_comparison=old['archive'],reason='measurements_after_first30_valid_generation_order'));j.put('warm_state',state)
  warm_progress(j,state)
  if len(state['rows'])==30:
   if best(state['rows']) is None:raise ValueError('No exact warm incumbent')
   result=dict(identity('warm30',task),attempts=30,generation_attempts=state['attempts'],rows=state['rows'],rejected=state['rejected'],status='complete',best_runtime=best(state['rows']),at=time.time());p=folder/'complete.json'
   if not p.exists():atomic(p,result)
  summary[task]=len(state['rows'])
 finally:j.close()
atomic(ROOT/'evidence/batch_reuse_import.json',dict(passed=True,measurements=sum(summary.values()),tasks=summary,SQL_calls=0,inference_calls=0,at=time.time()));print(json.dumps(summary,indent=2))
