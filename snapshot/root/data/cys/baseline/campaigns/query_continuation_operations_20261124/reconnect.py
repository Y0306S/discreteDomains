import os,signal,subprocess,time
import fleet
records=[]
for kind,root in fleet.ROOTS.items():
 record=fleet.load(root/'control/isolated_controller.process.json')
 if fleet.process_ticks(record['pid']) is not None:
  assert fleet.process_ticks(record['pid'])==record['start_ticks']
  os.kill(record['pid'],signal.SIGTERM)
 for i in range(100):
  if fleet.process_ticks(record['pid']) is None:break
  time.sleep(.05)
 else:raise RuntimeError('Controller did not stop')
 workers=fleet.load(root/'control/workers.process.json')
 for w in workers['workers']:w['start_ticks']=fleet.process_ticks(w['pid'])
 fleet.atomic(root/'control/workers.process.json',workers)
 with (root/'logs/controller.isolated.log').open('ab',buffering=0) as log:
  child=subprocess.Popen([fleet.PYTHON,'-u',str(fleet.OPS/'fleet.py'),'controller',kind],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 new=dict(pid=child.pid,start_ticks=fleet.process_ticks(child.pid),kind=kind,host='LLMOptimizer2',supervisor=str(fleet.OPS/'fleet.py'))
 fleet.atomic(root/'control/isolated_controller.process.json',new)
 records.append(dict(kind=kind,previous_controller=record,new_controller=new,adopted_workers=[w['slot'] for w in workers['workers'] if w['start_ticks']]))
fleet.atomic(fleet.OPS/'reconnect.json',dict(records=records))
print(records)
