"""Read-only hourly training health monitor; no automatic restarts or chat delivery."""
import fcntl,json,os,subprocess,time,traceback,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CAMPAIGNS=[Path('/root/data/cys/baseline/campaigns')/x for x in ['query_our_orpo_rl800_20261112','query_warmstart_full800_20261112','semanticopt_eval200_20261110','query_our_fsbo_test200_20261116']]
def atomic(path,value):
 temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
def load(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return None
def alive(pid,start=None):
 try:
  stat=(Path('/proc')/str(pid)/'stat').read_text().rsplit(') ',1)[1].split()
  return stat[0]!='Z' and (start is None or stat[19]==str(start))
 except (OSError,ValueError):return False
def observe(campaign,previous):
 statuses={}
 for name in ['status.json','queue_status.json','query/progress.json','query/LLMOptimizer2.status.json','query/LLMOptimizer4.status.json','training_status.json','orpo_status.json','services/queue_status.json','services/pipeline_status.json','reports/build_progress.json']:
  value=load(campaign/name)
  if value is not None:statuses[name]=value
 processes={}
 for p in list((campaign/'services').glob('*_process.json'))+list((campaign/'services').glob('*.process.json'))+list(campaign.glob('queue_process.json')):
  value=load(p)
  if value and 'pid' in value:processes[p.name]=dict(pid=value['pid'],alive=alive(value['pid'],value.get('start_ticks')),command=value.get('command'))
 primary=statuses.get('services/queue_status.json',statuses.get('queue_status.json',statuses.get('status.json',{})));state=primary.get('status','unverified')
 queue=processes.get('queue_process.json')
 if state not in ['complete','completed','error','failed','interrupted'] and queue and not queue['alive']:state='crashed'
 errors={str(p.relative_to(campaign)):load(p) for p in campaign.glob('*error.json')}
 completed_rollout_groups=len(list((campaign/'rollouts').glob('*/*/*/r0/rollouts_*.jsonl'))) if (campaign/'rollouts').exists() else 0
 record=dict(state=state,statuses=statuses,processes=processes,errors=errors,completed_rollout_groups=completed_rollout_groups)
 if campaign.name=='semanticopt_eval200_20261110':
  try:
   with urllib.request.urlopen('http://127.0.0.1:19990/health',timeout=15) as response:health=json.load(response)
   record['inference_health']=dict(ready=health.get('ready'),replicas=len(health.get('replica_health',[])),queued=health.get('queued'),workers=[{k:r.get(k) for k in ['name','ready','busy','completed','error']} for r in health.get('replica_health',[])])
  except Exception as error:record['inference_health']=dict(ready=False,error=repr(error))
 if campaign.name=='query_our_fsbo_test200_20261116':
  controller=load(campaign/'services/batch_controller.process.json')
  if controller:
   record['controller_alive']=alive(controller['pid'],controller.get('start_ticks'))
   if state=='running' and not record['controller_alive']:record['state']='crashed'
  record['errors'].update({str(p.relative_to(campaign)):load(p) for stage in ['warm30','orpo_batch10'] for p in (campaign/stage/'runs').glob('*/error.json')})
  record['inference_health']={}
  active_stage=load(campaign/'services/active_inference_stage.json')
  for name,port in [('bolt',21810),('orpo',21811)]:
   if active_stage and name!=('bolt' if active_stage['stage']=='warm30' else 'orpo'):
    record['inference_health'][name]=dict(state='inactive_in_current_stage');continue
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=10) as response:h=json.load(response)
    record['inference_health'][name]=dict(ready=h.get('ready'),queued=h.get('queued'),replicas=[{k:r.get(k) for k in ['name','ready','busy','completed','error','prompt_cache']} for r in h.get('replica_health',[])])
   except Exception as error:record['inference_health'][name]=dict(ready=False,error=repr(error))
  resources=load(campaign/'resources.json')
  if resources:
   record['resource_health']=load(campaign/'services/resources_health.json')
   record['worker_statuses']={str(w['index']):load(campaign/'services'/f"batch_worker{w['index']}.status.json") for w in resources['workers']}
   remote_records={name:value for name,value in processes.items() if (load(campaign/'services'/name) or {}).get('host')=='LLMOptimizer4'}
   try:
    probe="import json\nfrom pathlib import Path\nr={}\nfor p in Path('/proc').glob('[0-9]*/stat'):\n try:r[p.parent.name]=p.read_text().rsplit(') ',1)[1].split()\n except OSError:pass\nprint(json.dumps(r))"
    remote_stats=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','LLMOptimizer4','python3 -c '+__import__('shlex').quote(probe)],text=True,timeout=30))
    for name,value in remote_records.items():
     row=remote_stats.get(str(value['pid']));saved=load(campaign/'services'/name)
     value['alive']=bool(row and row[0]!='Z' and (saved.get('start_ticks') is None or row[19]==str(saved['start_ticks'])))
     value['host']='LLMOptimizer4'
    record['remote_host_reachable']=True
   except Exception as error:
    record['remote_host_reachable']=False;record['remote_error']=repr(error)
    for value in remote_records.values():value['alive']=None;value['host']='LLMOptimizer4'
   active_records=[load(campaign/'services'/f"batch_worker{w['index']}.process.json") for w in resources['workers']]
   record['expected_evaluators']=len(resources['workers'])
   record['configured_gpu_services']=dict(llm=len(resources['llm']),vae=len(resources['vae']))
  current=primary.get('counted_measurements')
  old_count=(previous or {}).get('statuses',{}).get('status.json',{}).get('counted_measurements')
  record['measurement_delta']=None if current is None or old_count is None else current-old_count
  record['no_new_measurements_since_previous_check']=bool(state=='running' and record['measurement_delta']==0)
 if previous:
  record['previous_state']=previous.get('state');record['changed']=bool(record.get('measurement_delta')) or state!=previous.get('state') or statuses.get('orpo_status.json',{}).get('step')!=previous.get('statuses',{}).get('orpo_status.json',{}).get('step') or statuses.get('training_status.json',{}).get('step')!=previous.get('statuses',{}).get('training_status.json',{}).get('step') or completed_rollout_groups!=previous.get('completed_rollout_groups')
 return record
def main():
 lock=(ROOT/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 previous=load(ROOT/'latest.json') or {};start=time.monotonic();tick=0
 while True:
  at=time.time();result=dict(at=at,campaigns={})
  try:
   for c in CAMPAIGNS:result['campaigns'][c.name]=observe(c,previous.get('campaigns',{}).get(c.name))
   result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True,timeout=30).strip()
   fs=os.statvfs(CAMPAIGNS[0]);result['disk_free_gib']=fs.f_bavail*fs.f_frsize/2**30;result['check']='completed'
  except Exception as e:result.update(check='failed',error=repr(e))
  atomic(ROOT/'latest.json',result)
  with (ROOT/'history.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  tick=max(tick+1,int((time.monotonic()-start)//3600)+1);delay=max(0,start+tick*3600-time.monotonic())
  atomic(ROOT/'state.json',dict(loop_id=ROOT.name,job='Read-only query training, SemanticOPT and BOLT→ORPO+FSBO health/progress/crash checks',targets=list(map(str,CAMPAIGNS)),interval_seconds=3600,timezone='Asia/Shanghai',mechanism='persistent local Python process with exclusive flock',schedule_id=f'pid:{os.getpid()}',pid=os.getpid(),start_ticks=(Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(') ',1)[1].split()[19],state='active',last_start=at,last_finish=time.time(),last_result=result['check'],next_due=time.time()+delay,chat_notifications=False,automatic_restarts=False))
  previous=result;time.sleep(delay)
if __name__=='__main__':main()
