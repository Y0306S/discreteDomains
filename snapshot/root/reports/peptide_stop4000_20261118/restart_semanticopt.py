import hashlib,json,os,signal,sqlite3,subprocess,time
from pathlib import Path
ROOT=Path('/root/data/cys/baseline/campaigns/semanticopt_peptide_test10_5000_20261117')
OUT=ROOT/'evidence/stop4000_20261118'
OUT.mkdir(parents=True,exist_ok=True)

def argv(pid):return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
def snapshot(task):
 db=sqlite3.connect(f'file:{ROOT}/peptide/runs/{task}/observations.sqlite?mode=ro',uri=True,timeout=1)
 try:
  db.execute('BEGIN')
  journal=db.execute("SELECT value FROM state WHERE key='oracle_batch'").fetchone()
  journal=json.loads(journal[0]) if journal else {}
  records=db.execute('SELECT n,value FROM records ORDER BY n').fetchall()
  return dict(count=len(records),records_sha256=hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest(),oracle_journal=journal)
 finally:db.close()
cfg=json.loads((ROOT/'processes.json').read_text());old=json.loads((ROOT/'controller_process.json').read_text())['pid']
assert str(ROOT/'control/controller.py').encode() in argv(old)
for task,pid in cfg['workers'].items():assert str(ROOT/'code/evaluate.py').encode() in argv(pid) and task.encode() in argv(pid)
for pid in cfg['replicas']+[cfg['broker']]:assert str(ROOT/'services/qwen_inference_server.py').encode() in argv(pid)
os.kill(old,signal.SIGKILL)
remaining=dict(cfg['workers']);before={}
while remaining:
 for task,pid in list(remaining.items()):
  os.kill(pid,signal.SIGSTOP)
  try:state=snapshot(task)
  except sqlite3.Error:
   os.kill(pid,signal.SIGCONT);continue
  if state['oracle_journal'].get('status')=='in_flight':
   os.kill(pid,signal.SIGCONT);continue
  assert state['count']<=4000
  before[task]=dict(pid=pid,**state)
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  del remaining[task]
  print('Stopped safely',task,state['count'],flush=True)
 if remaining:time.sleep(2)
for _ in range(40):
 alive=[pid for pid in cfg['workers'].values() if Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z']
 if not alive:break
 time.sleep(.25)
assert not alive,alive
for task,state in before.items():
 after=snapshot(task)
 assert (after['count'],after['records_sha256'])==(state['count'],state['records_sha256'])
(OUT/'semanticopt_before_restart.json').write_text(json.dumps(before,indent=2)+'\n')
with (ROOT/'logs/controller.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'control/controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
info=dict(pid=p.pid,host='LLMOptimizer3',command='python3 control/controller.py',started_at=time.time(),reason='Resume saved proposal journals under stop_policy.json cap4000; adopt existing inference services',previous_controller=old,retained_replicas=cfg['replicas'],retained_broker=cfg['broker'])
(ROOT/'controller_process.json').write_text(json.dumps(info,indent=2)+'\n')
(OUT/'semanticopt_restart.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info),flush=True)
