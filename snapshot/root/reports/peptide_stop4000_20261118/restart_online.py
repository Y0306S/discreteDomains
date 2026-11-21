import hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/root/data/logs/peptide_apexgo_recipe_fsbo_test10_20261118')
OUT=Path(__file__).resolve().parent

def processes():
 rows={}
 for p in Path('/proc').glob('[0-9]*'):
  try:rows[int(p.name)]=((p/'cmdline').read_bytes().split(b'\0'),int((p/'stat').read_text().split()[3]))
  except (OSError,ValueError):pass
 return rows
pids=processes();controller=json.loads((ROOT/'pipeline_status.json').read_text())['pid']
assert str(ROOT/'code/pipeline.py').encode() in pids[controller][0]
workers=[pid for pid,(argv,ppid) in pids.items() if str(ROOT/'code/evaluate.py').encode() in argv]
assert len(workers)==4
snapshot=[]
for pid in workers:
 argv=pids[pid][0];task=argv[argv.index(b'--task')+1].decode();folder=ROOT/'evaluation/adapted_online'/task
 status=json.loads((folder/'status.json').read_text())
 snapshot.append(dict(pid=pid,task=task,progress=status,checkpoint_sha256=hashlib.sha256((folder/'checkpoint.pt').read_bytes()).hexdigest()))
(OUT/'online_before_restart.json').write_text(json.dumps(snapshot,indent=2)+'\n')
os.kill(controller,signal.SIGKILL)
for pid in workers:os.kill(pid,signal.SIGSTOP)
children=[pid for pid,(argv,ppid) in processes().items() if ppid in workers and str(ROOT/'code/source_meta.py').encode() in argv]
for pid in children:os.kill(pid,signal.SIGTERM)
for pid in workers:
 os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
for _ in range(40):
 alive=[]
 for pid in workers+children:
  p=Path(f'/proc/{pid}/stat')
  if p.exists() and p.read_text().split()[2]!='Z':alive.append(pid)
 if not alive:break
 time.sleep(.25)
assert not alive,alive
with (ROOT/'pipeline.log').open('ab',buffering=0) as log:
 process=subprocess.Popen(['python3','-u',str(ROOT/'code/pipeline.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
info=dict(pid=process.pid,started_at=time.time(),reason='Resume unchanged optimizer checkpoints under stop_policy.json cap4000',previous_controller=controller,stopped_workers=workers,stopped_source_training_children=children,code_manifest_sha256=hashlib.sha256((ROOT/'code_manifest.json').read_bytes()).hexdigest())
(ROOT/'pipeline_process.json').write_text(json.dumps(info,indent=2)+'\n')
(OUT/'online_restart.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info),flush=True)
