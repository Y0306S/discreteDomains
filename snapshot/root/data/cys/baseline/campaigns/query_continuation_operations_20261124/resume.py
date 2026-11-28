import fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import fleet

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

# Validate the original file pins; the supervisors are deliberately outside these files.
sem=fleet.ROOTS['semanticopt'];bolt=fleet.ROOTS['bolt']
checks=[]
for kind,mapping in [('semanticopt',fleet.load(sem/'identity.json')['files']),('bolt',fleet.load(bolt/'recipe.json')['pinned_files'])]:
 for p,want in mapping.items():
  actual=digest(p)
  if actual!=want:raise RuntimeError(f'{kind} pin mismatch: {p}')
 checks.append(dict(kind=kind,verified_files=len(mapping)))
for host in ['LLMOptimizer2','LLMOptimizer3','LLMOptimizer4']:
 results=fleet.load(fleet.OPS/f'preflight_{host}.json')['results']
 assert all(r['ok'] for r in results)
 assert len(results)=={'LLMOptimizer2':8,'LLMOptimizer3':5,'LLMOptimizer4':15}[host]
# Hold controller locks during archive and removal of the old automatic stop flags.
locks=[fleet.lock(r/'control/controller.lock') for r in fleet.ROOTS.values()]
archive=fleet.OPS/'before_resume'
archived=[]
for kind,root in fleet.ROOTS.items():
 candidates=[root/'PAUSE',root/'control/error.json',root/'status.json',root/'control/workers.process.json']
 candidates+=list((root/'query').glob('worker*.error.json')) if kind=='semanticopt' else list((root/'runs').glob('*/error.json'))
 candidates+=list((root/('query' if kind=='semanticopt' else 'services')).glob('worker*.status.json'))
 for path in candidates:
  if not path.exists():continue
  target=archive/kind/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():raise RuntimeError('Archive exists: '+str(target))
  shutil.copy2(path,target)
  if path.name=='PAUSE' or path.name=='error.json' or '.error.json' in path.name:path.unlink()
  archived.append(str(path))
fleet.atomic(fleet.OPS/'resume_preparation.json',dict(pin_checks=checks,archived=archived,code_sha256=digest(fleet.OPS/'fleet.py')))
for handle in locks:handle.close()
processes=[]
for kind,root in fleet.ROOTS.items():
 with (root/'logs/controller.isolated.log').open('ab',buffering=0) as log:
  p=subprocess.Popen([fleet.PYTHON,'-u',str(fleet.OPS/'fleet.py'),'controller',kind],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 start_ticks=Path(f'/proc/{p.pid}/stat').read_text().rsplit(') ',1)[1].split()[19]
 record=dict(pid=p.pid,start_ticks=start_ticks,kind=kind,host='LLMOptimizer2',supervisor=str(fleet.OPS/'fleet.py'))
 fleet.atomic(root/'control/isolated_controller.process.json',record)
 processes.append(record)
fleet.atomic(fleet.OPS/'launch.json',dict(controllers=processes))
print(json.dumps(dict(pin_checks=checks,controllers=processes),indent=2))
