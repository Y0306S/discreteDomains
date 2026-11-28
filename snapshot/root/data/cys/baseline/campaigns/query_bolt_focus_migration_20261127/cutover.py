"""Run on Optimizer4 after the complete drained audit and endpoint probe."""
import fcntl
from common import *

audit=load(ROOT/'saved_state_audit.json');preflight=load(ROOT/'preflight.json')
assert audit['retained_prefixes_unchanged'] and not audit['uncertain_measurements']
assert len(preflight['services'])==16 and len(preflight['databases'])==15
assert (SEM/'PAUSE').exists() and (BOLT/'PAUSE').exists()
assert load(ROOT/'forward.status.json')['live']==1
assert load(ROOT/'vae.status.json')['live']==16
handles=[]
for name in ['control/controller.lock']+[f'services/slot{i}.lock' for i in range(8)]:
    handle=(BOLT/name).open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
for kind in ['bolt','semanticopt']:
    for file,expected in load(ROOT/'before'/kind/'identity.json')['files'].items():
        assert sha(file)==expected,file
atomic(ROOT/'cutover_baseline.json',dict(at=time.time(),status=load(BOLT/'status.json'),
    progress={t:load(BOLT/'runs'/t/'progress.json',{}) for t in load(BOLT/'recipe.json')['tasks']}))
for h in handles:h.close()
atomic(BOLT/'control/bolt_focus_migration.json',dict(root=str(ROOT),placement=str(ROOT/'placement.json'),at=time.time()))
(BOLT/'PAUSE').unlink()
record=launch([PYTHON,'-u',str(ROOT/'runtime.py'),'bolt'],'bolt_supervisor',environment())
print(json.dumps(record))
