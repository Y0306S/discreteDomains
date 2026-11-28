"""Reload only the detached controller; all task and SQL processes stay alive."""
import signal
from common import *
require_host('LLMOptimizer4')
record=load(ROOT/'processes/bolt_supervisor.json')
assert alive(record) and str(ROOT/'runtime.py') in command(record['pid'])
before={p.name:load(p) for p in (ROOT/'processes').glob('sql*.json') if int(p.stem[3:])<15}
assert all(alive(r) for r in before.values())
os.kill(record['pid'],signal.SIGTERM)
for _ in range(50):
    if not alive(record):break
    time.sleep(.1)
assert not alive(record)
assert all(alive(r) for r in before.values()), 'A detached worker unexpectedly stopped'
new=launch([PYTHON,'-u',str(ROOT/'runtime.py'),'bolt'],'bolt_supervisor',environment())
atomic(ROOT/'cpu5/controller_reload.json',dict(at=time.time(),previous=record,new=new,
    existing_workers_preserved=15))
print(json.dumps(dict(controller_pid=new['pid'],existing_workers_preserved=15)))
