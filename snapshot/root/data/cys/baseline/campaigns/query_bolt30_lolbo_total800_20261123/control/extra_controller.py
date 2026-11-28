"""Supervise the authorized eighth worker alongside the original seven."""
import fcntl,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY='/root/data/cys/envs/bolt_clean_venv/bin/python'
def atomic(path,data):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(dict(at=time.time(),**data),indent=2)+'\n');tmp.replace(path)
def main():
    lock=(ROOT/'control/extra_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'PAUSE').exists()
    assert json.loads((ROOT/'evidence/worker07/database.ready.json').read_text())['ready']
    start=Path('/proc/self/stat').read_text().split()[21]
    atomic(ROOT/'control/extra_controller.process.json',dict(pid=os.getpid(),start_ticks=start))
    cmd=['ssh','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
         'LLMOptimizer3',shlex.join([PY,'-u',str(ROOT/'control/extra_worker.py')])]
    with (ROOT/'logs/slot07.log').open('ab',buffering=0) as log:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while True:
        code=child.poll()
        worker=ROOT/'services/worker07.status.json'
        state=json.loads(worker.read_text()) if worker.exists() else {}
        status='running' if code is None else 'complete' if code==0 and state.get('status')=='idle_complete' else 'paused' if code==0 and (ROOT/'PAUSE').exists() else 'failed'
        atomic(ROOT/'control/extra_controller.status.json',dict(status=status,supervisor_pid=os.getpid(),supervisor_start_ticks=start,child_pid=child.pid,
               live_worker_slots=int(code is None),slot=7,host='LLMOptimizer3',task=state.get('task'),exit_code=code))
        if status=='failed':
            (ROOT/'PAUSE').touch();raise RuntimeError('Additional BOLT worker failed; inspect slot07.log and its journal')
        if code is not None:return
        time.sleep(5)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        atomic(ROOT/'control/extra_controller.error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
