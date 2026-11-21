"""Restore the remote worker that stopped when it encountered the failed task."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time

root=Path('/root/data/logs/query_fsbo_local_expansion_20261107_v1')
campaign=Path('/root/data/logs/query_fsbo_test200_50s_20261106_v1')
ledger=json.loads((campaign/'current_workers.json').read_text())
worker=next(w for w in ledger['workers'] if w['name']=='w17')
proc=Path('/proc')/str(worker['pid'])
if proc.exists():
    assert (proc/'stat').read_text().split()[2]=='Z'
assert 'Inspect failed query before resuming: CEB_2A593' in Path(worker['log']).read_text()[-1000:]
assert hashlib.sha256((campaign/'recipe.json').read_bytes()).hexdigest()==json.loads((root/'reservation.json').read_text())['recipe_sha256']
assert not (campaign/'STOP').exists()
assert not any(json.loads(p.read_text())['status']=='failed' for p in (campaign/'runs').glob('*/status.json'))
endpoint=f"127.0.0.1:{worker['db_port']}:imdb"
lock=Path('/tmp')/('ceb_baseline_db_'+hashlib.sha256(endpoint.encode()).hexdigest()[:16]+'.lock')
with lock.open('a') as handle:
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert socket.gethostname() != 'cs-94bac-0693e-server'
cg=Path('/sys/fs/cgroup')
assert int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())>=32*2**30
command=['taskset','-c',str(worker['cpus'][0]),'/root/data/cys/envs/bolt_clean_venv/bin/python','-u',
         '/root/data/cys/llmoptimizer_benchmark/tools/query_fsbo_eval.py','run-worker','--campaign',str(campaign),'--worker','w17']
log=campaign/'logs'/('w17.recovered_'+time.strftime('%Y%m%d_%H%M%S')+'.log')
with log.open('a') as out:
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,
                             env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
updated=dict(worker,pid=process.pid,actual_command=command,command=command,log=str(log),started_at=time.time(),
             process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],previous_pid=worker['pid'],
             recovery_reason='Original task completed, then queue stopped on the failed CEB_2A593 task; that failure is now recovered.')
(root/'w17_recovery_process.json').write_text(json.dumps(updated,indent=2)+'\n')
(root/'current_workers.before_w17_recovery.json').write_text(json.dumps(ledger,indent=2)+'\n')
ledger['workers']=[updated if w['name']=='w17' else w for w in ledger['workers']]
tmp=campaign/'current_workers.w17_recovery.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(tmp,campaign/'current_workers.json')
print(json.dumps(updated,indent=2))
