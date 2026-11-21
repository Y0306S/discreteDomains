"""Add local endpoints to the existing query queue without changing its frozen recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import query_fsbo_eval as evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--stop-after-batches', type=int)
    args = parser.parse_args()
    plan = json.loads((ROOT/'plan.json').read_text())
    reservation = json.loads((ROOT/'reservation.json').read_text())
    campaign = Path(reservation['campaign'])
    if evaluation.digest(campaign/'recipe.json') != reservation['recipe_sha256']:
        raise ValueError('Frozen campaign recipe changed')
    if socket.gethostname() != 'cs-94bac-0693e-server':
        raise ValueError('Local expansion must run on LLMOptimizer2')
    recipe = evaluation.verify(campaign)
    worker = next(w for w in plan['workers'] if w['name'] == args.worker)
    if worker['host'] != 'LLMOptimizer2':
        raise ValueError('Invalid worker host')
    os.sched_setaffinity(0, {worker['cpus'][0]})
    status_path = ROOT/'workers'/args.worker/'production.json'
    def state(status, **values):
        evaluation.atomic_json(status_path, dict(status=status, at=time.time(), pid=os.getpid(),
            worker=worker['name'], cpus=worker['cpus'], db_port=worker['db_port'],
            campaign=str(campaign), recipe_sha256=reservation['recipe_sha256'], **values))
    endpoint = f"127.0.0.1:{worker['db_port']}:imdb"
    lock_id = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    with evaluation.lock(Path('/tmp')/('ceb_baseline_db_'+lock_id+'.lock')):
        state('starting')
        for task in recipe['allowed_tasks']:
            if (campaign/'STOP').exists() or (ROOT/'STOP').exists():
                state('stopped')
                return
            task_status = campaign/'runs'/task/'status.json'
            if task_status.exists():
                old = json.loads(task_status.read_text())
                if old['status'] == 'complete':
                    continue
                if old['status'] == 'failed':
                    raise RuntimeError('Inspect failed query before resuming: '+task)
            cg = Path('/sys/fs/cgroup')
            available = int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())
            reserve = (reservation['orpo_memory_gib']+reservation['headroom_gib'])*2**30
            if available < reserve:
                state('waiting_for_reserved_memory', available_gib=available/2**30)
                raise RuntimeError('ORPO memory admission envelope is unavailable')
            try:
                state('running', task=task)
                result = evaluation.run_task(campaign, recipe, task, worker, args.stop_after_batches)
            except BlockingIOError:
                continue
            except BaseException as exc:
                state('failed', task=task, error=repr(exc))
                raise
            print(json.dumps(result), flush=True)
            if args.stop_after_batches:
                state('pilot_complete', task=task)
                return
        state('finished')


if __name__ == '__main__':
    main()
