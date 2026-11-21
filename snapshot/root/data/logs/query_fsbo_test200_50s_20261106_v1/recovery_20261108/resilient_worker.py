"""Operational queue recovery; use the immutable FSBO evaluator and recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EXPANSION = Path('/root/data/logs/query_fsbo_local_expansion_20261107_v1')
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')


def runnable(status):
    return status not in ('complete', 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--task')
    parser.add_argument('--stop-after-batches', type=int)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    import query_fsbo_eval as evaluation
    recipe = evaluation.verify(ROOT)
    ledger = json.loads((ROOT/'current_workers.json').read_text())
    worker = next(w for w in ledger['workers'] if w['name'] == args.worker)
    os.sched_setaffinity(0, {worker['cpus'][0]})
    reservation = json.loads((EXPANSION/'reservation.json').read_text())
    if evaluation.digest(ROOT/'recipe.json') != reservation['recipe_sha256']:
        raise ValueError('Frozen recipe changed')
    # Only transport waiting changes. SQL execution stays capped at 50 seconds.
    original_init = evaluation.QueryVAEBackend.__init__
    def initialize(self, manifest, task, fallback, seed=123, timeout=300):
        original_init(self, manifest, task, fallback, seed=seed, timeout=1800)
    evaluation.QueryVAEBackend.__init__ = initialize
    endpoint = f"127.0.0.1:{worker['db_port']}:imdb"
    lock_id = hashlib.sha256(endpoint.encode()).hexdigest()[:16]
    status_path = ROOT/'recovery_20261108'/f'{args.worker}.json'
    def state(status, **values):
        evaluation.atomic_json(status_path, dict(status=status, at=time.time(), pid=os.getpid(),
            worker=args.worker, host=worker['host'], inference_timeout_seconds=1800,
            sql_timeout_seconds=recipe['timeout_seconds'], **values))
    with evaluation.lock(Path('/tmp')/('ceb_baseline_db_'+lock_id+'.lock')):
        state('starting')
        skipped = []
        for task in ([args.task] if args.task else recipe['allowed_tasks']):
            if (ROOT/'STOP').exists() or (worker['host']=='LLMOptimizer2' and (EXPANSION/'STOP').exists()):
                state('stopped')
                return
            task_status = ROOT/'runs'/task/'status.json'
            if task_status.exists():
                status = json.loads(task_status.read_text())['status']
                if not runnable(status):
                    if status == 'failed':
                        skipped.append(task)
                        print(json.dumps(dict(event='failed_task_isolated', task=task)), flush=True)
                    continue
            if worker['host'] == 'LLMOptimizer2':
                cg = Path('/sys/fs/cgroup')
                available = int((cg/'memory.max').read_text())-int((cg/'memory.current').read_text())
                if available < (reservation['orpo_memory_gib']+reservation['headroom_gib'])*2**30:
                    state('memory_admission_stopped', available_gib=available/2**30)
                    raise RuntimeError('Reserved ML memory headroom is unavailable')
            try:
                state('running', task=task, skipped_failed_tasks=skipped)
                result = evaluation.run_task(ROOT, recipe, task, worker, args.stop_after_batches)
            except BlockingIOError:
                continue
            except Exception as exc:
                # Retain the failed task and its checkpoint. Never replay an
                # unknown physical query, and never manufacture an objective.
                failed = json.loads(task_status.read_text()) if task_status.exists() else {}
                if failed.get('status') != 'failed':
                    state('infrastructure_failed', task=task, error=repr(exc))
                    raise
                skipped.append(task)
                print(json.dumps(dict(event='task_failed_and_isolated', task=task,
                                      error=repr(exc))), flush=True)
                traceback.print_exc()
                if args.task:
                    raise
                continue
            print(json.dumps(result), flush=True)
            if args.stop_after_batches:
                state('bounded_check_complete', task=task)
                return
        state('queue_pass_finished', skipped_failed_tasks=skipped)


if __name__ == '__main__':
    main()
