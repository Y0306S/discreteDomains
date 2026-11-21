"""Exercise queue isolation without a model, database, or production mutations."""
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('resilient_worker',Path(__file__).with_name('resilient_worker.py'))
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class QueueIsolation(unittest.TestCase):
    def run_case(self, commit_failure):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'recovery_20261108').mkdir()
            (root/'recipe.json').write_text('{}')
            (root/'reservation.json').write_text('{"recipe_sha256":"frozen"}')
            (root/'current_workers.json').write_text(json.dumps(dict(workers=[dict(
                name='test',host='LLMOptimizer4',cpus=[min(os.sched_getaffinity(0))],db_port=1234)])))
            for task,status in [('done','complete'),('old_failure','failed')]:
                (root/'runs'/task).mkdir(parents=True)
                (root/'runs'/task/'status.json').write_text(json.dumps(dict(status=status)))
            calls=[]
            class Backend:
                def __init__(self,*args,**kwargs): self.timeout=kwargs['timeout']
            def atomic(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
            def run_task(campaign,recipe,task,slot,batches):
                calls.append(task)
                backend=Backend(None,None,None)
                assert backend.timeout==1800
                if task=='new_failure':
                    if commit_failure:
                        atomic(root/'runs'/task/'status.json',dict(status='failed'))
                    raise TimeoutError('simulated inference failure')
                return dict(status='complete',task=task)
            evaluation=types.SimpleNamespace(verify=lambda _:dict(allowed_tasks=[
                'done','old_failure','new_failure','next'],timeout_seconds=50),digest=lambda _:'frozen',
                QueryVAEBackend=Backend,atomic_json=atomic,lock=lambda _:nullcontext(),run_task=run_task)
            with patch.dict(sys.modules,query_fsbo_eval=evaluation),patch.object(worker,'ROOT',root),\
                 patch.object(worker,'EXPANSION',root),patch.object(sys,'argv',['worker','--worker','test']),\
                 patch.object(os,'sched_setaffinity'):
                if commit_failure:
                    worker.main()
                    self.assertEqual(calls,['new_failure','next'])
                    status=json.loads((root/'recovery_20261108/test.json').read_text())
                    self.assertEqual(status['skipped_failed_tasks'],['old_failure','new_failure'])
                else:
                    with self.assertRaises(TimeoutError):worker.main()
                    self.assertEqual(calls,['new_failure'])

    def test_known_task_failure_allows_next_task(self):
        self.run_case(True)

    def test_unrecorded_failure_stops_worker(self):
        self.run_case(False)


if __name__=='__main__':
    unittest.main()
