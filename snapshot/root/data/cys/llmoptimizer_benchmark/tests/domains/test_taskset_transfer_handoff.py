import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import fcntl


SOURCE = Path(__file__).resolve().parents[2] / 'examples/benchmark/domains/taskset_transfer_handoff.py'
spec = importlib.util.spec_from_file_location('taskset_handoff_tested', SOURCE)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTest(unittest.TestCase):
    def controller_fixture(self, root):
        campaign, training = root / 'campaign', root / 'training'
        for path in (campaign / 'locks', campaign / 'distributed', training):
            path.mkdir(parents=True, exist_ok=True)
        job = dict(id='train.tpe.' + handoff.TASK, task_id=handoff.TASK, method='tpe',
                   side='train', output_root=str(root / 'histories'))
        history = root / 'histories' / handoff.TASK / 'tpe/seed1/history.json'
        history.parent.mkdir(parents=True)
        history.write_text(json.dumps({'observations': [{'trial_state': 0, 'objectives': [1.0]}] * 100}))
        plan = {'hosts': {handoff.HOST: {'hostname': 'test-host', 'workers': [
            {'name': handoff.WORKER, 'cpus': handoff.CPUS}], 'reservations': []}}}
        handoff.write(campaign / 'distributed/plan.json', plan)
        handoff.write(training / 'handoff.json', dict(nonce='test', returncode=0, job=job,
                      evaluator_exit_observed_at=123, original_worker_state={'name': handoff.WORKER}))
        return campaign, training, plan

    def test_controller_uses_disjoint_cores_and_full_schedule(self):
        with tempfile.TemporaryDirectory() as td:
            campaign, training, plan = self.controller_fixture(Path(td))
            commands = []

            class Child:
                pid = 123456

                def poll(self):
                    return 0

            def launch(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index('--output') + 1])
                handoff.write(output / 'status.json', {'status': 'complete'})
                self.assertEqual(len(kwargs['pass_fds']), 2)
                return Child()

            with patch.object(handoff, 'CAMPAIGN', campaign), patch.object(handoff, 'check_host', return_value=plan), \
                    patch.object(handoff, 'validate_ready', return_value={}), \
                    patch.object(handoff.os, 'sched_setaffinity'), patch.object(handoff.os, 'nice'), \
                    patch.object(handoff.subprocess, 'run') as run, patch.object(handoff.subprocess, 'Popen', side_effect=launch):
                handoff.execute(training, 'test')
            self.assertEqual([c[2] for c in commands], ['28,29', '30,31'])
            self.assertTrue(all(c[c.index('--steps') + 1] == '100000' for c in commands))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(handoff.read(campaign / 'distributed/plan.json'), plan)
            self.assertEqual(handoff.read(training / 'controller_state.json')['status'], 'complete')

    def test_busy_core_prevents_every_training_launch(self):
        with tempfile.TemporaryDirectory() as td:
            campaign, training, plan = self.controller_fixture(Path(td))
            with (campaign / 'locks/core.LLMOptimizer4.28.lock').open('a') as occupied:
                fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(handoff, 'CAMPAIGN', campaign), patch.object(handoff, 'check_host', return_value=plan), \
                        patch.object(handoff, 'validate_ready', return_value={}), patch.object(handoff.subprocess, 'Popen') as launch:
                    with self.assertRaises(BlockingIOError):
                        handoff.execute(training, 'test')
                    launch.assert_not_called()

    def test_drain_preserves_running_evaluator_and_observes_exit(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            child_code = 'import pathlib,sys,time; time.sleep(0.5); pathlib.Path(sys.argv[1]).write_text("finished")'
            parent_code = ('import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]]); '
                           'print(p.pid,flush=True); p.wait(); time.sleep(30)')
            parent = subprocess.Popen([sys.executable, '-c', parent_code, child_code, str(directory / 'evidence')],
                                      stdout=subprocess.PIPE, text=True)
            child_pid = int(parent.stdout.readline())
            parent_info, child_info = handoff.process(parent.pid), handoff.process(child_pid)
            try:
                handoff.pause_supervisor(parent_info, child_info)
                self.assertEqual(handoff.process(parent.pid)['state'], 'T')
                self.assertIsNone(handoff.exit_code(child_info))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and handoff.exit_code(child_info) is None:
                    time.sleep(0.05)
                self.assertEqual(handoff.exit_code(child_info), 0)
                self.assertEqual((directory / 'evidence').read_text(), 'finished')
                self.assertEqual(handoff.process(parent.pid)['state'], 'T')
                handoff.retire_supervisor(parent_info)
                self.assertEqual(parent.wait(timeout=5), -signal.SIGTERM)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()
                parent.stdout.close()

    def test_pid_reuse_cannot_be_treated_as_completion(self):
        identity = handoff.process(os.getpid())
        identity['start_ticks'] = 'wrong'
        self.assertIsNone(handoff.same_process(identity))
        with self.assertRaisesRegex(RuntimeError, 'disappeared'):
            handoff.exit_code(identity)

    def test_parent_child_mismatch_is_rejected_before_signal(self):
        parent = handoff.process(os.getpid())
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            handoff.pause_supervisor(parent, parent)

    def test_unverified_preparation_cannot_launch(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            (directory / 'launch_ready.json').write_text(json.dumps({'component_checks_passed': False}))
            with self.assertRaisesRegex(RuntimeError, 'readiness'):
                handoff.validate_ready(directory)

    def test_changed_training_code_cannot_launch(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            (directory / 'code').mkdir()
            (directory / 'code/train.py').write_text('changed')
            (directory / 'launch_ready.json').write_text(json.dumps({
                'component_checks_passed': True, 'cpus': [28, 29, 30, 31],
                'methods': ['fsbo', 'rgpe'], 'full_fsbo_updates': 100000,
                'files': {'train.py': 'wrong'},
            }))
            with self.assertRaisesRegex(RuntimeError, 'code changed'):
                handoff.validate_ready(directory)


if __name__ == '__main__':
    unittest.main()
