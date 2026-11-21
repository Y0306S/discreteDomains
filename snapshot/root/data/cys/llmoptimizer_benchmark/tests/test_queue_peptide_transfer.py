import importlib.util
import json
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'queue_peptide_transfer.py'
spec = importlib.util.spec_from_file_location('peptide_transfer_queue', SCRIPT)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.campaign = self.base / 'campaign'
        self.campaign.joinpath('workers').mkdir(parents=True)
        for method in queue.METHODS:
            queue.write(self.campaign / method / 'recipe.json', {'test': ['pep_235']})
        self.args = Mock(side='local', cpus=[12], servers=['http://127.0.0.1:18871'])

    def run_dispatch(self, occupied, fsbo_done=False, rgpe_ready=False):
        child = Mock(pid=123456)
        with patch.object(queue, 'CAMPAIGN', self.campaign), \
                patch.object(queue, 'BASELINE', self.base / 'baseline'), \
                patch.object(queue, 'TRAINING', self.base / 'training'), \
                patch.object(queue, 'processes', return_value=[]), \
                patch.object(queue, 'process', return_value={'start': '100'}), \
                patch.object(queue, 'process_start', return_value='100'), \
                patch.object(queue, 'occupied_cpus', return_value=set(occupied)), \
                patch.object(queue, 'memory_available', return_value=100.0), \
                patch.object(queue, 'completed', side_effect=lambda m: {'pep_235'} if m == 'fsbo' and fsbo_done else set()), \
                patch.object(queue, 'ready', side_effect=lambda m: m == 'fsbo' or rgpe_ready), \
                patch.object(queue.subprocess, 'Popen', return_value=child) as start, \
                patch.object(queue.signal, 'signal'), \
                patch.object(queue.time, 'sleep', side_effect=InterruptedError('end test cycle')):
            with self.assertRaises(InterruptedError):
                queue.dispatch(self.args)
            return start.call_args_list, json.loads((self.campaign / 'workers' / 'queue_local.json').read_text())

    def test_running_tpe_cpu_is_not_reused(self):
        calls, state = self.run_dispatch({12})
        self.assertEqual(calls, [])
        self.assertEqual(state['released_cpus'], [])

    def test_free_cpu_starts_fsbo_before_rgpe(self):
        calls, state = self.run_dispatch(set(), rgpe_ready=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state['active'][0]['method'], 'fsbo')
        self.assertEqual(calls[0].args[0][:3], ['taskset', '-c', '12'])

    def test_rgpe_waits_for_complete_bank_validation(self):
        calls, state = self.run_dispatch(set(), fsbo_done=True)
        self.assertEqual(calls, [])
        self.assertFalse(state['ready']['rgpe'])

    def test_fsbo_completion_hands_slot_to_validated_rgpe(self):
        calls, state = self.run_dispatch(set(), fsbo_done=True, rgpe_ready=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state['active'][0]['method'], 'rgpe')

    def test_two_dispatchers_cannot_claim_same_task(self):
        path = self.base / 'claim.lock'
        with queue.claim(path) as first:
            self.assertIsNotNone(first)
            with queue.claim(path) as second:
                self.assertIsNone(second)
        with queue.claim(path) as third:
            self.assertIsNotNone(third)

    def test_running_task_lock_prevents_duplicate_dispatch(self):
        with queue.claim(self.campaign / 'fsbo' / 'locks' / 'pep_235.lock'):
            calls, state = self.run_dispatch(set())
        self.assertEqual(calls, [])
        self.assertEqual(state['active'], [])

    def test_skipped_busy_tpe_task_does_not_park_released_cpu(self):
        queue.write(self.campaign / 'workers' / 'queue_local.json', {
            'halted_cpus': [12],
            'baseline_owners': {'12': {'last_task': 'pep_still_owned_elsewhere', 'index': 0}},
        })
        calls, state = self.run_dispatch(set())
        self.assertEqual(len(calls), 1)
        self.assertEqual(state['halted_cpus'], [])
        self.assertEqual(state['active'][0]['method'], 'fsbo')

    def test_actual_transfer_error_keeps_cpu_parked(self):
        queue.write(self.campaign / 'workers' / 'error_local_cpu12.json', {'returncode': 1})
        queue.write(self.campaign / 'workers' / 'queue_local.json', {'halted_cpus': [12]})
        calls, state = self.run_dispatch(set())
        self.assertEqual(calls, [])
        self.assertEqual(state['halted_cpus'], [12])

    def test_route_overrides_only_its_machine_and_core(self):
        queue.write(self.campaign / 'workers' / 'vae_routes.json', {
            'local': {'12': 'http://127.0.0.1:18881'},
            'remote': {'12': 'http://127.0.0.1:18885'},
        })
        with patch.object(queue, 'CAMPAIGN', self.campaign):
            self.assertEqual(queue.configured_server('local', 12, 0, self.args.servers),
                             'http://127.0.0.1:18881')
            self.assertEqual(queue.configured_server('remote', 12, 0, self.args.servers),
                             'http://127.0.0.1:18885')
            self.assertEqual(queue.configured_server('local', 13, 0, self.args.servers),
                             self.args.servers[0])

    def test_reroute_signals_once_and_checks_process_identity(self):
        row = dict(pid=123, process_start='100', method='fsbo',
                   server=self.args.servers[0])
        with patch.object(queue, 'CAMPAIGN', self.campaign), \
                patch.object(queue, 'process', return_value={'start': '100'}), \
                patch.object(queue.os, 'kill') as kill:
            queue.request_reroute(row, 'http://127.0.0.1:18881')
            queue.request_reroute(row, 'http://127.0.0.1:18881')
        kill.assert_called_once_with(123, queue.signal.SIGTERM)
        self.assertEqual(row['reroute_to'], 'http://127.0.0.1:18881')
        stale = dict(pid=123, process_start='99', method='fsbo', server=self.args.servers[0])
        with patch.object(queue, 'CAMPAIGN', self.campaign), \
                patch.object(queue, 'process', return_value={'start': '100'}), \
                patch.object(queue.os, 'kill') as kill:
            queue.request_reroute(stale, 'http://127.0.0.1:18881')
        kill.assert_not_called()

    def test_reroute_requires_a_new_matching_checkpoint_and_successful_exit(self):
        recipe = self.campaign / 'fsbo' / 'recipe.json'
        row = dict(cpu=12, method='fsbo', task='pep_235', reroute_to='http://127.0.0.1:18881',
                   reroute_requested_at=time.time() - 1)
        progress = self.campaign / 'fsbo' / 'runs' / 'pep_235' / 'progress.json'
        queue.write(progress, {'identity': {'task': 'pep_235', 'recipe_sha256': queue.digest(recipe)},
                               'proposals': 50})
        with patch.object(queue, 'CAMPAIGN', self.campaign):
            self.assertTrue(queue.reroute_checkpoint(row, 0))
            self.assertFalse(queue.reroute_checkpoint(row, 1))
            row['reroute_requested_at'] = time.time() + 10
            self.assertFalse(queue.reroute_checkpoint(row, 0))

    def test_gpu_transition_restarts_cpu_runner_without_changing_vae_route(self):
        row = dict(pid=123, process_start='100', method='fsbo', server=self.args.servers[0])
        current = {'start': '100', 'words': [str(queue.ROOT/'peptide_transfer_eval.py')]}
        with patch.object(queue, 'process', return_value=current), \
                patch.object(queue, 'task_runner', return_value=queue.ROOT/'peptide_fsbo_gpu_eval.py'), \
                patch.object(queue.os, 'kill') as kill:
            queue.request_reroute(row, row['server'])
            queue.request_reroute(row, row['server'])
        kill.assert_called_once_with(123, queue.signal.SIGTERM)
        self.assertEqual(row['reroute_to'], self.args.servers[0])
        self.assertEqual(row['reroute_runner'], str(queue.ROOT/'peptide_fsbo_gpu_eval.py'))

    def test_gpu_runner_is_not_restarted_when_its_route_is_current(self):
        row = dict(pid=123, process_start='100', method='fsbo', server=self.args.servers[0])
        runner = queue.ROOT/'peptide_fsbo_gpu_eval.py'
        with patch.object(queue, 'process', return_value={'start': '100', 'words': [str(runner)]}), \
                patch.object(queue, 'task_runner', return_value=runner), \
                patch.object(queue.os, 'kill') as kill:
            queue.request_reroute(row, row['server'])
        kill.assert_not_called()

    def test_gpu_service_does_not_exclusively_reserve_its_shared_cpu(self):
        service = dict(pid=123,words=['python',str(queue.ROOT/'peptide_fsbo_gpu_service.py'),
                                     '--cpu','35'])
        worker = dict(pid=124,words=['python',str(queue.ROOT/'peptide_fsbo_gpu_eval.py'),
                                    '--cpu','36'])
        self.assertEqual(queue.occupied_cpus([service,worker]),{36})

    def test_intentionally_rerouted_partial_task_resumes_without_parking(self):
        queue.write(self.campaign / 'workers' / 'queue_local.json', {
            'active': [dict(pid=122, process_start='99', method='fsbo', task='pep_235', cpu=12,
                            server=self.args.servers[0], reroute_to='http://127.0.0.1:18881',
                            reroute_requested_at=time.time() - 1)],
        })
        queue.write(self.campaign / 'fsbo' / 'runs' / 'pep_235' / 'progress.json', {
            'identity': {'task': 'pep_235',
                         'recipe_sha256': queue.digest(self.campaign / 'fsbo' / 'recipe.json')},
            'proposals': 50,
        })
        queue.write(self.campaign / 'workers' / 'vae_routes.json', {
            'local': {'12': 'http://127.0.0.1:18881'},
        })
        calls, state = self.run_dispatch(set())
        self.assertEqual(len(calls), 1)
        self.assertEqual(state['halted_cpus'], [])
        self.assertEqual(state['active'][0]['task'], 'pep_235')
        self.assertEqual(state['active'][0]['server'], 'http://127.0.0.1:18881')


if __name__ == '__main__':
    unittest.main()
