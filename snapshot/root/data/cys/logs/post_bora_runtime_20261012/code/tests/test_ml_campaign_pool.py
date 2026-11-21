import json
import contextlib
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmark'))
import ml_campaign_pool as pool
import ml_deferred_campaign as deferred
import ml_fixed_campaign as campaign


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bolt, self.semantic, self.tosfit, self.bora = [self.root / m for m in
                                                         ('bolt', 'semantic', 'tosfit', 'bora')]
        self.config = {
            'host': 'fixture', 'hostname': socket.gethostname(),
            'campaigns': [str(p) for p in (self.bolt, self.semantic, self.tosfit, self.bora)],
            'cpus': list(range(4, 22)) + list(range(23, 40)), 'scheduler_cpus': [22],
            'max_workers': 12, 'gpu_budget_gib': {'0': 72, '1': 76, '2': 76, '3': 76},
            'gpu_headroom_gib': 4,
            'gpu_reservations_gib': {'bolt_ml': [24], 'semanticopt': [24],
                                     'tosfit': [32, 22], 'bora': [36]},
        }
        self.memory = {g: {'used': 0, 'total': 80} for g in range(4)}
        self.pending = {str(p): 0 for p in (self.bolt, self.semantic, self.tosfit, self.bora)}
        self.config_path = self.root / 'pool.json'
        campaign.save(self.config_path, self.config)
        for root, method in [(self.bolt, 'bolt_ml'), (self.semantic, 'semanticopt'),
                             (self.tosfit, 'tosfit'), (self.bora, 'bora')]:
            campaign.save(root / 'protocol.json', {
                'methods': [method], 'resource_pool': str(self.config_path),
                'resource_pool_sha256': campaign.digest(self.config_path),
                'hosts': {'fixture': {'hostname': socket.gethostname(), 'cpus': self.config['cpus'],
                                      'limits': {method: 12}, 'gpu_slots': [0, 1, 2, 3]}},
            })
            campaign.save(root / 'queues' / f'{method}.json', [])

    def allocation(self, states=(), occupied=()):
        return pool.Allocation(self.config, list(states), dict(self.pending), self.memory, occupied)

    def test_free_slots_fill_while_two_bolt_evaluators_remain(self):
        allocation = self.allocation([
            {'method': 'bolt_ml', 'cpus': [4], 'gpu': 1},
            {'method': 'bolt_ml', 'cpus': [14], 'gpu': 0},
        ])
        self.assertTrue(allocation.available(self.semantic))
        assigned = []
        for cpu in [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]:
            gpu = allocation.choose_gpu('semanticopt', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
            self.assertIsNotNone(gpu)
            allocation.reserve('semanticopt', [cpu], gpu)
            assigned.append(gpu)
        self.assertEqual(allocation.workers, 12)
        self.assertFalse(allocation.available(self.semantic))
        self.assertEqual(dict(allocation.used_gpu), {0: 72, 1: 72, 2: 72, 3: 72})
        self.assertIsNone(allocation.choose_gpu('semanticopt', [0, 1, 2, 3]))
        self.assertEqual(assigned[:2], [2, 3])

    def test_unassigned_earlier_tasks_preserve_queue_priority(self):
        self.pending[str(self.bolt)] = 1
        allocation = self.allocation()
        self.assertFalse(allocation.available(self.semantic))
        self.assertTrue(allocation.available(self.bolt))
        self.assertEqual(allocation.waiting_for(self.bora), str(self.bolt))

    def test_completed_worker_releases_capacity_for_next_method(self):
        states = [{'method': 'semanticopt', 'cpus': [4 + i], 'gpu': i // 3} for i in range(12)]
        allocation = self.allocation(states)
        self.assertFalse(allocation.available(self.tosfit))
        allocation = self.allocation(states[:6])
        self.assertTrue(allocation.available(self.tosfit))
        self.assertEqual(allocation.choose_gpu('tosfit', ['0,1', '1,0', '2,3', '3,2']), '2,3')

    def test_tosfit_reserves_both_ordered_devices(self):
        allocation = self.allocation()
        allocation.reserve('tosfit', [4], '1,0')
        self.assertEqual(dict(allocation.used_gpu), {1: 32, 0: 22})
        allocation.reserve('tosfit', [5], '0,1')
        self.assertEqual(dict(allocation.used_gpu), {1: 54, 0: 54})
        self.assertIsNone(allocation.choose_gpu('semanticopt', [0, 1]))

    def test_bora_memory_allowance_blocks_overpacking(self):
        allocation = self.allocation()
        allocation.reserve('bora', [4], 0)
        allocation.reserve('bora', [5], 0)
        self.assertEqual(allocation.used_gpu[0], 72)
        self.assertIsNone(allocation.choose_gpu('semanticopt', [0]))

    def test_external_gpu_and_cpu_occupancy_is_respected(self):
        self.memory[0]['used'] = 65
        allocation = self.allocation(occupied=[4, 5, 6])
        self.assertIsNone(allocation.choose_gpu('semanticopt', [0]))
        with self.assertRaisesRegex(RuntimeError, 'Overlapping'):
            allocation.reserve('semanticopt', [5], 1)

    def test_pending_model_loads_reserve_future_memory(self):
        allocation = self.allocation()
        for cpu in [4, 5, 6]:
            allocation.reserve('semanticopt', [cpu], 1)
        self.assertIsNone(allocation.choose_gpu('semanticopt', [1]))

    def test_changed_pool_fails_closed(self):
        campaign.save(self.config_path, {**self.config, 'max_workers': 100})
        with self.assertRaisesRegex(RuntimeError, 'pool changed'):
            pool.configuration(self.semantic, 'fixture')

    def test_resource_lock_excludes_another_scheduler(self):
        with patch.object(pool, 'snapshot', return_value=self.allocation()):
            with pool.resources(self.semantic, 'fixture'):
                self.assertIsNone(campaign.claim(self.config_path.with_suffix('.lock')))
        lock = campaign.claim(self.config_path.with_suffix('.lock'))
        self.assertIsNotNone(lock)
        lock.close()

    def test_snapshot_counts_live_peers_and_only_unassigned_tasks(self):
        task = {'task_type': 'cls', 'dataset': 'toy', 'algorithm': 'adaboost'}
        campaign.save(self.bolt / 'queues/bolt_ml.json', [task])
        state = {**task, 'method': 'bolt_ml', 'host': 'fixture', 'status': 'running',
                 'gpu': 0, 'cpus': [4], 'pid': 123}
        campaign.save(self.bolt / 'state/bolt_ml/cls/toy/adaboost.json', state)
        with patch.object(campaign, 'process_alive', return_value=True), \
                patch.object(pool, 'gpu_memory', return_value=self.memory), \
                patch.object(pool, 'occupied_cpus', return_value=set()):
            allocation = pool.snapshot(self.config)
        self.assertEqual(allocation.workers, 1)
        self.assertTrue(allocation.available(self.semantic))
        self.assertEqual(allocation.used_cpus, {4})

    def test_shared_watcher_uses_capacity_instead_of_completion_barrier(self):
        protocol = campaign.load(self.semantic / 'protocol.json')
        protocol['after_campaign'] = str(self.bolt)
        protocol['hosts']['fixture']['scheduler_cpus'] = sorted(os.sched_getaffinity(0))[:1]
        campaign.save(self.semantic / 'protocol.json', protocol)
        with patch.object(deferred, 'predecessor_readiness') as predecessor, \
                patch.object(deferred, 'resource_readiness', return_value=[]) as ready, \
                patch.object(deferred, 'verify_campaign') as verify, \
                patch.object(deferred, 'launch') as launch, \
                patch.object(deferred.os, 'sched_setaffinity'):
            deferred.run(self.semantic, 'fixture')
        predecessor.assert_not_called()
        verify.assert_called_once_with(self.semantic)
        launch.assert_called_once()
        self.assertEqual(ready.call_count, 2)
        self.assertTrue(all(call.kwargs['shared'] for call in ready.call_args_list))

    def test_scheduler_starts_next_method_without_stopping_predecessor(self):
        task = {'task_type': 'cls', 'dataset': 'earlier', 'algorithm': 'adaboost', 'metric': 'bal_acc'}
        session = self.bolt / 'sessions/bolt_ml/cls/earlier/adaboost'
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)',
                                  '--session-dir', str(session)])
        self.addCleanup(lambda: child.wait() if child.poll() is not None else (child.terminate(), child.wait()))
        campaign.save(self.bolt / 'queues/bolt_ml.json', [task])
        campaign.save(self.bolt / 'state/bolt_ml/cls/earlier/adaboost.json', {
            **task, 'method': 'bolt_ml', 'host': 'fixture', 'status': 'running', 'pid': child.pid,
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks': Path(f'/proc/{child.pid}/stat').read_text().rsplit(') ', 1)[1].split()[19],
            'session': str(session), 'cpus': [4], 'gpu': 0,
        })
        self.config['max_workers'] = 2
        campaign.save(self.config_path, self.config)
        for root in (self.bolt, self.semantic, self.tosfit, self.bora):
            protocol = campaign.load(root / 'protocol.json')
            protocol['resource_pool_sha256'] = campaign.digest(self.config_path)
            campaign.save(root / 'protocol.json', protocol)
        tasks = [{**task, 'dataset': str(i)} for i in range(3)]
        campaign.save(self.semantic / 'tasks_132.json', tasks)
        campaign.save(self.semantic / 'runtime_policy.json', {})
        campaign.save(self.semantic / 'queues/semanticopt.json', tasks)
        campaign.save(self.semantic / 'reused/semanticopt.json', [])
        (self.semantic / 'code').mkdir()
        protocol = campaign.load(self.semantic / 'protocol.json')
        protocol.update(task_count=3, ckpt32_policy='rerun-all',
                        manifest_sha256=campaign.digest(self.semantic / 'tasks_132.json'),
                        policy_sha256=campaign.digest(self.semantic / 'runtime_policy.json'))
        spec = protocol['hosts']['fixture']
        spec.update(cpus=[4, 5, 6, 7], memory_gib=10000, gpu_methods=['semanticopt'])
        campaign.save(self.semantic / 'protocol.json', protocol)
        started = []

        def command(folder, method, current_task, cpus, gpu):
            self.assertIsNone(child.poll())
            self.assertNotIn(4, cpus)
            started.append(current_task['dataset'])
            target = folder / 'sessions' / method / campaign.task_key(current_task)
            code = ("import json,pathlib,sys,time; time.sleep(.1); p=pathlib.Path(sys.argv[2]); "
                    "p.mkdir(parents=True,exist_ok=True); p.joinpath('progress.json').write_text(json.dumps("
                    "dict(attempts=100,successful=100,failed=0,clean_complete=True,"
                    "budget_exhausted=True,status='complete')))")
            return [sys.executable, '-c', code, '--session-dir', str(target)], os.environ.copy(), target

        sleep = time.sleep
        with patch.object(campaign, 'command', command), \
                patch.object(campaign.time, 'sleep', lambda s: sleep(min(s, .03))), \
                patch.object(pool, 'gpu_memory', return_value=self.memory), \
                patch.object(pool, 'occupied_cpus', return_value=set()), \
                contextlib.redirect_stdout(io.StringIO()):
            campaign.run(self.semantic, 'fixture')
        self.assertEqual(started, ['0', '1', '2'])
        self.assertIsNone(child.poll())
        self.assertEqual(campaign.status(self.semantic, quiet=True)['semanticopt']['new_clean'], 3)


if __name__ == '__main__':
    unittest.main()
