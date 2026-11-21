"""Recovery of interrupted installation and remote task ownership cleanup."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/examples/benchmark'))
spec = importlib.util.spec_from_file_location('post_bora_deployment', ROOT/'deployment.py')
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


class DeploymentTests(unittest.TestCase):
    def test_inference_pins_autotuning_without_changing_training(self):
        import optformer_worker
        with patch.dict(sys.modules, {'tensorflow':MagicMock()}), \
                patch.dict(os.environ, {'XLA_FLAGS':'--xla_force_host_platform_device_count=1'}):
            optformer_worker.configure_runtime(deterministic_inference=False)
            self.assertEqual(os.environ['XLA_FLAGS'], '--xla_force_host_platform_device_count=1')
            optformer_worker.configure_runtime()
            optformer_worker.configure_runtime()
            self.assertEqual(os.environ['XLA_FLAGS'].count('--xla_gpu_autotune_level=0'), 1)
            os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=4'
            with self.assertRaisesRegex(ValueError, 'deterministic'):
                optformer_worker.configure_runtime()

    def test_interrupted_install_keeps_backups_and_resumes_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaigns = {name:root/name for name in ('tosfit','optformer')}
            original = {'queue_enabled':False, 'old':True}
            prepared = {'queue_enabled':False, 'new':True}
            for method, campaign in campaigns.items():
                deployment.save(campaign/'protocol.json', original)
                deployment.save(campaign/'checksums.json', {'protocol.json':deployment.digest(campaign/'protocol.json')})
                deployment.save(campaign/'prepared.json', {'checksums_sha256':deployment.digest(campaign/'checksums.json')})
                deployment.save(campaign/'task_index.json', {'tasks':[]})
                deployment.save(root/'config'/(method+'.protocol.json'), prepared)
            manifest = {'files':{}, 'original_protocols':{m:deployment.digest(c/'protocol.json') for m,c in campaigns.items()}}
            deployment.save(root/'manifest.json', manifest)
            real_save = deployment.save
            failed = []
            def fail_after_protocol(path, value):
                if path == campaigns['tosfit']/'checksums.json' and not failed:
                    failed.append(True)
                    raise OSError('interrupted metadata write')
                real_save(path, value)
            with patch.object(deployment, 'ROOT', root), patch.object(deployment, 'CAMPAIGNS', campaigns), \
                    patch.object(deployment, 'verify', return_value=manifest), \
                    patch.object(deployment, 'hold'), patch.object(deployment, 'on_host'):
                with patch.object(deployment, 'save', side_effect=fail_after_protocol):
                    with self.assertRaisesRegex(OSError, 'interrupted'):
                        deployment.install()
                self.assertFalse((root/'installed.json').exists())
                self.assertFalse(deployment.install()['production_enabled'])
                self.assertFalse(deployment.install()['production_enabled'])
                for method, campaign in campaigns.items():
                    self.assertEqual(deployment.load(root/'before'/method/'protocol.json'), original)
                    self.assertEqual(deployment.load(campaign/'protocol.json'), dict(prepared, external_checksums={}))
                    checks = deployment.load(campaign/'checksums.json')
                    self.assertEqual(checks['protocol.json'], deployment.digest(campaign/'protocol.json'))
                    self.assertEqual(deployment.load(campaign/'prepared.json')['checksums_sha256'], deployment.digest(campaign/'checksums.json'))

    def test_scheduler_releases_only_its_task(self):
        from ml_fixed_campaign import release_remote_runtime
        with patch('ml_fixed_campaign.load', return_value={'url':'http://127.0.0.1:9'}), \
                patch('mindware.components.optimizers.runtime_rpc.call') as health, \
                patch('mindware.components.optimizers.tosfit_remote.RemoteToSFiTRuntime') as client:
            state = {'remote_runtime_config':'/config', 'session':'/task-a'}
            health.return_value = {'run_id':'other', 'session':'/task-b'}
            release_remote_runtime(state)
            client.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'different task'):
                release_remote_runtime(state, strict=True)
            health.return_value = {'run_id':'own', 'session':'/task-a'}
            release_remote_runtime(state, strict=True)
            client.return_value.detach.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
