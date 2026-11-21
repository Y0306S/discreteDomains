"""CPU budgets, shared RAM admission and the public remote ToSFiT workflow."""

import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/benchmark'))
from ml_memory_admission import forecast
from ml_fixed_campaign import command, save


class AllocationTests(unittest.TestCase):
    def test_pending_fits_reserve_memory_across_campaigns(self):
        pending = [{'reserved_gib':40, 'resident_gib':1}, {'reserved_gib':16, 'resident_gib':0}]
        self.assertFalse(forecast(160, pending, 16, 256, 32))
        self.assertTrue(forecast(100, pending, 16, 256, 32))

    def test_staged_code_and_three_core_policy_reach_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root/'protocol.json', {'baseline_python':'/python', 'code_root':'/staged/code',
                'method_args':{'tosfit':['--tosfit-device','cuda:0','--tosfit-reference-device','cuda:1']}})
            task = {'task_type':'cls','dataset':'fixture','algorithm':'adaboost','metric':'bal_acc'}
            args, env, _ = command(root, 'tosfit', task, [4,5,6], '3,2')
            self.assertIn('/staged/code/examples/benchmark/run_toy_categorical_optimizers.py', args)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '3,2')
            self.assertEqual(env['EVAL_N_JOBS'], '3')


class PublicRemoteTests(unittest.TestCase):
    def test_real_ml_evaluations_remote_gradients_and_completed_resume(self):
        import torch
        from sklearn.datasets import make_classification
        from mindware.components.feature_engineering.transformation_graph import DataNode
        from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs
        from mindware.components.utils.constants import CLASSIFICATION, NUMERICAL
        from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
        from mindware.components.optimizers.runtime_rpc import server
        from mindware.components.optimizers.tosfit_remote import DurableToSFiTActor
        from test_tosfit_runtime import RuntimeTests
        import run_toy_categorical_optimizers as runner
        torch.set_num_threads(1)
        cs = build_hpo_cs('cls', 'random_forest')
        def factory():
            runtime = RuntimeTests().make()
            runtime.recipe['n_marginal_likelihood_warmup_steps'] = 2
            runtime.tokenizer.decode = lambda *a, **kw: json.dumps(dict(cs.get_default_configuration()))
            return runtime
        x, y = make_classification(n_samples=80, n_features=6, random_state=3)
        node = DataNode([x,y], feature_type=[NUMERICAL]*6, task_type=CLASSIFICATION)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), \
                patch.object(KnowledgeDatabase, '_instance', None):
            root = Path(directory)
            actor = DurableToSFiTActor(root, 'fixture-runtime', factory)
            httpd = server(actor, 0)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                config = root/'client.json'
                save(config, {'url':'http://127.0.0.1:%d'%httpd.server_address[1],
                              'identity':'fixture-runtime','timeout_seconds':30})
                os.environ['TOSFIT_REMOTE_CONFIG'] = str(config)
                os.environ['EVAL_N_JOBS'] = '1'
                with patch.object(sys, 'argv', ['run','--optimizer','tosfit','--dataset','fixture',
                    '--algorithm','random_forest','--n-trials','5','--time-limit','none',
                    '--per-run-time-limit','60','--skip-test-eval','--tosfit-device','cpu',
                    '--tosfit-reference-device','cpu','--recommendation-fallback','default',
                    '--extractor-max-tokens','2048','--extractor-max-calls','100','--extractor-timeout','60',
                    '--output-dir',str(root/'outputs'),'--history-dir',str(root/'history'),
                    '--session-dir',str(root/'session')]):
                    args = runner.parse_args()
                result = runner.run_one_optimizer(args, 'tosfit', node, node, ['random_forest'], {})
                self.assertEqual(result['progress']['attempts'], 5)
                self.assertEqual(result['progress']['successful'], 5)
                with (root/'session/checkpoint.pkl').open('rb') as stream:
                    saved = pickle.load(stream)
                self.assertTrue(saved['extra']['remote'])
                self.assertEqual(saved['extra']['version'], 3)
                self.assertTrue(saved['extra']['weights_retired'])
                count = actor.connection.execute('SELECT COUNT(*) FROM receipts').fetchone()[0]
                args.resume = True
                resumed = runner.run_one_optimizer(args, 'tosfit', node, node, ['random_forest'], {})
                self.assertEqual(resumed['progress'], result['progress'])
                self.assertEqual(actor.connection.execute('SELECT COUNT(*) FROM receipts').fetchone()[0], count)
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join()
                actor._drop()


if __name__ == '__main__':
    unittest.main()
