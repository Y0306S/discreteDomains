import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'batch_code'))
import resources


class CapacityHealthTests(unittest.TestCase):
    def test_busy_vae_health_has_bounded_decoder_grace(self):
        with patch.object(resources.urllib.request,'urlopen') as request:
            request.return_value.__enter__.return_value.read.return_value=b'{"status":"ready"}'
            self.assertEqual(resources.json_get('http://127.0.0.1:21834/health'),dict(status='ready'))
            request.assert_called_once_with('http://127.0.0.1:21834/health',timeout=90)

    def check_pool(self, count, reported=None, unavailable=False):
        cfg = dict(llm=[dict(name=f'llm{i}', port=21800+i) for i in range(count)],
                   vae=[dict(name=f'vae{i}', port=21820+i) for i in range(16)])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'services').mkdir()
            (root / 'services/active_inference_stage.json').write_text(json.dumps(
                dict(role='orpo', stage='orpo_batch10')))
            for name in [w['name']+'_resources' for w in cfg['llm']+cfg['vae']]+['orpo_resources_broker','resources_tunnel']:
                (root / 'services' / (name+'.process.json')).write_text('{}')
            def response(url):
                if url.endswith(':21811/health'):
                    replicas=[dict(ready=not unavailable, placement=dict(embedding_device='cuda:0', language_devices=['cuda:0'])) for _ in cfg['llm']]
                    return dict(ready=True, replicas=count if reported is None else reported, replica_health=replicas)
                return dict(status='ready', checkpoint_sha256='83a4942701bffaaf8311887e530e3157b6843134e52e9f53d59f957c24aebab1')
            with patch.object(resources, 'ROOT', root), patch.object(resources, 'config', return_value=cfg), patch.object(resources, 'record_alive', return_value=True), patch.object(resources, 'json_get', side_effect=response):
                return resources.health()

    def test_four_and_eight_llm_pools(self):
        for count in [4,8]:
            result=self.check_pool(count)
            self.assertEqual(len(result['vae']),16)
            self.assertEqual(result['llm'][0]['replicas'],count)
            self.assertEqual(result['service_processes_alive'],count+17)

    def test_wrong_broker_count_rejected(self):
        with self.assertRaisesRegex(ValueError,'incorrect count'):
            self.check_pool(8,reported=2)

    def test_partial_llm_outage_rejected(self):
        with self.assertRaisesRegex(ValueError,'unavailable'):
            self.check_pool(8,unavailable=True)

if __name__ == '__main__':
    unittest.main()
