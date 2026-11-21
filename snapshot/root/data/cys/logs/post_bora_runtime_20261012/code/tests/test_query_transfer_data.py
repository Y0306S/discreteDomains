"""Protect query split boundaries, feature scaling, and chronological coverage."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from examples.benchmark.domains import query_transfer_common as data


class QueryTransferTests(unittest.TestCase):
    def test_campaign_rejects_local_vae_and_modified_bao_input(self):
        root = Path('/unused/query_root')
        checkpoint = root / 'vae/CEB_64_author.ckpt'
        with patch.object(data, 'digest', side_effect=['old-local-vae']):
            with self.assertRaisesRegex(ValueError, 'author VAE'):
                data.validate_assets(root, checkpoint)
        with patch.object(data, 'digest', side_effect=[data.AUTHOR_VAE_SHA256, 'altered-bao']):
            with self.assertRaisesRegex(ValueError, 'official Bao'):
                data.validate_assets(root, checkpoint)
        with patch.object(data, 'digest', side_effect=[data.AUTHOR_VAE_SHA256, data.OFFICIAL_BAO_SHA256]):
            self.assertEqual(data.validate_assets(root, checkpoint), checkpoint)

    def test_windows_cover_every_observation_in_order(self):
        for count in (2, 31, 32, 33, 64, 65, 799):
            windows = list(data.window_indices(count))
            self.assertEqual(set(i for w in windows for i in w), set(range(count)))
            self.assertTrue(all(2 <= len(w) <= 32 and w == sorted(set(w)) for w in windows))
        self.assertEqual(list(data.window_indices(33))[-1], [31, 32])

    def test_singleton_source_cannot_form_a_window(self):
        with self.assertRaises(ValueError):
            list(data.window_indices(1))

    def test_scaling_round_trips_and_refuses_clipping(self):
        raw = np.tile(np.linspace(-4, 4, 64), (2, 1))
        lo, hi = np.full(64, -4.), np.full(64, 4.)
        x = data.scale_latents(raw, lo, hi)
        np.testing.assert_allclose(x * (hi - lo) + lo, raw)
        raw[0, 0] = -5
        with self.assertRaisesRegex(ValueError, 'no clipping'):
            data.scale_latents(raw, lo, hi)
        raw[0, 0] = np.nan
        with self.assertRaises(ValueError):
            data.scale_latents(raw, lo, hi)

    def fixture(self, directory, task='CEB_1A1', duplicate=False):
        raw = np.zeros((2, 64))
        x = np.full((2, 64), .5)
        np.savez(directory / 'features.npz', encoder_means=raw, X=x, offsets=[0, 2])
        plans = [[1, 2, 3, 4, 5], [2, 2, 3, 4, 5]]
        if duplicate:
            plans[1] = plans[0]
        records = [dict(task_id=task, plans=plans, source_rows=[0, 8], runtime_seconds=[1., 2.],
                        source_path=str(data.source_path(task)), source_sha256='unused')]
        data.atomic_json(directory / 'tasks.json', records)
        manifest = dict(domain='ceb_query_runtime', status='complete', policy=data.POLICY,
                        representation='frozen_vae_encoder_mean_64',
                        train_split_sha256='train', test_split_sha256='test', vae_sha256='vae',
                        source_tasks=[task], excluded_tasks=[], lower=[-4.] * 64,
                        upper=[4.] * 64, observations=2,
                        files={n: data.digest(directory / n) for n in ('tasks.json', 'features.npz')})
        data.atomic_json(directory / 'manifest.json', manifest)

    def test_loader_rejects_heldout_tasks_and_duplicate_plans(self):
        original_digest = data.digest

        def fake_digest(path):
            identities = {data.TRAIN_FILE: 'train', data.TEST_FILE: 'test', data.VAE_FILE: 'vae'}
            return identities[Path(path)] if Path(path) in identities else original_digest(path)

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with patch.object(data, 'read_split', return_value=(['CEB_1A1'], ['CEB_1A2'])), \
                    patch.object(data, 'digest', side_effect=fake_digest):
                self.fixture(directory)
                _, tasks = data.load_corpus(directory, check_sources=False)
                self.assertEqual(len(tasks['CEB_1A1']['X']), 2)
                manifest_path = directory / 'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                manifest['vae_sha256'] = 'old-local-vae'
                data.atomic_json(manifest_path, manifest)
                with self.assertRaisesRegex(ValueError, 'VAE identity mismatch'):
                    data.load_corpus(directory, check_sources=False)
                self.fixture(directory, task='CEB_1A2')
                with self.assertRaisesRegex(ValueError, 'permitted split'):
                    data.load_corpus(directory, check_sources=False)
                self.fixture(directory, duplicate=True)
                with self.assertRaisesRegex(ValueError, 'distinct completed'):
                    data.load_corpus(directory, check_sources=False)


if __name__ == '__main__':
    unittest.main()
