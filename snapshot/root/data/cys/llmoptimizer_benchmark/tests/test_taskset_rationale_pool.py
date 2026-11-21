"""Disjoint multi-host assignments and validation before merging SFT rationales."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/benchmark/domains'))
from taskset_rationale_pool import checked_shard, pool_progress, validate_pool, worker_status_path
from taskset_rationale_validation import generation_details, rationale_text
from train_taskset_s1b_s2g import digest_arrays


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(i) for i in ids if i != 0)


class RationalePool(unittest.TestCase):
    def setUp(self):
        self.plan = dict(pending_indices=[1, 2], retained_examples=1, total_examples=3,
                         workers={'local': {'indices': [1]}, 'remote0': {'indices': [2]}})
        self.tokenizer = Tokenizer()
        self.events = [{'event': i} for i in range(3)]

    def record(self, i):
        text = 'Rationale: A smaller learning rate may help.'
        generation = generation_details(self.tokenizer, self.tokenizer.encode(text) + [0], 192, [0])
        return dict(event_index=i, event_sha256=digest_arrays(self.events[i]), rationale=rationale_text(text),
                    attempts=[text], generation_attempts=[generation], validation_version=2)

    def test_disjoint_assignments(self):
        validate_pool(self.plan, 3)
        self.plan['workers']['remote0']['indices'] = [1]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_pool(self.plan, 3)

    def test_missing_and_out_of_range_assignments(self):
        self.plan['workers']['remote0']['indices'] = []
        with self.assertRaisesRegex(ValueError, 'cover'):
            validate_pool(self.plan, 3)
        self.plan['workers']['remote0']['indices'] = [3]
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_pool(self.plan, 3)

    def test_shard_coverage_and_source_identity(self):
        row = self.record(2)
        self.assertEqual(set(checked_shard([row], [2], self.events, self.tokenizer, digest_arrays)), {2})
        with self.assertRaisesRegex(ValueError, 'missing or unexpected'):
            checked_shard([], [2], self.events, self.tokenizer, digest_arrays)
        row['event_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'different source'):
            checked_shard([row], [2], self.events, self.tokenizer, digest_arrays)

    def test_invalid_remote_response_blocks_merge(self):
        row = self.record(2)
        row['rationale'] = 'Silently changed.'
        with self.assertRaisesRegex(ValueError, 'Invalid rationale shard'):
            checked_shard([row], [2], self.events, self.tokenizer, digest_arrays)

    def test_progress_ignores_another_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            worker_status_path(campaign, 'local').write_text(json.dumps(dict(pool_sha256='current', status='completed', completed=1)))
            worker_status_path(campaign, 'remote0').write_text(json.dumps(dict(pool_sha256='old', status='completed', completed=1)))
            progress = pool_progress(campaign, self.plan, 'current')
            self.assertEqual(progress['completed'], 2)
            self.assertEqual(progress['workers']['remote0']['status'], 'pending')
            worker_status_path(campaign, 'remote0').write_text(json.dumps(dict(pool_sha256='current', status='completed', completed=1)))
            self.assertEqual(pool_progress(campaign, self.plan, 'current')['completed'], 3)


if __name__ == '__main__':
    unittest.main()
