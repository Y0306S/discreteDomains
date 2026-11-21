"""A cached acquisition batch must advance the real TuRBO failure state."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'native'))
from lolbo.lolbo import LOLBOState
from lolbo.utils.bo_utils.turbo import TurboState


class CachedBatchTests(unittest.TestCase):
    def state(self):
        state = LOLBOState.__new__(LOLBOState)
        state.train_x = [[1], [2]]
        state.train_z = torch.zeros(2, 64)
        state.train_y = torch.tensor([[-1.], [-10.]])
        state.censoring = torch.zeros(2, 1)
        state.progress_fails_since_last_e2e = 0
        state.tr_state = TurboState(dim=64, batch_size=10, best_value=-1.)
        state.initialize_tr_state = Mock()
        return state

    def batch(self, state, count=10, acquisition=True):
        state.update_next(torch.ones(count, 64), torch.full((count, 1), -10.),
                          [[2]] * count, torch.zeros(count, 1), acquisition=acquisition)

    def test_cached_batches_shrink_after_the_normal_failure_tolerance(self):
        state = self.state()
        for _ in range(32):
            self.batch(state)
        self.assertEqual(state.progress_fails_since_last_e2e, 32)
        self.assertEqual(state.tr_state.length, 0.4)
        self.assertEqual(state.tr_state.failure_counter, 0)
        self.assertEqual(len(state.train_z), 2)
        self.assertEqual(len(state.train_x), 2)

    def test_singleton_cached_batch_preserves_tensor_shape(self):
        state = self.state()
        self.batch(state, count=1)
        self.assertEqual(state.tr_state.failure_counter, 1)
        self.assertEqual(state.progress_fails_since_last_e2e, 1)

    def test_non_acquisition_cache_does_not_consume_progress(self):
        state = self.state()
        self.batch(state, acquisition=False)
        self.assertEqual(state.tr_state.failure_counter, 0)
        self.assertEqual(state.progress_fails_since_last_e2e, 0)

    def test_cached_batch_can_trigger_the_existing_restart_path(self):
        state = self.state()
        state.tr_state.length = state.tr_state.length_min
        state.tr_state.failure_counter = 31
        self.batch(state)
        self.assertTrue(state.tr_state.restart_triggered)
        state.initialize_tr_state.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
