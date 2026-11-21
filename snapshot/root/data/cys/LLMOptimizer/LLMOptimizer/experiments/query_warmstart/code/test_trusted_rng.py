import hashlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from trusted_rng import restore_rng


class TrustedRngTests(unittest.TestCase):
    def test_exact_random_stream_continuation_and_changed_file_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            random.seed(57); np.random.seed(58); torch.manual_seed(59)
            state = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state())
            path = checkpoint/'rng_state.pth'; torch.save(state, path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = (random.random(), np.random.rand(), torch.rand(4))
            restore_rng(checkpoint, checkpoint, checksum)
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.rand(), expected[1])
            torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'Unexpected RNG checkpoint'):
                restore_rng(checkpoint/'other', checkpoint, checksum)
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum changed'):
                restore_rng(checkpoint, checkpoint, checksum)


if __name__ == '__main__':
    unittest.main()
