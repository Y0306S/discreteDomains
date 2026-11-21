import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'batch_code'))
from qwen_inference_server import Replica


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, prompt):
        return range(int(prompt))


def hold(path, tokens, entered, release):
    replica = Replica.__new__(Replica)
    replica.config = dict(max_input_tokens=32768,
        gpu_admission=dict(lock_file=path, shared_max_input_tokens=16384))
    client = SimpleNamespace(_tokenizer=Tokenizer(), enable_thinking=False)
    body = dict(use_s1b_template=False, messages=[dict(content=str(tokens))])
    with replica.gpu_admission(body, client):
        entered.set()
        release.wait(10)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = str(Path(self.folder.name) / 'gpu.lock')
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join(3)
        self.folder.cleanup()

    def start(self, tokens):
        entered, release = mp.Event(), mp.Event()
        process = mp.Process(target=hold, args=(self.path, tokens, entered, release))
        process.start()
        self.processes.append(process)
        return process, entered, release

    def test_three_short_requests_overlap(self):
        runs = [self.start(16384) for _ in range(3)]
        self.assertTrue(all(entered.wait(5) for _, entered, _ in runs))

    def test_long_waits_for_short_and_blocks_other_short(self):
        first, short_entered, release_short = self.start(8000)
        self.assertTrue(short_entered.wait(5))
        long, long_entered, release_long = self.start(32768)
        self.assertFalse(long_entered.wait(.2))
        release_short.set()
        self.assertTrue(long_entered.wait(5))
        _, next_entered, _ = self.start(8000)
        self.assertFalse(next_entered.wait(.2))
        release_long.set()
        self.assertTrue(next_entered.wait(5))

    def test_process_exit_releases_gpu(self):
        process, entered, _ = self.start(32768)
        self.assertTrue(entered.wait(5))
        _, next_entered, _ = self.start(8000)
        self.assertFalse(next_entered.wait(.2))
        process.terminate()
        process.join(3)
        self.assertTrue(next_entered.wait(5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
