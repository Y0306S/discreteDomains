import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/benchmark"))

import generate_ckpt32_warm_starts as generator
from mindware.components.experience.surrogate_landscape import get_algo_cs


class WarmStartGenerationTest(unittest.TestCase):
    def test_three_duplicate_proposals_are_accepted_without_extra_generations(self):
        cs = get_algo_cs("adaboost", "regression")
        raw = {"adaboost:" + key: value for key, value in cs.get_default_configuration().get_dictionary().items()}
        raw["algorithm"] = "adaboost"
        requests = []

        class Client:
            adapter_verification = {"verified_tensors": 496, "sha256": "fixture"}

            def __init__(self, *args, **kwargs):
                self.temperature = kwargs["temperature"]
                self_outer.assertEqual(self.temperature, 0.9)

            def _ensure_loaded(self):
                pass

            def complete(self, system, user):
                requests.append((system, user))
                return json.dumps({"config": raw})

        self_outer = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "warm_start_presets.json").write_text(json.dumps({"target:adaboost": [raw] * 3}))
            (root / "tasks_132.json").write_text(json.dumps([
                {"dataset": "target", "algorithm": "adaboost", "task_type": "rgs"}]))
            (root / "protocol.json").write_text(json.dumps({"base_model": "fixture"}))
            (root / "prompts.jsonl").write_text(json.dumps({
                "metadata": {"dataset": "target", "algorithm": "adaboost"},
                "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]}))
            argv = ["generate", "--campaign", str(root), "--output", str(root / "generated"),
                    "--task-prompts", str(root / "prompts.jsonl"), "--temperature", "0.9"]
            with patch.object(sys, "argv", argv), patch.object(generator, "HuggingFaceQwenClient", Client):
                generator.main()
            audit = json.loads((root / "generated/audit.json").read_text())
            self.assertEqual(len(requests), 3)
            self.assertEqual(requests, [("system", "task")] * 3)
            self.assertTrue(all(row["accepted"] for row in audit["proposals"]))
            self.assertEqual([row["duplicate_of_earlier_slot"] for row in audit["proposals"]], [False, True, True])
            self.assertEqual(audit["tasks"][0]["distinct_configurations"], 1)
            self.assertTrue(audit["complete"])


if __name__ == "__main__":
    unittest.main()
