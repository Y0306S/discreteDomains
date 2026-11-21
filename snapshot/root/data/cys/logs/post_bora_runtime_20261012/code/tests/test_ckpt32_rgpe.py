"""Scheduling, proposal ownership and durable recovery for the RGPE hybrid."""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mindware.components.optimizers import ml_checkpoint
from mindware.components.optimizers.pt4opt_llm_rgpe_optimizer import PT4OptLLMRGPEOptimizer
from tests import test_ckpt32_retry5 as retry_tests


class ToyRGPE(PT4OptLLMRGPEOptimizer):
    def update_saver(self, configs, objectives):
        pass

    def _new_rgpe_bridge(self):
        def suggest(observed):
            bridge.last_observed = observed
            cfg = self._sample_nonduplicate_config()
            return cfg, "RGPE test proposal"
        bridge = SimpleNamespace(asset_identity={"cache": "test"}, schema_fingerprint="test",
                                 _proposal_diagnostics="test-decision", _suggest_model=suggest,
                                 _key=lambda cfg: json.dumps(dict(cfg), sort_keys=True))
        return bridge


class RGPEPolicyTest(unittest.TestCase):
    response = staticmethod(retry_tests.RetryPolicyTest.response)
    add_history = retry_tests.RetryPolicyTest.add_history

    def optimizer(self):
        with patch("tests.test_ckpt32_retry5.ToyOptimizer", ToyRGPE):
            return retry_tests.RetryPolicyTest.optimizer(self, bo_arm="rgpe", task_type="classification",
                                              llm_phase_trials=8, pucb_p1=0.7, pucb_p_end=None)

    def test_forced_phase_and_fallback_accounting(self):
        opt = self.optimizer()
        for _ in range(8):
            opt.iterate()
        self.assertIsNone(opt.study)
        self.assertEqual([r["selected_arm"] for r in opt.traj_rows], ["llm_warm_start"] * 3 + ["llm"] * 5)
        self.assertTrue(all(r["stage"] == "rgpe" for r in opt.traj_rows[3:]))
        self.assertEqual(opt.selector.n, {"llm": 0, "rgpe": 5})
        self.assertEqual(opt.selector.attempt_n, {"llm": 5, "rgpe": 5})
        self.assertEqual(opt.objective_function.call_count, 8)
        self.assertTrue(all(set(r) == set(opt.traj_rows[0]) for r in opt.traj_rows))
        summary = opt.get_opt_trajectory()
        self.assertEqual((summary["n_evals"], summary["n_successful"], summary["rgpe_trials"]), (8, 8, 5))

    def test_cooldown_boundaries_prior_endpoints_and_budget(self):
        opt = self.optimizer()
        opt.selector.choose = lambda *args, **kwargs: "llm"
        for _ in range(101):
            opt.iterate()
        rows = opt.traj_rows
        self.assertEqual(len(rows), 100)
        self.assertEqual(opt.objective_function.call_count, 100)
        self.assertEqual([r["selected_arm"] for r in rows[8:20]], ["llm", "rgpe"] * 6)
        self.assertTrue(all(r["selected_arm"] == "llm" for r in rows[20:]))
        self.assertFalse(rows[8]["cooldown_active"])
        self.assertTrue(rows[19]["cooldown_active"])
        self.assertFalse(rows[20]["cooldown_active"])
        self.assertEqual((rows[8]["schedule_index"], rows[-1]["schedule_index"]), (0, 91))
        self.assertAlmostEqual(opt.selector.omega("llm", 0, 91), 0.7)
        self.assertAlmostEqual(opt.selector.omega("llm", 91, 91), 0.3)
        self.assertAlmostEqual(opt.selector.omega("rgpe", 91, 91), 0.7)

    def test_successful_observations_include_warm_duplicates_failures_are_excluded(self):
        opt = self.optimizer()
        self.add_history(opt, [1, 1, 2, 3], [0.2, 0.3, -math.inf, 0.4])
        bridge = opt._new_rgpe_bridge()
        with patch.object(opt, "_new_rgpe_bridge", return_value=bridge):
            config, _, _, _ = opt._suggest_bo_config()
        self.assertTrue(hasattr(config, "extra_info"))
        self.assertEqual([p for _, p in bridge.last_observed], [0.2, 0.3, 0.4])
        self.assertEqual(len(bridge.attempted_configs), 3)

    def test_pending_rgpe_fallback_resumes_once_then_applies_cooldown(self):
        opt = self.optimizer()
        for _ in range(8):
            opt.iterate()
        directory = Path(opt.output_dir) / "session"
        with patch.dict(os.environ, {"ML_RUN_ID": "rgpe-test"}):
            opt._ml_session = ml_checkpoint.MLSession(opt, directory)
            opt.objective_function.side_effect = RuntimeError("interrupted")
            opt.selector.n["rgpe"] = 500
            opt.selector.attempt_n["rgpe"] = 500
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                opt.iterate()
            self.assertEqual(opt._ml_session.pending["meta"]["selected_arm"], "llm")
            pending_cfg = opt._ml_session.pending["cfg"]
            counts = dict(opt.selector.attempt_n)
            resumed = self.optimizer()
            resumed._ml_session = ml_checkpoint.MLSession(resumed, directory)
            self.assertEqual(resumed._ml_session.restore(), 8)
            resumed.iterate()
            resumed.llm_client.complete.assert_not_called()
            self.assertEqual(resumed.objective_function.call_args.args[0], pending_cfg)
            self.assertEqual(resumed.selector.attempt_n, {k: v + 1 for k, v in counts.items()})
            self.assertEqual(resumed._choose_source(9)[0], "rgpe")
            self.assertTrue(resumed._selection_details["cooldown_active"])


if __name__ == "__main__":
    unittest.main()
