"""FSBO arm policy parity and durable pending-proposal recovery (offline)."""
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest

from mindware.domains.dlhpo.taskset_llm import loop as L, spec as S
from mindware.domains.dlhpo.taskset_llm.llm_client import ScriptedClient
from tests.domains.test_taskset_llm_rgpe import LoopTest, SPECS, TASK, sample


class FSBOPolicyTest(unittest.TestCase):
    def test_all_100_rounds_preserve_rgpe_policy_with_fsbo_arm(self):
        for seed in (1, 7, 42):
            with tempfile.TemporaryDirectory() as tmp:
                rgpe, cs = LoopTest().make(Path(tmp) / "rgpe", ScriptedClient([]), budget=100)
                fsbo = L.TaskSetLLMFSBO(task_id=TASK, cs=cs, evaluate=None, run_dir=Path(tmp) / "fsbo",
                                        llm=ScriptedClient([]), presets=[sample(cs,1000+i) for i in range(3)],
                                        specs=SPECS, bo_suggest=None, budget=100, seed=seed)
                rgpe.selector.rng.seed(seed)
                for trial in range(1,101):
                    r_arm, r_details = rgpe.choose_source(trial)
                    f_arm, f_details = fsbo.choose_source(trial)
                    self.assertEqual(f_arm, "fsbo" if r_arm == "rgpe" else r_arm)
                    self.assertEqual(r_details["cooldown_active"], f_details["cooldown_active"])
                    for loop, arm in ((rgpe,r_arm),(fsbo,f_arm)):
                        loop.observations.append({"extra_info":{"selected_arm":arm}})
                        if trial > 3:
                            credit = loop.bo_arm if trial % 5 == 0 else arm
                            loop.selector.update(credit, trial/100, higher_is_better=True,
                                                 beat_incumbent=trial%3==0,beat_arm_best=trial%2==0)
                            if credit != arm:
                                loop.selector.record_attempt(arm)

    def test_pending_proposal_restores_selector_draw_and_stale_pending_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            counter = iter(range(100))
            rgpe, cs = LoopTest().make(folder, ScriptedClient(lambda *_: json.dumps({"config":sample(cs,45000+next(counter))})), budget=100)
            for _ in range(8):
                rgpe.step()
            old_t = rgpe.selector.t
            pending = rgpe.propose(9)
            L._atomic_json(rgpe.pending_path, pending)
            expected_t = pickle.loads(bytes.fromhex(pending["selector_after_selection"])).t
            self.assertEqual(expected_t, old_t+1)
            resumed, _ = LoopTest().make(folder, ScriptedClient([]), budget=100)
            resumed.step()
            self.assertEqual(resumed.selector.t, expected_t)
            self.assertEqual(resumed.observations[-1]["config"],pending["config"])
            # Simulate a crash after commit but before pending removal; round 9
            # must not be evaluated again. Stop deliberately at round 10's oracle.
            L._atomic_json(resumed.pending_path,pending)
            class Stop(Exception): pass
            resumed.evaluate = lambda *_: (_ for _ in ()).throw(Stop())
            with self.assertRaises(Stop):
                resumed.step()
            self.assertEqual(json.loads(resumed.pending_path.read_text())["trial"],10)
            self.assertEqual(len(resumed.observations),9)


if __name__ == "__main__":
    unittest.main()
