#!/usr/bin/env python3
"""Bounded preflight of real FSBO min-max plus scripted handoff edge cases."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.dlhpo.taskset_llm.fsbo_bridge import FSBOBridge
from mindware.domains.dlhpo.taskset_llm import p5_prompt as P, spec as S
from mindware.domains.dlhpo.taskset_llm.loop import TaskSetLLMFSBO, _atomic_json
from mindware.domains.dlhpo.taskset_llm.llm_client import ScriptedClient
from mindware.domains.history import observation
from taskset_transfer_common import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []

    def record(name, evidence, started):
        results.append({"test": name, "status": "PASS", "seconds": time.monotonic() - started, "evidence": evidence})
        atomic_json(args.out / "checks.json", results)
        print(json.dumps(results[-1]), flush=True)

    cs = TaskSetAdapter(dummy=True).search_space()
    def sample(seed):
        cs.seed(seed)
        return dict(cs.sample_configuration().get_dictionary())

    presets = [sample(8000+i) for i in range(3)]
    observations = [observation(config, .9-.2*i, trial_state=0, elapsed_time=0)
                    for i, config in enumerate(presets)]
    bridge = FSBOBridge(cs, args.assets, args.out / "surrogate")
    try:
        started = time.monotonic()
        cfg, diagnostic = bridge(observations, 9)
        record("Real FSBO checkpoint / encoding / min-max / 256 EI candidates",
               {key: diagnostic[key] for key in ("checkpoint_sha256", "source_observation_indices", "candidate_count", "target_scaling", "service_pid")}, started)
        request = dict(schema_fingerprint=bridge.meta["schema_fingerprint"],
                       X_obs=[bridge.codec.encode(c).tolist() for c in presets], y_obs=[-.9,-.7,-.5],
                       X_pen=[bridge.codec.encode(c).tolist() for c in diagnostic["candidates"]],
                       seed=1, finetune_epochs=100, target_scaling="minmax")
        atomic_json(args.out / "parity_request.json", request)
        started = time.monotonic()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="/root/data/cys/baseline/repos/FSBO",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (args.out / "reference.log").open("w") as log:
            subprocess.run(["/root/data/cys/baseline/envs/bl_fsbo_exact/bin/python",
                            "/root/data/logs/taskset_fsbo_minmax_test210_20261006/code/examples/benchmark/fsbo_worker.py",
                            "suggest", "--ckpt", str(args.assets), "--request", str(args.out / "parity_request.json"),
                            "--out", str(args.out / "reference.json")], env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=300, check=True)
        reference = json.loads((args.out / "reference.json").read_text())
        import numpy as np
        errors = {}
        for key in ("posterior_mean", "posterior_stddev"):
            errors[key] = float(np.max(np.abs(np.asarray(reference[key])-diagnostic[key])))
            assert np.allclose(reference[key], diagnostic[key], rtol=1e-6, atol=1e-7), (key, errors[key])
        assert np.allclose(reference["acquisition"]["values"], diagnostic["acquisition"]["values"], rtol=1e-6, atol=1e-7)
        assert reference["index"] == diagnostic["index"]
        record("Persistent FSBO agrees with original frozen worker", {"max_abs_errors": errors, "same_candidate_index": True}, started)
        started = time.monotonic()
        modified = dict(request, y_obs=[.1,.2,.8])
        bridge.worker.request(modified)
        again = bridge.worker.request(request)
        assert again["service_pid"] == diagnostic["service_pid"]
        assert np.array_equal(again["posterior_mean"], diagnostic["posterior_mean"])
        assert again["index"] == diagnostic["index"]
        record("A/B/A task isolation and checkpoint reset", {"same_pid": True, "A_repeated_bit_for_bit": True,
               "warm_request_seconds": again["request_seconds"], "cold_request_seconds": diagnostic["request_seconds"]}, started)

        specs = S.load_task_specs()
        task = "quadratic_family_seed0"
        dummy = TaskSetAdapter(task={"task_id": task}, dummy=True)
        duplicate = json.dumps({"config": presets[0]})
        llm = ScriptedClient(lambda *_: duplicate)
        loop = TaskSetLLMFSBO(task_id=task, cs=cs, evaluate=dummy.evaluate_config,
                             run_dir=args.out / "forced_retry", llm=llm, presets=presets,
                             specs=specs, bo_suggest=bridge, budget=100)
        started = time.monotonic()
        for _ in range(4):
            loop.step()
        info = loop.observations[-1]["extra_info"]
        assert info["selected_arm"] == "llm" and info["source"] == "fsbo"
        assert len(llm.calls) == 5 and len(loop.observations) == 4
        assert all("Retry reminder" in call["user"] for call in llm.calls[1:])
        assert loop.selector.attempt_counts()["llm"] == 1 and loop.selector.n["fsbo"] == 1
        record("Five duplicates hand off to real FSBO with correct arm credit", {"calls": 5, "evaluations": 4,
               "selected_arm": info["selected_arm"], "credited_source": info["source"]}, started)
        started = time.monotonic()
        rows = [{"trial": i+1, "config": sample(9000+i), "loss": i/30, "successful": True} for i in range(30)]
        old = rows[10]
        _, original, meta = P.build_p5_prompt(task, specs, cs, rows, trial=31, seed=1)
        assert old["trial"] not in meta["selected_trials"]
        expanded = P.retry_prompt(original, [P.retry_reminder(rows, old["config"])])
        assert expanded.startswith(original)
        assert S.display_config(old["config"]) in expanded[len(original):]
        assert "trial=11" in expanded[len(original):]
        assert len([v for v in meta["selected_tags"].values() if "recent" in v]) == 15
        record("Old duplicate is appended alongside unchanged Recent15", {"original_recent_count": 15, "appended_old_trial": 11,
               "original_prompt_preserved": True}, started)
        started = time.monotonic()
        # Force the next proposal to be FSBO using a controlled selector choice.
        # Trial 10 follows an LLM-selected turn and is forced FSBO by cooldown.
        for i in range(4, 9):
            config = sample(10000+i)
            loop.observations.append(observation(config,.5+i/100,trial_state=0,elapsed_time=0,
                                      extra={"source":"llm","selected_arm":"llm"}))
        loop._commit()
        pending = loop.propose(10)
        _atomic_json(loop.pending_path, pending)
        selected_state = pending["selector_after_selection"]
        fresh_llm = ScriptedClient(lambda *_: (_ for _ in ()).throw(AssertionError("No new LLM call expected")))
        resumed = TaskSetLLMFSBO(task_id=task, cs=cs, evaluate=dummy.evaluate_config,
                                run_dir=loop.run_dir,llm=fresh_llm,presets=presets,specs=specs,bo_suggest=bridge,budget=100)
        resumed.step()
        assert resumed.observations[-1]["config"] == pending["config"]
        import pickle
        expected_selector = pickle.loads(bytes.fromhex(selected_state))
        assert resumed.selector.t == expected_selector.t
        record("Pending FSBO handoff resumes without another proposal", {"trial":10,"same_config":True,
               "new_llm_calls":0,"selector_selection_count_preserved":True}, started)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
