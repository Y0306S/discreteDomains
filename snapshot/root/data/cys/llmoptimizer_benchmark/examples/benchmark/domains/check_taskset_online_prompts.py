#!/usr/bin/env python3
"""Real ORPO calls on archived target prefixes; no new oracle evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.dlhpo.taskset_llm.inference import SharedQwenClient
from mindware.domains.dlhpo.taskset_llm.loop import TaskSetLLMFSBO, rows_from_observations
from mindware.domains.dlhpo.taskset_llm import p5_prompt as P, spec as S
from taskset_transfer_common import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--service-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    service = args.service_dir
    client = SharedQwenClient("http://127.0.0.1:18765", service / "inference.token",
                              expected_identity=json.loads((service / "inference_identity.json").read_text())["identity_sha256"])
    task = "quadratic_family_seed0"
    path = Path("/root/data/logs/taskset_major_families_cpu_20260929/test") / task / "tpe/seed1.tar.gz"
    with tarfile.open(path) as archive:
        raw = archive.extractfile("run/history.json").read()
        manifest = json.load(archive.extractfile("manifest.json"))
    entry = next(item for item in manifest["entries"] if item["name"] == "run/history.json")
    assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    history = json.loads(raw)
    cs, specs = TaskSetAdapter(dummy=True).search_space(), S.load_task_specs()
    presets = json.loads((service / "bolt_ws3_online.json").read_text())[task]
    results = []
    for trial, branch in [(50, "explore"), (52, "exploit")]:
        loop = TaskSetLLMFSBO(task_id=task, cs=cs, evaluate=None, run_dir=args.out / str(trial),
                             llm=client, presets=presets, specs=specs, bo_suggest=None, budget=100)
        loop.observations = history["observations"][:trial-1]
        assert P.explore_branch(1, trial) == branch
        log = {}
        started = time.monotonic()
        config, info = loop._llm_turn(trial, log)
        assert log["prompt_meta"]["experience_branch"] == branch
        assert config is not None and info["configuration_source"] == "llm", info
        text = log["user_prompt"]
        if branch == "explore":
            assert "Exploration counts per optimizer family" in text and "Good/Bad regions" not in text
            rows = rows_from_observations(loop.observations)
            for family in S.families(cs):
                count = sum(row["config"]["optimizer_family"] == family for row in rows)
                assert f"{family}: {count} evaluations" in text
        else:
            assert "Good/Bad regions" in text and "Exploration counts" not in text
            assert "good [" in text or "good " in text
        atomic_json(args.out / f"trial{trial}_trace.json", log)
        result = {"test": f"Real ORPO {branch} prompt", "status": "PASS", "trial": trial,
                  "prefix": "archived TPE target observations (shadow proposal only)",
                  "observations": trial-1, "attempts": len(log["attempts"]),
                  "new_oracle_evaluations": 0, "seconds": time.monotonic()-started,
                  "parsed_config": config}
        results.append(result)
        atomic_json(args.out / "checks.json", results)
        print(json.dumps(result), flush=True)

    # Inject one known duplicate, then ask the REAL model to respond to its
    # appended reminder. This is fault injection, not a naturally observed repeat.
    rows = rows_from_observations(history["observations"][:30])
    _, _, meta = P.build_p5_prompt(task, specs, cs, rows, trial=31, seed=1)
    old = next(row for row in rows if row["trial"] not in meta["selected_trials"])
    class InjectDuplicate:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user):
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"config": old["config"]})
            return client.complete(system, user)

    injected = InjectDuplicate()
    loop = TaskSetLLMFSBO(task_id=task, cs=cs, evaluate=None, run_dir=args.out / "duplicate_retry",
                         llm=injected, presets=presets, specs=specs, bo_suggest=None, budget=100)
    loop.observations = history["observations"][:30]
    trace = {}
    config, info = loop._llm_turn(31, trace)
    assert config is not None and info["configuration_source"] == "llm"
    assert trace["attempts"][0]["duplicate"] and len(trace["attempts"]) >= 2
    retry = trace["attempts"][1]["user_prompt"]
    assert retry.startswith(trace["user_prompt"]) and "Retry reminder" in retry
    assert f"trial={old['trial']}" in retry[len(trace["user_prompt"]):]
    assert len([v for v in trace["prompt_meta"]["selected_tags"].values() if "recent" in v]) == 15
    atomic_json(args.out / "duplicate_retry_trace.json", trace)
    result = {"test": "Real ORPO retry after injected old duplicate", "status": "PASS",
              "injected_duplicate_trial": old["trial"], "attempts": len(trace["attempts"]),
              "real_model_calls": len(trace["attempts"])-1, "Recent15_preserved": True,
              "new_oracle_evaluations": 0}
    results.append(result)
    atomic_json(args.out / "checks.json", results)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
