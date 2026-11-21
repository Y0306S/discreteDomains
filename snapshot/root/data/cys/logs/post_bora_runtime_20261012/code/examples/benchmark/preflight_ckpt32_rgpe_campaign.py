"""Exercise the real adapter, evaluator, RGPE worker and cooldown before launch."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import ml_fixed_campaign as campaign

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--resume-evidence", type=Path)
    args = parser.parse_args()
    root = args.resume_evidence or args.campaign / "preflight" / str(time.time_ns())
    root.mkdir(parents=True, exist_ok=bool(args.resume_evidence))
    protocol = campaign.load(args.campaign / "protocol.json")
    for name, expected in protocol["rgpe_asset_hashes"].items():
        if campaign.digest(Path(protocol["rgpe_assets"]) / name) != expected:
            raise RuntimeError("RGPE artifact changed: " + name)
    os.environ["ML_TRANSFER_HISTORY_MIRROR"] = str(Path(protocol["rgpe_assets"]) / "history")
    os.environ["ML_TRANSFER_TRAIN_MANIFEST"] = str(Path(protocol["rgpe_assets"]) / "train_manifest.jsonl")
    from mindware.components.optimizers.transfer_hpo.protocol import validate_dump
    sources = set()
    for path in (Path(protocol["rgpe_assets"]) / "dumps").glob("*.json"):
        sources.update(validate_dump(campaign.load(path)))
    test = {campaign.task_key(t) for t in campaign.load(args.campaign / "tasks_132.json")}
    assert not sources & test
    policy = campaign.load(args.campaign / "runtime_policy.json")
    policy.update(policy_id="ckpt32_rgpe_preflight_12", attempts_per_search=12)
    policy.pop("resume_from_policy_sha256", None)
    policy.pop("resume_from_policy_id", None)
    campaign.save(root / "policy.json", policy)
    task = {"task_type": "cls", "dataset": "car", "algorithm": "extra_trees", "metric": "bal_acc"}
    command, env, _ = campaign.command(args.campaign, "ckpt32", task, [16, 17, 18], 0)
    replacements = {"--runtime-policy": root / "policy.json", "--session-dir": root / "session",
                    "--output-dir": root / "outputs", "--history-dir": root / "history"}
    for flag, value in replacements.items():
        command[command.index(flag) + 1] = str(value)
    campaign.save(root / "command.json", command)
    with (root / "run.log").open("a") as stream:
        result = subprocess.run(command, cwd=args.campaign / "code", env=env,
                                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Preflight failed: " + str(root / "run.log"))
    progress = campaign.load(root / "session/progress.json")
    assert progress["clean_complete"] and progress["successful"] == 12
    with Path(progress["trajectory"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 and all(None not in row for row in rows)
    assert [r["selected_arm"] for r in rows[:8]] == ["llm_warm_start"] * 3 + ["llm"] * 5
    assert any(r["stage"] == "rgpe" for r in rows)
    for i in range(9, 12):
        if rows[i - 1]["selected_arm"] == "llm":
            assert rows[i]["selected_arm"] == "rgpe"
    decisions = list((root / "session/decisions").glob("trial-*/attempt-*/worker.json"))
    assert decisions
    for path in decisions:
        meta = campaign.load(path)
        assert meta["status"] == "succeeded"
        request, response = campaign.load(path.parent / "request.json"), campaign.load(path.parent / "response.json")
        assert meta["selected_config"] == request["pending_configs"][response["index"]]
        assert len(response["acquisition"]["values"]) == 256
        assert len(response["source_weights"]) == response["n_sources"]
    report = {"passed": True, "time": time.time(), "evaluations": 12, "successful": 12,
              "training_sources": len(sources), "no_test_overlap": True,
              "rgpe_decisions": len(decisions), "excluded_from_campaign": True,
              "adapter_sha256": protocol["adapter_sha256"], "progress": progress}
    campaign.save(root / "passed.json", report)
    campaign.save(root.parent / "passed.json", dict(report, evidence_directory=str(root)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
