"""Validate immutable campaign inputs and a completed real preflight."""

import argparse
from pathlib import Path
import time

import ml_fixed_campaign as campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    protocol = campaign.load(root / "protocol.json")
    checks = campaign.load(root / "code_sha256.json")
    for name, expected in checks.items():
        if campaign.digest(root / "code" / name) != expected:
            raise RuntimeError("Snapshot changed: " + name)
    for row in campaign.load(root / "data_manifest.json").values():
        if campaign.digest(root / "code" / row["snapshot"]) != row["sha256"]:
            raise RuntimeError("Dataset changed: " + row["snapshot"])
    for name, field in {"tasks_132.json": "manifest_sha256", "runtime_policy.json": "policy_sha256",
                        "experience_bank.jsonl": "bank_sha256", "warm_start_presets.json": "warm_start_sha256",
                        "source_gp_train80_v1/manifest.json": "source_gp_manifest_sha256"}.items():
        if campaign.digest(root / name) != protocol[field]:
            raise RuntimeError("Input changed: " + name)
    assert campaign.digest(Path(protocol["adapter"]) / "adapter_model.safetensors") == protocol["adapter_sha256"]
    preflight = campaign.load(root / "preflight/passed.json")
    assert preflight["passed"] and preflight["successful"] == 12
    tasks = campaign.load(root / "queues/ckpt32.json")
    assert len(tasks) == len({campaign.task_key(t) for t in tasks}) == 132
    spec = protocol["hosts"]["LLMOptimizer2"]
    physical = []
    for cpu in spec["cpus"]:
        topology = Path("/sys/devices/system/cpu/cpu%d/topology" % cpu)
        physical.append(((topology / "physical_package_id").read_text().strip(),
                         (topology / "core_id").read_text().strip()))
    assert len(physical) == len(set(physical))
    report = {"passed": True, "time": time.time(), "code_files": len(checks), "tasks": 132,
              "dataset_hashes_verified": True, "physical_worker_cores": spec["cpus"],
              "preflight": preflight["evidence_directory"]}
    campaign.save(root / "launch_verification.json", report)
    print(report)


if __name__ == "__main__":
    main()
