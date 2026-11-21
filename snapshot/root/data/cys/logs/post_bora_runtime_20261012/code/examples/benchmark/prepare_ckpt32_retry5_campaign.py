"""Prepare a fresh ckpt32 campaign from verified immutable ML inputs."""

import argparse
import csv
from pathlib import Path
import shutil
import socket
import time

import ml_fixed_campaign as campaign


def prepare(previous, target):
    protocol = campaign.load(previous / "protocol.json")
    expected = {"tasks_132.json": "manifest_sha256", "runtime_policy.json": "policy_sha256",
                "experience_bank.jsonl": "bank_sha256", "warm_start_presets.json": "warm_start_sha256",
                "source_gp_train80_v1/manifest.json": "source_gp_manifest_sha256"}
    for path, field in expected.items():
        if campaign.digest(previous / path) != protocol[field]:
            raise RuntimeError(f"Input hash mismatch: {path}")
    if campaign.digest(Path(protocol["adapter"]) / "adapter_model.safetensors") != protocol["adapter_sha256"]:
        raise RuntimeError("Adapter hash mismatch")
    tasks = campaign.load(previous / "tasks_132.json")
    if len(tasks) != len({campaign.task_key(t) for t in tasks}) or len(tasks) != 132:
        raise RuntimeError("Expected exactly 132 tasks")
    target.mkdir(parents=True, exist_ok=False)
    for name in ("mindware", "examples", "configs"):
        shutil.copytree(campaign.SOURCE / name, target / "code" / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "code/datasets_bundle").symlink_to((previous / "code/datasets_bundle").resolve(), target_is_directory=True)
    for name in ("tasks_132.json", "runtime_policy.json", "experience_bank.jsonl", "warm_start_presets.json", "data_manifest.json"):
        shutil.copy2(previous / name, target / name)
    shutil.copytree(previous / "source_gp_train80_v1", target / "source_gp_train80_v1")
    durations = {}
    csv.field_size_limit(16 * 1024 * 1024)
    for item in campaign.load(previous / "task_index.json")["tasks"]:
        if item["method"] == "ckpt32" and item.get("trajectory"):
            with Path(item["trajectory"]).open(newline="") as stream:
                durations[item["task_id"]] = sum(float(r.get("eval_time") or 0) for r in csv.DictReader(stream))
    tasks.sort(key=lambda t: (-durations.get(campaign.task_key(t), 0), campaign.task_key(t)))
    for method in campaign.inventory.METHODS:
        campaign.save(target / "queues" / f"{method}.json", tasks if method == "ckpt32" else [])
        campaign.save(target / "reused" / f"{method}.json", [])
    spec = {"hostname": socket.gethostname(), "cpus": list(range(32)), "cpu_quota": 40,
            "memory_gib": 400, "limits": {"ckpt32": 8}, "gpu_slots": [0, 0, 1, 1, 2, 2, 3, 3],
            "gpu_methods": ["ckpt32"]}
    protocol = {key: protocol[key] for key in (
        "manifest_sha256", "policy_sha256", "bank_sha256", "warm_start_sha256",
        "source_gp_manifest_sha256", "adapter_sha256", "adapter", "base_model", "ckpt_python", "baseline_python")}
    protocol.update(campaign=target.name, created_at=time.time(), parent_campaign=str(previous),
                    hosts={"LLMOptimizer2": spec}, methods=["ckpt32"], task_count=132, seed=1, ckpt32_policy="rerun-all",
                    ckpt32_revision="ckpt32_retry5_p05_20260929", ckpt32_hybrid_variant="HPUCB_arm_p05_retry5",
                    ckpt32_response_format="json_only", reuse="None; all 132 tasks start fresh under the new policy",
                    counts={method: {"fresh": 132 if method == "ckpt32" else 0, "reused_clean": 0}
                            for method in campaign.inventory.METHODS},
                    ckpt32_duplicate_policy="Five LLM calls, recent 10 through 14 with saved results; then TPE with TPE-only credit",
                    automatic_process_retries=0,
                    cpu_policy="Four cores per worker; one for serial estimators. RGPE retains cores 36-39.",
                    gpu_policy="Two workers per GPU on 0-3; existing OptFormer training on GPU 2 remains running",
                    task_order="Descending evaluation time in previous verified ckpt32 run")
    campaign.save(target / "protocol.json", protocol)
    campaign.save(target / "selected_legacy.json", [])
    campaign.save(target / "code_sha256.json", {str(p.relative_to(target / "code")): campaign.digest(p)
                  for p in (target / "code").rglob("*") if p.is_file() and "datasets_bundle" not in p.parts})
    campaign.status(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous, args.campaign)
