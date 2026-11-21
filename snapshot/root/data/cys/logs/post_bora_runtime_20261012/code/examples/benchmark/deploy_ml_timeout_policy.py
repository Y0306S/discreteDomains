"""Apply the approved 2400-second policy to a stopped, resumable ML campaign."""

import argparse
import ast
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from zoneinfo import ZoneInfo

import ml_fixed_campaign as campaign_tools


FILES = (
    "mindware/components/optimizers/ml_evaluation_feedback.py",
    "mindware/components/optimizers/pt4opt_optuna_optimizer.py",
    "mindware/components/optimizers/pt4opt_llm_tpe_optimizer.py",
    "mindware/components/optimizers/pt4opt_tpe_optimizer.py",
    "mindware/components/optimizers/pt4opt_llm_optimizer.py",
    "mindware/components/optimizers/llm_optimizer_utils.py",
    "mindware/components/optimizers/isolated_evaluator.py",
    "mindware/components/optimizers/ml_checkpoint.py",
    "examples/benchmark/ml_runtime_policy.py",
    "examples/benchmark/run_toy_categorical_optimizers.py",
    "examples/benchmark/ml_fixed_campaign.py",
    "configs/ml_runtime_policy.json",
)


def replace_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def deploy(source, campaign):
    protocol = campaign_tools.load(campaign / "protocol.json")
    if protocol.get("methods") != ["ckpt32"]:
        raise RuntimeError("This migration is scoped to the ckpt32 campaign")
    lock = campaign_tools.claim(campaign / "locks/LLMOptimizer2.lock")
    if lock is None:
        raise RuntimeError("Stop the campaign scheduler before changing its runtime policy")
    with lock:
        for path in (campaign / "state").rglob("*.json"):
            if campaign_tools.process_alive(campaign_tools.load(path)):
                raise RuntimeError(f"Worker is still alive: {path}")
        old_policy = campaign_tools.load(campaign / "runtime_policy.json")
        if old_policy["per_evaluation_seconds"] != 172800:
            raise RuntimeError("Expected the original 48-hour policy; refusing a repeated migration")
        contents = {relative: (source / relative).read_bytes() for relative in FILES}
        for relative, content in contents.items():
            if relative.endswith(".py"):
                ast.parse(content, filename=relative)
        policy = json.loads(contents["configs/ml_runtime_policy.json"])
        if policy["per_evaluation_seconds"] != 2400 or policy["overall_search_seconds"] is not None:
            raise RuntimeError("Unexpected replacement runtime limits")
        policy.update(resume_from_policy_sha256=campaign_tools.digest(campaign / "runtime_policy.json"),
                      resume_from_policy_id=old_policy["policy_id"])
        tasks = campaign_tools.load(campaign / "tasks_132.json")
        before, unchanged = [], {}
        archive_paths = [campaign / name for name in ("runtime_policy.json", "protocol.json", "code_sha256.json")]
        archive_paths += [campaign / "code" / name for name in FILES if (campaign / "code" / name).exists()]
        for task in tasks:
            key = campaign_tools.task_key(task)
            session = campaign / "sessions/ckpt32" / key
            progress = campaign_tools.progress(campaign, "ckpt32", task)
            before.append({"task": key, "attempts": progress["attempts"],
                           "clean_complete": progress["clean_complete"],
                           "identity": progress.get("identity")})
            for path in (session / "checkpoint.pkl", session / "progress.json"):
                if path.exists():
                    unchanged[str(path)] = campaign_tools.digest(path)
                    if not progress["budget_exhausted"]:
                        archive_paths.append(path)
            trajectory = Path(progress["trajectory"]) if progress.get("trajectory") else None
            if trajectory and trajectory.exists():
                unchanged[str(trajectory)] = campaign_tools.digest(trajectory)
                config = trajectory.parent / "config.json"
                if config.exists() and not progress["budget_exhausted"]:
                    archive_paths.append(config)
        revision = campaign / "maintenance/timeout2400_20261001"
        revision.mkdir(parents=True, exist_ok=False)
        with tarfile.open(revision / "before.tar", "w") as archive:
            for path in dict.fromkeys(archive_paths):
                archive.add(path, arcname=str(path.relative_to(campaign)))
        campaign_tools.save(revision / "progress_before.json", before)
        campaign_tools.save(revision / "preserved_artifacts_sha256.json", unchanged)
        hashes = campaign_tools.load(campaign / "code_sha256.json")
        for relative, content in contents.items():
            replace_bytes(campaign / "code" / relative, content)
            hashes[relative] = hashlib.sha256(content).hexdigest()
        campaign_tools.save(campaign / "code_sha256.json", hashes)
        campaign_tools.save(campaign / "runtime_policy.json", policy)
        protocol.update(initial_policy_id=old_policy["policy_id"], runtime_policy_id=policy["policy_id"],
                        policy_sha256=campaign_tools.digest(campaign / "runtime_policy.json"),
                        runtime_revision="timeout2400_optuna_attribution_20261001",
                        runtime_revision_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                        runtime_revision_archive=str(revision / "before.tar"),
                        resume_policy="Retain completed tasks and resume partial tasks; retain prior timing metadata")
        spec = protocol["hosts"]["LLMOptimizer2"]
        spec["gpu_slots"] = [0, 0, 1, 1, 2, 2]
        spec["limits"]["ckpt32"] = 6
        spec["cpus"] = list(range(24))
        protocol["gpu_policy"] = "Two ckpt32 workers per GPU on 0-2; GPU3 remains stopped; existing training retained"
        campaign_tools.save(campaign / "protocol.json", protocol)
        for path, expected in unchanged.items():
            if campaign_tools.digest(path) != expected:
                raise RuntimeError(f"An existing result changed during deployment: {path}")
        report = {"deployed": True, "workers_started": False, "campaign": str(campaign),
                  "policy": policy, "completed_retained": sum(t["clean_complete"] for t in before),
                  "partial_to_resume": sum(0 < t["attempts"] < 100 for t in before),
                  "pending": sum(t["attempts"] == 0 for t in before),
                  "preserved_artifacts": len(unchanged), "code_files": len(contents),
                  "archive": str(revision / "before.tar")}
        campaign_tools.save(revision / "deployment.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    arguments = parser.parse_args()
    deploy(arguments.source.resolve(), arguments.campaign.resolve())
