#!/usr/bin/env python3
"""Create an ml_test_20_fixed evaluation campaign for a new LLM+RGPE adapter.

Everything comes from the ckpt32+RGPE campaign (tasks, 2400 s policy, warm starts,
train80 bank, source GPs, RGPE assets, task order) except:
  - the adapter (hash-verified),
  - the response format: rationale + JSON (the S1b/ORPO/RL contract),
  - the code: the frozen v14 RL snapshot (adapter-aware no-think prompt template),
    plus the campaign driver with protocol-driven response format and a throttle guard,
  - algorithm-based cores on LLMOptimizer2: 1 for AdaBoost/GradientBoosting, else 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

REPO = Path("/root/blockdata/llmoptimizer_benchmark")
PARENT = REPO / "benchmark_data/ml_test_20_fixed_ckpt32_rgpe_20261002"
CODE_SOURCE = REPO / "benchmark_data/rl_rollouts/v14_orpo_rationale_20261005/code"
DRIVER = REPO / "examples/benchmark/ml_fixed_campaign.py"
HOST = {
    "hostname": "cs-94bac-0693e-server",
    "cpus": [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 23] + list(range(40, 50)),
    "cpu_quota": 40, "memory_gib": 400, "limits": {"ckpt32": 11},
    "gpu_slots": [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3], "gpu_methods": ["ckpt32"],
    "serial_worker_cores": 1, "parallel_worker_cores": 3,
    "scheduler_cpus": [22], "memory_guard": "cgroup_working_set", "throttle_quiet_passes": 6,
}
COPY = ["tasks_132.json", "runtime_policy.json", "warm_start_presets.json", "experience_bank.jsonl",
        "data_manifest.json", "queues/ckpt32.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="e.g. rl_v14 or orpo_rationale")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--adapter-sha256", required=True)
    p.add_argument("--lineage", required=True, help="One-line provenance of the adapter")
    a = p.parse_args()
    dest = REPO / f"benchmark_data/ml_test_20_fixed_{a.label}_20261005"
    if dest.exists():
        raise SystemExit(f"{dest} exists; refusing to overwrite")
    if sha256(a.adapter / "adapter_model.safetensors") != a.adapter_sha256:
        raise SystemExit("adapter hash mismatch")
    parent = json.loads((PARENT / "protocol.json").read_text())
    dest.mkdir(parents=True)
    shutil.copytree(CODE_SOURCE, dest / "code", symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(DRIVER, dest / "code/examples/benchmark/ml_fixed_campaign.py")
    for rel in COPY:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT / rel, dest / rel)
    shutil.copytree(PARENT / "source_gp_train80_v1", dest / "source_gp_train80_v1")
    for name in ("ckpt32",):
        (dest / "reused").mkdir(exist_ok=True)
        (dest / "reused" / f"{name}.json").write_text("[]\n")
    (dest / "selected_legacy.json").write_text("{}\n")
    for sub in ("state", "sessions", "outputs", "logs", "locks", "history", "hosts", "preflight"):
        (dest / sub).mkdir(exist_ok=True)
    checks = {
        "manifest": (sha256(dest / "tasks_132.json"), parent["manifest_sha256"]),
        "policy": (sha256(dest / "runtime_policy.json"), parent["policy_sha256"]),
        "warm_starts": (sha256(dest / "warm_start_presets.json"), parent["warm_start_sha256"]),
        "bank": (sha256(dest / "experience_bank.jsonl"), parent["bank_sha256"]),
        "source_gp": (sha256(dest / "source_gp_train80_v1/manifest.json"), parent["source_gp_manifest_sha256"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(f"Copied inputs differ from the parent campaign: {bad}")
    code_files = {str(f.relative_to(dest / "code")): sha256(f) for f in sorted((dest / "code").rglob("*"))
                  if f.is_file() and "datasets_bundle" not in f.relative_to(dest / "code").parts}
    (dest / "code_sha256.json").write_text(json.dumps(code_files, indent=2) + "\n")
    protocol = dict(parent)
    protocol.update(
        campaign=dest.name, created_at=time.time(), parent_campaign=str(PARENT),
        adapter=str(a.adapter), adapter_sha256=a.adapter_sha256, adapter_lineage=a.lineage,
        ckpt32_revision=f"{a.label}_rgpe_p07p03_llm8_cool20_20261005",
        ckpt32_response_format="rationale_json",
        hosts={"LLMOptimizer2": HOST},
        reuse="None; all 132 tasks start fresh",
        code_snapshot_source=str(CODE_SOURCE), code_files=len(code_files),
        cpu_policy="Algorithm-based: AdaBoost/GradientBoosting 1 core, others 3; pool 0-7,16-21,23,40-49; "
                   "never 64-127 (hyperthread twins of queryopt/postgres cores); throttle guard",
        gpu_policy="Up to 11 collectors: GPU0 x3, GPU1 x3, GPU2 x2 (shared with OptFormer), GPU3 x3 (shared with queryopt LOLBO)",
        differences_from_parent=[
            "adapter", "response format rationale_json (parent json_only)",
            "code: v14 RL snapshot (adapter-aware no-think prompt template; CatBoost thread count = pinned cores)",
            "algorithm-based worker cores and 11 GPU slots",
        ],
    )
    (dest / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"campaign": str(dest), "code_files": len(code_files), "checks": "all match parent"}))


if __name__ == "__main__":
    main()
