"""Prepare the approved fresh LLM8 / RGPE / PUCB cooldown campaign."""

import argparse
from pathlib import Path
import socket

import ml_fixed_campaign as campaign
from prepare_ckpt32_retry5_campaign import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.previous, args.campaign)
    protocol = campaign.load(args.campaign / "protocol.json")
    protocol.update(
        ckpt32_optimizer="ptllm_rgpe_qwen",
        ckpt32_revision="ckpt32_rgpe_p07p03_llm8_cool20_20261002",
        ckpt32_hybrid_variant="HPUCB_rgpe_p07p03_retry5_llm8_cool20",
        ckpt32_duplicate_policy="Five LLM calls; RGPE supplies after exhaustion. Both arms get an attempt; only RGPE gets evaluation credit.",
        rgpe_assets=str(args.assets.resolve()),
        rgpe_asset_hashes={str(p.relative_to(args.assets)): campaign.digest(p)
                           for p in args.assets.rglob("*") if p.is_file() and "fsbo" not in p.parts},
        cpu_policy="Two workers, three disjoint physical cores each: 16-18 and 19-21; scheduler on monitoring core 22.",
        gpu_policy="One ckpt32 worker on GPU0 and one on GPU1, alongside existing BOLT training.",
        schedule={"warm_start_rounds": [1, 3], "forced_llm_rounds": [4, 8],
                  "pucb_rounds": [9, 100], "llm_prior": [0.7, 0.3], "rgpe_prior": [0.3, 0.7],
                  "cooldown_window": [9, 20], "round9_starts_without_cooldown": True,
                  "cooldown_trigger": "selected LLM, including fallback", "q": "arm_best"},
        adapter_lineage="Verified loaded v4 ckpt32 tensors; historical GRPO parent-inheritance audit remains unresolved.",
        hosts={"LLMOptimizer2": {"hostname": socket.gethostname(), "cpus": list(range(16, 22)),
                                 "cpu_quota": 40, "memory_gib": 400, "limits": {"ckpt32": 2},
                                 "gpu_slots": [0, 1], "gpu_methods": ["ckpt32"],
                                 "worker_cores": {"ckpt32": 3}, "scheduler_cpus": [22],
                                 "memory_guard": "cgroup_working_set"}})
    campaign.save(args.campaign / "protocol.json", protocol)
    campaign.status(args.campaign)


if __name__ == "__main__":
    main()
