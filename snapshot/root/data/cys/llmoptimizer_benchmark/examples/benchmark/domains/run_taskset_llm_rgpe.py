#!/usr/bin/env python3
"""Run TaskSet LLM-RGPE on one test210 task (not launched by default anywhere).

Ingredients (all hashed into the run identity, so a resume with changed inputs fails):
  --adapter        in-loop TaskSet LoRA on Qwen3.5-9B
  --presets        warm-start presets JSON {task_id: [3 flat configs]} from gen_taskset_ws3_presets.py
  --bank           experience bank from build_taskset_experience_bank.py
  --rgpe-assets    verified 838-source RGPE ensemble (same as the RGPE test210 baseline)

Protocol matches the existing TaskSet baselines: 100 evaluations, 1000 steps,
eval at 1000, one replica, min_normalized_valid_outer. ``--dummy`` swaps in the
analytic dummy oracle for wiring checks (and requires ``--check``).
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.dlhpo.taskset_llm import experience as E, spec as S
from mindware.domains.dlhpo.taskset_llm.llm_client import QwenLoRAClient
from mindware.domains.dlhpo.taskset_llm.inference import SharedQwenClient
from mindware.domains.dlhpo.taskset_llm.loop import TaskSetLLMRGPE
from taskset_transfer_common import METRIC, PROTOCOL, atomic_json, digest, read_split, run_lock

RGPE_CAMPAIGN = Path("/root/data/logs/taskset_rgpe_test210_20261003")
BANK = ROOT / "history_data/domain_taskset/experience_bank_train838.jsonl"
RGPE_PYTHON = "/root/data/cys/baseline/envs/bl_rgpe/bin/python"
RGPE_PYTHONPATH = "/root/data/cys/baseline/repos/transfer-hpo-framework"


def file_sha(path):
    return digest(path)


class SourcePredictor:
    """Calls taskset_source_rank_worker.py in the bl_rgpe environment."""

    def __init__(self, assets, work_dir):
        self.assets, self.work = Path(assets), Path(work_dir)
        self.cache_sha256 = json.loads((self.assets / "ensemble.json").read_text())["cache_sha256"]

    def __call__(self, tasks, configs):
        self.work.mkdir(parents=True, exist_ok=True)
        request, response = self.work / "request.json", self.work / "response.json"
        atomic_json(request, dict(cache_sha256=self.cache_sha256, source_tasks=list(tasks), target_configs=configs))
        response.unlink(missing_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=RGPE_PYTHONPATH)
        with (self.work / "worker.log").open("w") as log:
            subprocess.run([RGPE_PYTHON, str(Path(__file__).with_name("taskset_source_rank_worker.py")),
                            "--assets", str(self.assets), "--request", str(request), "--out", str(response)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
        result = json.loads(response.read_text())
        if result["cache_sha256"] != self.cache_sha256 or set(result["predictions"]) != set(tasks):
            raise ValueError("Source prediction response does not match the request")
        return result["predictions"]


def rgpe_bridge(cs, run_dir, assets, task):
    from run_taskset_rgpe import suggest

    def call(observations, trial):
        return suggest(cs, list(observations), Path(run_dir), Path(assets), task, trial)
    return call


def main(args):
    split = read_split()
    if args.task not in split["test"]:
        raise ValueError("LLM-RGPE targets must be test210 tasks (RGPE sources are train838)")
    if args.dummy and not args.check:
        raise ValueError("--dummy is only allowed with --check")
    if args.trials != 100 and not args.check:
        raise ValueError("Short runs require --check (isolated preflight directory)")
    if args.split:
        subset = json.loads(args.split.read_text())
        if (set(subset["train"]) != set(split["train"]) or not set(subset["test"]) <= set(split["test"])
                or len(set(subset["test"])) != len(subset["test"]) or args.task not in subset["test"]):
            raise ValueError("Invalid held-out campaign subset or task outside subset")
    presets_all = json.loads(args.presets.read_text())
    presets = presets_all.get(args.task) or []
    assets = args.rgpe_campaign / "assets"
    meta = json.loads((assets / "ensemble.json").read_text())
    run_dir = args.campaign / ("preflight" if args.check else "test") / args.task / "llm_rgpe" / f"seed{args.seed}"
    adapter = TaskSetAdapter(task={"task_id": args.task}, dummy=args.dummy, oracle_work_dir=run_dir / "_oracle",
                             baseline_cache=args.baseline_cache, objective_metric=METRIC,
                             warm_timeout=3600, replica_timeout=2400, **PROTOCOL)
    cs = adapter.search_space()
    identity = dict(domain="taskset", protocol=PROTOCOL, objective_metric=METRIC, dummy=bool(args.dummy),
                    split_sha256=digest(Path("/root/data/logs/taskset_major_families_cpu_20260929/split.json")),
                    adapter_path=str(args.adapter), adapter_sha256=file_sha(args.adapter / "adapter_model.safetensors"),
                    presets_sha256=file_sha(args.presets), bank_sha256=file_sha(args.bank),
                    rgpe_cache_sha256=meta["cache_sha256"], llm_temperature=args.temperature,
                    llm_max_new_tokens=args.max_new_tokens,
                    runner_sha256=file_sha(__file__),
                    code_sha256={p.name: file_sha(p) for p in sorted((ROOT / "mindware/domains/dlhpo/taskset_llm").glob("*.py"))})
    if args.llm_endpoint:
        if not args.llm_token_file or not args.llm_identity_file:
            raise ValueError("Shared inference requires --llm-token-file and --llm-identity-file")
        expected = json.loads(args.llm_identity_file.read_text())["identity_sha256"]
        llm = SharedQwenClient(args.llm_endpoint, args.llm_token_file, expected_identity=expected,
                               temperature=args.temperature, seed=args.seed)
        served = llm.model_identity
        if (served["files"].get(str((args.adapter / "adapter_model.safetensors").resolve()))
                != identity["adapter_sha256"]
                or served["client"]["max_new_tokens"] != args.max_new_tokens
                or not (served["client"].get("adapter_verification") or {}).get("exact")):
            raise ValueError("Shared service has the wrong adapter, precision verification or token limit")
        identity["inference"] = llm.identity()
    else:
        llm = QwenLoRAClient(str(args.adapter), temperature=args.temperature, max_new_tokens=args.max_new_tokens)
        identity["inference"] = {"client": "local_qwen"}
    identity["campaign_split_sha256"] = file_sha(args.split) if args.split else identity["split_sha256"]
    loop = TaskSetLLMRGPE(
        task_id=args.task, cs=cs, evaluate=adapter.evaluate_config, run_dir=run_dir, llm=llm, presets=presets,
        specs=S.load_task_specs(), rgpe_suggest=rgpe_bridge(cs, run_dir, assets, args.task),
        bank=E.load_bank(args.bank), source_predict=SourcePredictor(assets, run_dir / "_source_rank"),
        identity=identity, budget=args.trials, seed=args.seed)
    loop.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--llm-endpoint", help="Shared inference URL; CPU worker loads no Qwen model")
    parser.add_argument("--llm-token-file", type=Path)
    parser.add_argument("--llm-identity-file", type=Path)
    parser.add_argument("--split", type=Path, help="Frozen held-out subset, e.g. the training manifest's test200.json")
    parser.add_argument("--presets", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--rgpe-campaign", type=Path, default=RGPE_CAMPAIGN)
    parser.add_argument("--baseline-cache", type=Path, default=RGPE_CAMPAIGN / "task_baselines.json")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    directory = args.campaign / ("preflight" if args.check else "test") / args.task / "llm_rgpe" / f"seed{args.seed}"
    with run_lock(directory):
        main(args)
