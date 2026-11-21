#!/usr/bin/env python3
"""TaskSet test200 LLM + persistent FSBO min-max, with ML's selection policy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindware.domains.dlhpo.taskset_adapter import TaskSetAdapter
from mindware.domains.dlhpo.taskset_llm import experience as E, spec as S
from mindware.domains.dlhpo.taskset_llm.fsbo_bridge import FSBOBridge
from mindware.domains.dlhpo.taskset_llm.inference import SharedQwenClient
from mindware.domains.dlhpo.taskset_llm.loop import TaskSetLLMFSBO, _atomic_json
from run_taskset_llm_rgpe import SourcePredictor
from export_bolt_taskset_top10 import PARAMETER_NOTES
from taskset_transfer_common import METRIC, PROTOCOL, digest, read_split, run_lock

SPLIT = ROOT / "benchmark_data/taskset_s1b_s2g_train838_20261029_v2/test200.json"
PROTOCOL_TEXT = ("1000 evaluated training steps; VALID_OUTER measured at steps 0 and 1000; 1 replica. "
                 "nadamw_grid schedule horizon: 10000 steps, without rescaling to the evaluation budget. "
                 "Nesterov momentum is disabled.")


def main(args):
    parent, split = read_split(), json.loads(args.split.read_text())
    if (len(split["test"]) != 200 or len(set(split["test"])) != 200
            or set(split["train"]) != set(parent["train"]) or not set(split["test"]) <= set(parent["test"])
            or args.task not in split["test"]):
        raise ValueError("Expected exactly the frozen 200 held-out tasks")
    if (args.trials != 100 or args.dummy) and not args.check:
        raise ValueError("Short/dummy runs require --check")
    if (args.stop_after or args.prepare_next) and not args.check:
        raise ValueError("Bounded handoff checks require --check")
    directory = args.campaign / ("preflight" if args.check else "test") / args.task / "llm_fsbo" / f"seed{args.seed}"
    with run_lock(directory):
        presets = json.loads(args.presets.read_text())[args.task]
        if len(presets) != 3:
            raise ValueError("Exactly three frozen warm-start presets required")
        expected = json.loads(args.llm_identity_file.read_text())["identity_sha256"]
        llm = SharedQwenClient(args.llm_endpoint, args.llm_token_file, expected_identity=expected,
                              temperature=args.temperature, seed=args.seed)
        served = llm.model_identity
        adapter_sha = digest(args.adapter / "adapter_model.safetensors") if args.adapter else None
        if args.base_model:
            manifest = json.loads(args.presets.with_suffix(".manifest.json").read_text())
            if (served["client"].get("adapter") is not None
                    or Path(served["client"]["base"]).resolve() != args.base_model.resolve()
                    or manifest.get("model_mode") != "base"
                    or manifest.get("sha256") != digest(args.presets)
                    or manifest.get("model_identity_sha256") != expected):
                raise ValueError("Stock Qwen service/warm-start provenance mismatch")
        elif (served["files"].get(str((args.adapter / "adapter_model.safetensors").resolve())) != adapter_sha
                or not (served["client"].get("adapter_verification") or {}).get("exact")):
            raise ValueError("Shared service adapter/precision mismatch")
        if served["client"]["max_new_tokens"] != args.max_new_tokens:
            raise ValueError("Shared service adapter/precision/token limit mismatch")
        adapter = TaskSetAdapter(task={"task_id": args.task}, dummy=args.dummy,
                                 oracle_work_dir=directory / "_oracle", baseline_cache=args.baseline_cache,
                                 objective_metric=METRIC, warm_timeout=3600, replica_timeout=2400, **PROTOCOL)
        cs = adapter.search_space()
        bridge = FSBOBridge(cs, args.fsbo_assets, directory / "_fsbo", seed=args.seed)
        source_tasks = bridge.meta["train_tasks"] + bridge.meta["validation_tasks"]
        if len(source_tasks) != 838 or set(source_tasks) != set(parent["train"]):
            bridge.close()
            raise ValueError("FSBO meta-training must contain only the frozen train838 sources")
        identity = dict(domain="taskset", protocol=PROTOCOL, objective_metric=METRIC,
                        dummy=args.dummy, split_sha256=digest(args.split), adapter_sha256=adapter_sha,
                        presets_sha256=digest(args.presets), bank_sha256=digest(args.bank),
                        fsbo_weights_sha256=bridge.meta["weights_sha256"], fsbo_scaling="minmax",
                        inference=llm.identity(), runner_sha256=digest(__file__),
                        worker_sha256=digest(Path(__file__).with_name("taskset_fsbo_persistent_worker.py")),
                        code_sha256={p.name: digest(p) for p in sorted((ROOT / "mindware/domains/dlhpo/taskset_llm").glob("*.py"))})
        if args.base_model:
            identity["model_mode"] = "base"
        predictor = SourcePredictor(args.source_assets, directory / "_source_rank")
        identity["retrieval_gp_cache_sha256"] = predictor.cache_sha256
        try:
            loop = TaskSetLLMFSBO(task_id=args.task, cs=cs, evaluate=adapter.evaluate_config,
                                 run_dir=directory, llm=llm, presets=presets, specs=S.load_task_specs(),
                                 bo_suggest=bridge, bank=E.load_bank(args.bank), source_predict=predictor,
                                 identity=identity, budget=args.trials, seed=args.seed,
                                 prompt_options={"protocol_text": PROTOCOL_TEXT, "parameter_notes": PARAMETER_NOTES})
            if args.stop_after:
                if not 1 <= args.stop_after <= args.trials:
                    raise ValueError("--stop-after must be within the full trial budget")
                while len(loop.observations) < args.stop_after:
                    print(json.dumps(loop.step()), flush=True)
                if args.prepare_next and len(loop.observations) < args.trials and not loop.pending_path.exists():
                    _atomic_json(loop.pending_path, loop.propose(len(loop.observations) + 1))
                _atomic_json(directory / "preflight_status.json", {"status": "bounded_check_finished",
                             "evaluated": len(loop.observations), "full_budget": args.trials,
                             "pending_proposal": loop.pending_path.exists()})
            else:
                loop.run()
        finally:
            bridge.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "presets", "fsbo-assets", "llm-token-file", "llm-identity-file"):
        parser.add_argument("--" + name, type=Path, required=True)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--adapter", type=Path)
    model.add_argument("--base-model", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--llm-endpoint", required=True)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--bank", type=Path, default=ROOT / "history_data/domain_taskset/experience_bank_train838.jsonl")
    parser.add_argument("--source-assets", type=Path, default=Path("/root/data/logs/taskset_rgpe_test210_20261003/assets"))
    parser.add_argument("--baseline-cache", type=Path, default=Path("/root/data/logs/taskset_major_families_cpu_20260929/task_baselines.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=.2)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Preflight only: stop after N evaluations while keeping the full 100-round policy")
    parser.add_argument("--prepare-next", action="store_true", help="Preflight only: persist the next proposal to exercise restart handoff")
    main(parser.parse_args())
