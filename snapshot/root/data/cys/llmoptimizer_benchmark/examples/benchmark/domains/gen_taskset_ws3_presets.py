#!/usr/bin/env python3
"""Generate three frozen TaskSet warm starts (not launched by default).

Mirrors the ML ``gen_rec4_ws3_presets.py`` recipe: 3 samples per task from the
warm-start model, temperature 0.7, top_p 0.9, 768 new tokens, duplicates
accepted. --full-model uses BOLT's full-SFT training prompt builder; --adapter
retains the legacy warm-start prompt file. Both use qwen3_5_nothink.
Unparseable samples are re-drawn up
to ``--max-draws`` times; every draw is kept in the raw log.

Output: {task_id: [config, config, config]} plus <out>.raw.jsonl and <out>.manifest.json.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mindware.domains.dlhpo.taskset_llm import spec as S
from mindware.domains.dlhpo.taskset_llm.llm_client import QwenLoRAClient
from mindware.domains.dlhpo.taskset_space import space_from_taskset
from mindware.domains.dlhpo.taskset_llm.inference import SharedQwenClient, model_identity, fingerprint

PROMPTS = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/taskset_ws_test210_prompts.jsonl")


def bolt_records(split_path=None):
    # Use exactly the full-SFT training context, with no assistant target or outcomes.
    from export_bolt_taskset_top10 import build_messages, effective_space
    from taskset_transfer_common import read_split
    parent = read_split()
    split = json.loads(Path(split_path).read_text()) if split_path else parent
    if (set(split["train"]) != set(parent["train"]) or not set(split["test"]) <= set(parent["test"])
            or len(set(split["test"])) != len(split["test"])):
        raise ValueError("BOLT presets require a valid held-out TaskSet subset")
    cs, specs = effective_space(space_from_taskset()), S.load_task_specs()
    records = [{"metadata": {"task_id": task}, "messages": build_messages(task, specs[task], cs, {})[:2]}
               for task in split["test"]]
    return cs, records


def main(args):
    if args.full_model or args.base_model:
        if args.prompts:
            raise ValueError("Model-generated warm starts use the TaskSet prompt builder, not legacy --prompts")
        cs, records = bolt_records(args.split)
    else:
        if args.split:
            raise ValueError("--split is supported with the full BOLT model")
        cs = space_from_taskset()
        args.prompts = args.prompts or PROMPTS
        records = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.shard is not None:
        index, count = map(int, args.shard.split("/"))
        if not 0 <= index < count:
            raise ValueError("Shard must satisfy 0 <= i < n")
        records = records[index::count]
    if args.prepare_prompts_only:
        args.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        return
    if args.llm_endpoint:
        if not args.base_model or not args.llm_token_file or not args.llm_identity_file:
            raise ValueError("Shared warm starts require base model, token and identity")
        expected = json.loads(args.llm_identity_file.read_text())["identity_sha256"]
        client = SharedQwenClient(args.llm_endpoint, args.llm_token_file,
                                  expected_identity=expected, temperature=args.temperature)
        served = client.model_identity["client"]
        if (served.get("adapter") is not None or Path(served["base"]).resolve() != args.base_model.resolve()
                or served["max_new_tokens"] != args.max_new_tokens
                or served["top_p"] != args.top_p):
            raise ValueError("Shared warm-start model/generation settings mismatch")
    else:
        client = QwenLoRAClient(str(args.adapter) if args.adapter else None,
                            **({"base_path": str(args.full_model or args.base_model)}
                               if (args.full_model or args.base_model) else {}),
                                temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens)
    presets, raw_path = {}, args.out.with_suffix(".raw.jsonl")
    with raw_path.open("a") as raw_log:
        for record in records:
            task = record["metadata"]["task_id"]
            configs, draws = [], 0
            while len(configs) < args.num_samples and draws < args.max_draws:
                need = min(args.num_samples - len(configs), args.max_draws - draws)
                draw_seed = int(fingerprint({"seed": args.seed, "task": task, "draw": draws})[:15], 16)
                for text in client.generate(record["messages"], n=need, seed=draw_seed):
                    draws += 1
                    entry = {"task_id": task, "draw": draws, "raw": text}
                    try:
                        config, notes, _ = S.parse_reply(text, cs)
                        configs.append(config)
                        entry.update(config=config, repairs=notes)
                    except Exception as exc:
                        entry["error"] = f"{type(exc).__name__}: {exc}"
                    raw_log.write(json.dumps(entry, sort_keys=True) + "\n")
                    raw_log.flush()
            if len(configs) < args.num_samples:
                raise RuntimeError(f"{task}: only {len(configs)} valid presets after {draws} draws")
            presets[task] = configs[:args.num_samples]
            print(json.dumps({"task": task, "completed_tasks": len(presets),
                              "shard_tasks": len(records), "draws": draws}), flush=True)
    args.out.write_text(json.dumps(presets, indent=2, sort_keys=True) + "\n")
    args.out.with_suffix(".manifest.json").write_text(json.dumps({
        "model": client.model_identity if args.llm_endpoint else model_identity(client), "prompt_records_sha256": fingerprint(records),
        "prompt_builder": "export_bolt_taskset_top10.build_messages" if (args.full_model or args.base_model) else str(args.prompts),
        "model_mode": "base" if args.base_model else ("full_sft" if args.full_model else "adapter"),
        "seed": args.seed, "split": str(args.split) if args.split else None,
        "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
        "num_samples": args.num_samples, "shard": args.shard, "n_tasks": len(presets),
        "template": client.model_identity["client"]["template"] if args.llm_endpoint else client.identity()["template"],
        "duplicates": "accepted"}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--adapter", type=Path)
    model.add_argument("--full-model", type=Path)
    model.add_argument("--base-model", type=Path, help="Clean Qwen checkpoint for an unadapted control")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--llm-endpoint")
    parser.add_argument("--llm-token-file", type=Path)
    parser.add_argument("--llm-identity-file", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--prepare-prompts-only", action="store_true", help="Write inference prompts without loading/running a model")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-draws", type=int, default=12)
    parser.add_argument("--shard", help="i/n: process every n-th task starting at i")
    main(parser.parse_args())
