#!/usr/bin/env python3
"""Build TaskSet ckpt32 SFT ("s1b") and ORPO ("attr_t2") data from the train838 TPE corpus.

See mindware/domains/dlhpo/taskset_llm/training_data.py for the recipe. Similar-task
cards (trials 4-8) need source-GP predictions from the bl_rgpe worker; pass
``--no-retrieval`` for a fast build without cards (prompts then differ from inference
at trials 4-8, which the manifest records).

Outputs in --out-dir:
  taskset_p5_sft_train838.jsonl      sharegpt messages (+ metadata)
  taskset_p5_orpo_t2_train838.jsonl  sharegpt ranking (conversations / chosen / rejected)
  taskset_p5_train838.manifest.json
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindware.domains.dlhpo.taskset_llm import experience as E, spec as S, training_data as T
from mindware.domains.dlhpo.taskset_space import space_from_taskset
from taskset_transfer_common import METRIC, PROTOCOL, SPLIT_SHA256, digest, load_corpus

CORPUS = Path("/root/data/logs/taskset_transfer_train838_20261002/corpus")
OUT_DIR = Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data")
RGPE_ASSETS = Path("/root/data/logs/taskset_rgpe_test210_20261003/assets")


def sharegpt_turns(messages):
    role = {"system": "system", "user": "human", "assistant": "gpt"}
    return [{"from": role[m["role"]], "value": m["content"]} for m in messages]


def main(args):
    _, tasks = load_corpus(args.corpus, check_sources=not args.skip_source_check)
    if args.tasks:
        tasks = {t: tasks[t] for t in args.tasks}
    specs, cs = S.load_task_specs(), space_from_taskset()
    retrieval = None
    if not args.no_retrieval:
        from run_taskset_llm_rgpe import SourcePredictor

        cards, stats = E.load_bank(args.bank)
        predictor = SourcePredictor(RGPE_ASSETS, args.work_dir / "_source_rank")

        def retrieval(task_id, prefix):
            audit = E.rerank(task_id, prefix, cards, stats, specs, predictor)
            return E.retrieval_block(audit, cards)

    sft, orpo, counts = [], [], collections.Counter()
    for task_id, record in tasks.items():
        rows = T.history_rows(record)
        for event in T.improvement_events(rows):
            messages, meta = T.prefix_prompt(task_id, specs, cs, rows, event["trial"], retrieval)
            base = {"task_id": task_id, "task_family": S.task_family_of(task_id), "trial": event["trial"],
                    "prefix_len": event["prefix_len"], "loss": event["loss"],
                    "sample_weight": event["delta_rel_score"], "experience_branch": meta["experience_branch"],
                    "retrieval_included": meta["retrieval_included"],
                    "optimizer_family": event["config"][S.FAMILY_HP]}
            sft.append({"metadata": base,
                        "messages": messages + [{"role": "assistant", "content": T.answer(event["config"])}]})
            counts["sft"] += 1
            pair = T.attribution_pair(cs, rows, event)
            if pair is None:
                counts["orpo_dropped"] += 1
                continue
            orpo.append({"metadata": dict(base, blamed=pair["blamed"], rejected_trial=pair["rejected_trial"],
                                          rejected_loss=pair["rejected_loss"]),
                         "conversations": sharegpt_turns(messages),
                         "chosen": {"from": "gpt", "value": T.answer(pair["chosen"])},
                         "rejected": {"from": "gpt", "value": T.answer(pair["rejected"])}})
            counts["orpo"] += 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"sft": args.out_dir / f"taskset_p5_sft_{args.tag}.jsonl",
             "orpo": args.out_dir / f"taskset_p5_orpo_t2_{args.tag}.jsonl"}
    for key, rows in (("sft", sft), ("orpo", orpo)):
        with paths[key].open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {"corpus_manifest_sha256": digest(args.corpus / "manifest.json"), "split_sha256": SPLIT_SHA256,
                "protocol": PROTOCOL, "metric": METRIC, "n_tasks": len(tasks), "counts": dict(counts),
                "retrieval_cards": not args.no_retrieval, "min_prefix": T.MIN_PREFIX,
                "response_format": "json_only", "prompt_seed": T.PROMPT_SEED,
                "code_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in (__file__, T.__file__, S.__file__, E.__file__)},
                "files": {k: {"path": str(p), "sha256": digest(p)} for k, p in paths.items()}}
    (args.out_dir / f"taskset_p5_{args.tag}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--bank", type=Path, default=ROOT / "history_data/domain_taskset/experience_bank_train838.jsonl")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/taskset_p5_build"))
    parser.add_argument("--tag", default="train838")
    parser.add_argument("--tasks", nargs="*", help="Restrict to these train tasks (smoke builds).")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--skip-source-check", action="store_true")
    main(parser.parse_args())
