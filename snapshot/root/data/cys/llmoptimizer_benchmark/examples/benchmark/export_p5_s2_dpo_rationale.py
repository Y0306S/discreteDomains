#!/usr/bin/env python3
"""Export P5 S2 DPO pairs with **s1b-style** chosen/rejected: ``Rationale:`` + ``JSON:``.

Same Pareto pair selection as ``export_p5_s2_dpo_pareto.py`` (pool / worse-than-x4 / mixed sim).

Rough structure
---------------
1. **Pair selection** (unchanged): ``p5_s1b_events.jsonl`` + ``p5_s1a_bc.jsonl`` prompts → (x4, rejected).
2. **Chosen rationale**: lookup from ``p5_s1b_sft_train.jsonl`` by matching the **user** prompt
   (same P5 context as the improvement event).
3. **Rejected rationale**:
   - optional cache file ``p5_s2_dpo_rationale_cache.jsonl`` (key = user_hash + config_key);
   - if missing: **Qwen3.5-9B** generates a short rationale for the fixed rejected config
     (config is not changed — only the explanation text).
4. **Assistant strings** (both arms)::

       Rationale: <text>

       JSON:
       {"config": {...}}

Outputs: ``p5_s2_dpo_rationale.jsonl`` (+ ``_train.jsonl`` for LlamaFactory ranking).

Nothing has been trained on this dataset yet; register as ``p5_s2_dpo_rationale`` for a new DPO run
from ``s1b_ep7p5`` with ``rationale_json`` eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

from export_p5_s2_dpo_pareto import (
    DEFAULT_DATASET_INFO,
    DEFAULT_EVENTS,
    DEFAULT_PROMPTS,
    build_pair,
    cfg_key,
    cfg_only,
    register_dataset,
    to_train_row,
)

DEFAULT_S1B_SFT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s1b_sft_train.jsonl"
)
DEFAULT_OUT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale.jsonl"
)
DEFAULT_TRAIN = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_train.jsonl"
)
DEFAULT_CACHE = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_cache.jsonl"
)
DEFAULT_MODEL = Path("/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")

_RATIONALE_JSON_RE = re.compile(
    r"Rationale:\s*(.*?)\n\nJSON:\s*(\{.*\})\s*$",
    re.DOTALL,
)


def user_hash(user: str) -> str:
    return hashlib.sha256(user.encode("utf-8")).hexdigest()


def parse_s1b_assistant(text: str) -> tuple[str | None, dict | None]:
    m = _RATIONALE_JSON_RE.match(text.strip())
    if not m:
        return None, None
    try:
        body = json.loads(m.group(2))
    except json.JSONDecodeError:
        return m.group(1).strip(), None
    return m.group(1).strip(), body.get("config")


def format_s1b_assistant(rationale: str, cfg: dict[str, Any]) -> str:
    rat = rationale.strip()
    if rat.lower().startswith("rationale:"):
        rat = rat.split(":", 1)[1].strip()
    cfg_json = json.dumps({"config": cfg_only(cfg)}, ensure_ascii=False)
    return f"Rationale: {rat}\n\nJSON:\n{cfg_json}"


def load_chosen_rationale_index(s1b_sft: Path) -> dict[str, str]:
    """Map sha256(user prompt) -> rationale text for the SFT target (x4) row."""
    idx: dict[str, str] = {}
    with s1b_sft.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            if len(msgs) < 3:
                continue
            user = msgs[1]["content"]
            rat, _ = parse_s1b_assistant(msgs[2]["content"])
            if rat:
                idx[user_hash(user)] = rat
    return idx


def load_rationale_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["key"]] = row["rationale"]
    return out


def append_cache(path: Path, key: str, rationale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "rationale": rationale}, ensure_ascii=False) + "\n")


def cache_key(user: str, cfg: dict[str, Any]) -> str:
    return user_hash(user) + "::" + cfg_key(cfg)


def heuristic_rationale(cfg: dict[str, Any], *, worse: bool) -> str:
    parts = [f"{k}={cfg[k]}" for k in sorted(cfg_only(cfg))]
    summary = ", ".join(parts[:6])
    if len(parts) > 6:
        summary += ", ..."
    if worse:
        return (
            f"This configuration adjusts several hyperparameters ({summary}) "
            "and is expected to underperform the current best trial on validation."
        )
    return (
        f"This configuration ({summary}) extends promising regions from recent trials "
        "while exploring complementary trade-offs in the search space."
    )


_TARGET_CFG_RE = re.compile(
    r"(TARGET CONFIGURATION[^\n]*:\n\n)(\{.*?\})(\n\nWrite exactly)",
    re.DOTALL,
)


def rationale_user_for_config(event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> tuple[str, str]:
    """Same generator contract as s1b SFT rationale step (``rationale_messages``)."""
    msgs = event["rationale_messages"]
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    cfg_json = json.dumps(cfg_only(cfg), ensure_ascii=False)
    if _TARGET_CFG_RE.search(user):
        user = _TARGET_CFG_RE.sub(r"\1" + cfg_json + r"\3", user, count=1)
    else:
        user = user + f"\n\nTARGET CONFIGURATION:\n\n{cfg_json}\n"
    if rejected:
        user = user.replace(
            "this config improved best-so-far",
            "this candidate config is proposed for the next trial",
        )
    return system, user


class QwenRationaleGenerator:
    """Generate rationale text for a fixed config (JSON block is appended separately)."""

    def __init__(self, model_path: Path, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _decode_rationale(self, gen: str, cfg: dict[str, Any]) -> str:
        gen = gen.strip()
        if "Rationale:" in gen:
            gen = gen.split("Rationale:", 1)[1].strip()
        if "JSON:" in gen:
            gen = gen.split("JSON:", 1)[0].strip()
        return gen or heuristic_rationale(cfg, worse=True)

    def generate_from_rationale_messages(self, event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> str:
        import torch

        system, user = rationale_user_for_config(event, cfg, rejected=rejected)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self._decode_rationale(gen, cfg)

    def generate_batch_from_events(
        self,
        events: list[dict[str, Any]],
        configs: list[dict[str, Any]],
        *,
        rejected: bool,
    ) -> list[str]:
        import torch

        if len(events) != len(configs):
            raise ValueError("events and configs length mismatch")
        if not events:
            return []
        prompts: list[str] = []
        for event, cfg in zip(events, configs):
            system, user = rationale_user_for_config(event, cfg, rejected=rejected)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        results: list[str] = []
        for i, cfg in enumerate(configs):
            in_len = int(inputs["attention_mask"][i].sum().item())
            gen = self.tokenizer.decode(out[i][in_len:], skip_special_tokens=True)
            results.append(self._decode_rationale(gen, cfg))
        return results


def _gpu_worker_fill_cache(
    gpu: str,
    jobs: list[dict[str, Any]],
    model_path: str,
    max_new_tokens: int,
    batch_size: int,
    cache_part: str,
) -> int:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gen = QwenRationaleGenerator(Path(model_path), max_new_tokens=max_new_tokens)
    part_path = Path(cache_part)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with part_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i : i + batch_size]
            events = [j["event"] for j in chunk]
            configs = [j["cfg"] for j in chunk]
            rats = gen.generate_batch_from_events(events, configs, rejected=True)
            for job, rat in zip(chunk, rats):
                f.write(json.dumps({"key": job["key"], "rationale": rat}, ensure_ascii=False) + "\n")
                written += 1
    return written


def parallel_fill_rejected_cache(
    jobs: list[dict[str, Any]],
    gpus: list[str],
    *,
    workers_per_gpu: int,
    model_path: Path,
    max_new_tokens: int,
    cache_dir: Path,
) -> dict[str, str]:
    """One model per GPU; each GPU runs micro-batches of size ``workers_per_gpu``."""
    if not jobs:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_gpu = len(gpus)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_gpu)]
    for i, job in enumerate(jobs):
        shards[i % n_gpu].append(job)

    ctx = mp.get_context("spawn")
    payloads = []
    for gi, gpu in enumerate(gpus):
        if not shards[gi]:
            continue
        part = cache_dir / f"rejected_part_gpu{gpu}.jsonl"
        payloads.append(
            (
                gpu,
                shards[gi],
                str(model_path),
                max_new_tokens,
                workers_per_gpu,
                str(part),
            )
        )
    with ctx.Pool(processes=len(payloads)) as pool:
        counts = pool.starmap(_gpu_worker_fill_cache, payloads)
    print(f"[export] parallel cache fill: {sum(counts)} rationales on GPUs {gpus} batch={workers_per_gpu}")

    merged: dict[str, str] = {}
    for _, _, _, _, _, part_s in payloads:
        part = Path(part_s)
        if not part.is_file():
            continue
        with part.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                merged[row["key"]] = row["rationale"]
    return merged


def resolve_rationale(
    *,
    user: str,
    cfg: dict[str, Any],
    event: dict[str, Any] | None,
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    mode: str,
    is_chosen: bool,
) -> tuple[str, str]:
    """Returns (rationale, source_tag)."""
    if is_chosen:
        rat = chosen_index.get(user_hash(user))
        if rat:
            return rat, "s1b_sft"
        if generator is not None and event is not None:
            rat = generator.generate_from_rationale_messages(event, cfg, rejected=False)
            return rat, "qwen_chosen_fallback"
        return heuristic_rationale(cfg, worse=False), "heuristic_chosen"

    key = cache_key(user, cfg)
    if key in cache:
        return cache[key], "cache"

    if mode == "heuristic":
        rat = heuristic_rationale(cfg, worse=True)
        cache[key] = rat
        append_cache(cache_path, key, rat)
        return rat, "heuristic_rejected"

    if generator is None:
        rat = heuristic_rationale(cfg, worse=True)
        return rat, "heuristic_rejected_no_model"

    if event is not None:
        rat = generator.generate_from_rationale_messages(event, cfg, rejected=True)
    else:
        rat = heuristic_rationale(cfg, worse=True)
    cache[key] = rat
    append_cache(cache_path, key, rat)
    return rat, "qwen_rejected"


def enrich_pair(
    pair: dict,
    event: dict[str, Any],
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    reject_mode: str,
) -> dict:
    system = event["s1b_system_prompt"]
    user = event["s1b_user_prompt"]
    pair["conversations"][0]["value"] = system
    pair["conversations"][1]["value"] = user
    meta = pair["metadata"]
    x4 = meta["chosen_config"]
    rej = meta["rejected_config"]

    chosen_rat, chosen_src = resolve_rationale(
        user=user,
        cfg=x4,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=True,
    )
    rejected_rat, rejected_src = resolve_rationale(
        user=user,
        cfg=rej,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=False,
    )

    pair["chosen"]["value"] = format_s1b_assistant(chosen_rat, x4)
    pair["rejected"]["value"] = format_s1b_assistant(rejected_rat, rej)
    meta["chosen_rationale_source"] = chosen_src
    meta["rejected_rationale_source"] = rejected_src
    meta["format"] = "s1b_rationale_json"
    return pair


def collect_pair_jobs(
    args: argparse.Namespace,
    chosen_index: dict[str, str],
) -> tuple[list[tuple[dict, dict]], dict[str, int]]:
    prompts: dict[tuple, dict] = {}
    with args.prompts.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = row["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompts[key] = row

    work: list[tuple[dict, dict]] = []
    skips: dict[str, int] = {}
    hist_cache: dict[str, list[dict]] = {}
    with args.events.open(encoding="utf-8") as f:
        for line in f:
            if args.max_pairs and len(work) >= args.max_pairs:
                break
            event = json.loads(line)
            m = event["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompt_row = prompts.get(key)
            if prompt_row is None:
                skips["missing_prompt"] = skips.get("missing_prompt", 0) + 1
                continue
            pair, reason = build_pair(event, prompt_row, hist_cache)
            if pair is None:
                skips[reason] = skips.get(reason, 0) + 1
                continue
            work.append((event, pair))
    return work, skips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--s1b-sft", type=Path, default=DEFAULT_S1B_SFT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--register-name", type=str, default="p5_s2_dpo_rationale")
    parser.add_argument(
        "--reject-rationale-mode",
        choices=("qwen", "heuristic"),
        default="qwen",
        help="How to fill rejected rationales when not in cache.",
    )
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pair selection + chosen lookup only; rejected=heuristic; no Qwen load.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids (e.g. 0,1,2,3). One Qwen per GPU, batched rejected generation.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Micro-batch size per GPU (default 3 → ~12 concurrent seqs on 4 GPUs).",
    )
    args = parser.parse_args()

    chosen_index = load_chosen_rationale_index(args.s1b_sft)
    cache = load_rationale_cache(args.cache)

    reject_mode = "heuristic" if args.dry_run else args.reject_rationale_mode
    work, skips = collect_pair_jobs(args, chosen_index)

    parallel_gpus = [g.strip() for g in args.parallel_gpus.split(",") if g.strip()]
    use_parallel = bool(parallel_gpus) and not args.dry_run and reject_mode == "qwen"

    generator: QwenRationaleGenerator | None = None
    if not use_parallel and not args.dry_run and reject_mode == "qwen":
        print(f"[export] loading Qwen for rejected rationales: {args.qwen_model}")
        generator = QwenRationaleGenerator(args.qwen_model, max_new_tokens=args.max_new_tokens)

    if use_parallel:
        pending: list[dict[str, Any]] = []
        for event, pair in work:
            user = event["s1b_user_prompt"]
            rej = pair["metadata"]["rejected_config"]
            key = cache_key(user, rej)
            if key not in cache:
                pending.append({"key": key, "event": event, "cfg": rej, "user": user})
        print(
            f"[export] parallel: {len(pending)} rejected to generate on GPUs {parallel_gpus} "
            f"(batch {args.workers_per_gpu}/GPU)"
        )
        cache_dir = args.cache.parent / "p5_s2_dpo_rationale_cache_parts"
        new_entries = parallel_fill_rejected_cache(
            pending,
            parallel_gpus,
            workers_per_gpu=args.workers_per_gpu,
            model_path=args.qwen_model,
            max_new_tokens=args.max_new_tokens,
            cache_dir=cache_dir,
        )
        cache.update(new_entries)
        with args.cache.open("a", encoding="utf-8") as f:
            for key, rat in new_entries.items():
                f.write(json.dumps({"key": key, "rationale": rat}, ensure_ascii=False) + "\n")

    rows: list[dict] = []
    sources_chosen: dict[str, int] = {}
    sources_rejected: dict[str, int] = {}

    for event, pair in work:
        pair = enrich_pair(
            pair,
            event,
            chosen_index,
            cache,
            args.cache,
            generator,
            reject_mode,
        )
        rows.append(pair)
        cs = pair["metadata"]["chosen_rationale_source"]
        rs = pair["metadata"]["rejected_rationale_source"]
        sources_chosen[cs] = sources_chosen.get(cs, 0) + 1
        sources_rejected[rs] = sources_rejected.get(rs, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.train_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_train_row(row), ensure_ascii=False) + "\n")

    manifest = {
        "pairs": len(rows),
        "skips": skips,
        "chosen_rationale_sources": sources_chosen,
        "rejected_rationale_sources": sources_rejected,
        "out": str(args.out),
        "train_out": str(args.train_out),
        "cache": str(args.cache),
        "format": "sharegpt_ranking_dpo_s1b_rationale_json",
        "reject_rationale_mode": reject_mode,
        "dry_run": args.dry_run,
        "parallel_gpus": parallel_gpus,
        "workers_per_gpu": args.workers_per_gpu,
    }
    man_path = args.out.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    register_dataset(args.dataset_info, args.register_name, f"../data/{args.train_out.name}")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.train_out}")
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Export P5 S2 DPO pairs with **s1b-style** chosen/rejected: ``Rationale:`` + ``JSON:``.

Same Pareto pair selection as ``export_p5_s2_dpo_pareto.py`` (pool / worse-than-x4 / mixed sim).

Rough structure
---------------
1. **Pair selection** (unchanged): ``p5_s1b_events.jsonl`` + ``p5_s1a_bc.jsonl`` prompts → (x4, rejected).
2. **Chosen rationale**: lookup from ``p5_s1b_sft_train.jsonl`` by matching the **user** prompt
   (same P5 context as the improvement event).
3. **Rejected rationale**:
   - optional cache file ``p5_s2_dpo_rationale_cache.jsonl`` (key = user_hash + config_key);
   - if missing: **Qwen3.5-9B** generates a short rationale for the fixed rejected config
     (config is not changed — only the explanation text).
4. **Assistant strings** (both arms)::

       Rationale: <text>

       JSON:
       {"config": {...}}

Outputs: ``p5_s2_dpo_rationale.jsonl`` (+ ``_train.jsonl`` for LlamaFactory ranking).

Nothing has been trained on this dataset yet; register as ``p5_s2_dpo_rationale`` for a new DPO run
from ``s1b_ep7p5`` with ``rationale_json`` eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

from export_p5_s2_dpo_pareto import (
    DEFAULT_DATASET_INFO,
    DEFAULT_EVENTS,
    DEFAULT_PROMPTS,
    build_pair,
    cfg_key,
    cfg_only,
    register_dataset,
    to_train_row,
)

DEFAULT_S1B_SFT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s1b_sft_train.jsonl"
)
DEFAULT_OUT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale.jsonl"
)
DEFAULT_TRAIN = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_train.jsonl"
)
DEFAULT_CACHE = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_cache.jsonl"
)
DEFAULT_MODEL = Path("/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")

_RATIONALE_JSON_RE = re.compile(
    r"Rationale:\s*(.*?)\n\nJSON:\s*(\{.*\})\s*$",
    re.DOTALL,
)


def user_hash(user: str) -> str:
    return hashlib.sha256(user.encode("utf-8")).hexdigest()


def parse_s1b_assistant(text: str) -> tuple[str | None, dict | None]:
    m = _RATIONALE_JSON_RE.match(text.strip())
    if not m:
        return None, None
    try:
        body = json.loads(m.group(2))
    except json.JSONDecodeError:
        return m.group(1).strip(), None
    return m.group(1).strip(), body.get("config")


def format_s1b_assistant(rationale: str, cfg: dict[str, Any]) -> str:
    rat = rationale.strip()
    if rat.lower().startswith("rationale:"):
        rat = rat.split(":", 1)[1].strip()
    cfg_json = json.dumps({"config": cfg_only(cfg)}, ensure_ascii=False)
    return f"Rationale: {rat}\n\nJSON:\n{cfg_json}"


def load_chosen_rationale_index(s1b_sft: Path) -> dict[str, str]:
    """Map sha256(user prompt) -> rationale text for the SFT target (x4) row."""
    idx: dict[str, str] = {}
    with s1b_sft.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            if len(msgs) < 3:
                continue
            user = msgs[1]["content"]
            rat, _ = parse_s1b_assistant(msgs[2]["content"])
            if rat:
                idx[user_hash(user)] = rat
    return idx


def load_rationale_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["key"]] = row["rationale"]
    return out


def append_cache(path: Path, key: str, rationale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "rationale": rationale}, ensure_ascii=False) + "\n")


def cache_key(user: str, cfg: dict[str, Any]) -> str:
    return user_hash(user) + "::" + cfg_key(cfg)


def heuristic_rationale(cfg: dict[str, Any], *, worse: bool) -> str:
    parts = [f"{k}={cfg[k]}" for k in sorted(cfg_only(cfg))]
    summary = ", ".join(parts[:6])
    if len(parts) > 6:
        summary += ", ..."
    if worse:
        return (
            f"This configuration adjusts several hyperparameters ({summary}) "
            "and is expected to underperform the current best trial on validation."
        )
    return (
        f"This configuration ({summary}) extends promising regions from recent trials "
        "while exploring complementary trade-offs in the search space."
    )


_TARGET_CFG_RE = re.compile(
    r"(TARGET CONFIGURATION[^\n]*:\n\n)(\{.*?\})(\n\nWrite exactly)",
    re.DOTALL,
)


def rationale_user_for_config(event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> tuple[str, str]:
    """Same generator contract as s1b SFT rationale step (``rationale_messages``)."""
    msgs = event["rationale_messages"]
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    cfg_json = json.dumps(cfg_only(cfg), ensure_ascii=False)
    if _TARGET_CFG_RE.search(user):
        user = _TARGET_CFG_RE.sub(r"\1" + cfg_json + r"\3", user, count=1)
    else:
        user = user + f"\n\nTARGET CONFIGURATION:\n\n{cfg_json}\n"
    if rejected:
        user = user.replace(
            "this config improved best-so-far",
            "this candidate config is proposed for the next trial",
        )
    return system, user


class QwenRationaleGenerator:
    """Generate rationale text for a fixed config (JSON block is appended separately)."""

    def __init__(self, model_path: Path, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _decode_rationale(self, gen: str, cfg: dict[str, Any]) -> str:
        gen = gen.strip()
        if "Rationale:" in gen:
            gen = gen.split("Rationale:", 1)[1].strip()
        if "JSON:" in gen:
            gen = gen.split("JSON:", 1)[0].strip()
        return gen or heuristic_rationale(cfg, worse=True)

    def generate_from_rationale_messages(self, event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> str:
        import torch

        system, user = rationale_user_for_config(event, cfg, rejected=rejected)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self._decode_rationale(gen, cfg)

    def generate_batch_from_events(
        self,
        events: list[dict[str, Any]],
        configs: list[dict[str, Any]],
        *,
        rejected: bool,
    ) -> list[str]:
        import torch

        if len(events) != len(configs):
            raise ValueError("events and configs length mismatch")
        if not events:
            return []
        prompts: list[str] = []
        for event, cfg in zip(events, configs):
            system, user = rationale_user_for_config(event, cfg, rejected=rejected)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        results: list[str] = []
        for i, cfg in enumerate(configs):
            in_len = int(inputs["attention_mask"][i].sum().item())
            gen = self.tokenizer.decode(out[i][in_len:], skip_special_tokens=True)
            results.append(self._decode_rationale(gen, cfg))
        return results


def _gpu_worker_fill_cache(
    gpu: str,
    jobs: list[dict[str, Any]],
    model_path: str,
    max_new_tokens: int,
    batch_size: int,
    cache_part: str,
) -> int:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gen = QwenRationaleGenerator(Path(model_path), max_new_tokens=max_new_tokens)
    part_path = Path(cache_part)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with part_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i : i + batch_size]
            events = [j["event"] for j in chunk]
            configs = [j["cfg"] for j in chunk]
            rats = gen.generate_batch_from_events(events, configs, rejected=True)
            for job, rat in zip(chunk, rats):
                f.write(json.dumps({"key": job["key"], "rationale": rat}, ensure_ascii=False) + "\n")
                written += 1
    return written


def parallel_fill_rejected_cache(
    jobs: list[dict[str, Any]],
    gpus: list[str],
    *,
    workers_per_gpu: int,
    model_path: Path,
    max_new_tokens: int,
    cache_dir: Path,
) -> dict[str, str]:
    """One model per GPU; each GPU runs micro-batches of size ``workers_per_gpu``."""
    if not jobs:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_gpu = len(gpus)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_gpu)]
    for i, job in enumerate(jobs):
        shards[i % n_gpu].append(job)

    ctx = mp.get_context("spawn")
    payloads = []
    for gi, gpu in enumerate(gpus):
        if not shards[gi]:
            continue
        part = cache_dir / f"rejected_part_gpu{gpu}.jsonl"
        payloads.append(
            (
                gpu,
                shards[gi],
                str(model_path),
                max_new_tokens,
                workers_per_gpu,
                str(part),
            )
        )
    with ctx.Pool(processes=len(payloads)) as pool:
        counts = pool.starmap(_gpu_worker_fill_cache, payloads)
    print(f"[export] parallel cache fill: {sum(counts)} rationales on GPUs {gpus} batch={workers_per_gpu}")

    merged: dict[str, str] = {}
    for _, _, _, _, _, part_s in payloads:
        part = Path(part_s)
        if not part.is_file():
            continue
        with part.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                merged[row["key"]] = row["rationale"]
    return merged


def resolve_rationale(
    *,
    user: str,
    cfg: dict[str, Any],
    event: dict[str, Any] | None,
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    mode: str,
    is_chosen: bool,
) -> tuple[str, str]:
    """Returns (rationale, source_tag)."""
    if is_chosen:
        rat = chosen_index.get(user_hash(user))
        if rat:
            return rat, "s1b_sft"
        if generator is not None and event is not None:
            rat = generator.generate_from_rationale_messages(event, cfg, rejected=False)
            return rat, "qwen_chosen_fallback"
        return heuristic_rationale(cfg, worse=False), "heuristic_chosen"

    key = cache_key(user, cfg)
    if key in cache:
        return cache[key], "cache"

    if mode == "heuristic":
        rat = heuristic_rationale(cfg, worse=True)
        cache[key] = rat
        append_cache(cache_path, key, rat)
        return rat, "heuristic_rejected"

    if generator is None:
        rat = heuristic_rationale(cfg, worse=True)
        return rat, "heuristic_rejected_no_model"

    if event is not None:
        rat = generator.generate_from_rationale_messages(event, cfg, rejected=True)
    else:
        rat = heuristic_rationale(cfg, worse=True)
    cache[key] = rat
    append_cache(cache_path, key, rat)
    return rat, "qwen_rejected"


def enrich_pair(
    pair: dict,
    event: dict[str, Any],
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    reject_mode: str,
) -> dict:
    system = event["s1b_system_prompt"]
    user = event["s1b_user_prompt"]
    pair["conversations"][0]["value"] = system
    pair["conversations"][1]["value"] = user
    meta = pair["metadata"]
    x4 = meta["chosen_config"]
    rej = meta["rejected_config"]

    chosen_rat, chosen_src = resolve_rationale(
        user=user,
        cfg=x4,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=True,
    )
    rejected_rat, rejected_src = resolve_rationale(
        user=user,
        cfg=rej,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=False,
    )

    pair["chosen"]["value"] = format_s1b_assistant(chosen_rat, x4)
    pair["rejected"]["value"] = format_s1b_assistant(rejected_rat, rej)
    meta["chosen_rationale_source"] = chosen_src
    meta["rejected_rationale_source"] = rejected_src
    meta["format"] = "s1b_rationale_json"
    return pair


def collect_pair_jobs(
    args: argparse.Namespace,
    chosen_index: dict[str, str],
) -> tuple[list[tuple[dict, dict]], dict[str, int]]:
    prompts: dict[tuple, dict] = {}
    with args.prompts.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = row["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompts[key] = row

    work: list[tuple[dict, dict]] = []
    skips: dict[str, int] = {}
    hist_cache: dict[str, list[dict]] = {}
    with args.events.open(encoding="utf-8") as f:
        for line in f:
            if args.max_pairs and len(work) >= args.max_pairs:
                break
            event = json.loads(line)
            m = event["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompt_row = prompts.get(key)
            if prompt_row is None:
                skips["missing_prompt"] = skips.get("missing_prompt", 0) + 1
                continue
            pair, reason = build_pair(event, prompt_row, hist_cache)
            if pair is None:
                skips[reason] = skips.get(reason, 0) + 1
                continue
            work.append((event, pair))
    return work, skips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--s1b-sft", type=Path, default=DEFAULT_S1B_SFT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--register-name", type=str, default="p5_s2_dpo_rationale")
    parser.add_argument(
        "--reject-rationale-mode",
        choices=("qwen", "heuristic"),
        default="qwen",
        help="How to fill rejected rationales when not in cache.",
    )
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pair selection + chosen lookup only; rejected=heuristic; no Qwen load.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids (e.g. 0,1,2,3). One Qwen per GPU, batched rejected generation.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Micro-batch size per GPU (default 3 → ~12 concurrent seqs on 4 GPUs).",
    )
    args = parser.parse_args()

    chosen_index = load_chosen_rationale_index(args.s1b_sft)
    cache = load_rationale_cache(args.cache)

    reject_mode = "heuristic" if args.dry_run else args.reject_rationale_mode
    work, skips = collect_pair_jobs(args, chosen_index)

    parallel_gpus = [g.strip() for g in args.parallel_gpus.split(",") if g.strip()]
    use_parallel = bool(parallel_gpus) and not args.dry_run and reject_mode == "qwen"

    generator: QwenRationaleGenerator | None = None
    if not use_parallel and not args.dry_run and reject_mode == "qwen":
        print(f"[export] loading Qwen for rejected rationales: {args.qwen_model}")
        generator = QwenRationaleGenerator(args.qwen_model, max_new_tokens=args.max_new_tokens)

    if use_parallel:
        pending: list[dict[str, Any]] = []
        for event, pair in work:
            user = event["s1b_user_prompt"]
            rej = pair["metadata"]["rejected_config"]
            key = cache_key(user, rej)
            if key not in cache:
                pending.append({"key": key, "event": event, "cfg": rej, "user": user})
        print(
            f"[export] parallel: {len(pending)} rejected to generate on GPUs {parallel_gpus} "
            f"(batch {args.workers_per_gpu}/GPU)"
        )
        cache_dir = args.cache.parent / "p5_s2_dpo_rationale_cache_parts"
        new_entries = parallel_fill_rejected_cache(
            pending,
            parallel_gpus,
            workers_per_gpu=args.workers_per_gpu,
            model_path=args.qwen_model,
            max_new_tokens=args.max_new_tokens,
            cache_dir=cache_dir,
        )
        cache.update(new_entries)
        with args.cache.open("a", encoding="utf-8") as f:
            for key, rat in new_entries.items():
                f.write(json.dumps({"key": key, "rationale": rat}, ensure_ascii=False) + "\n")

    rows: list[dict] = []
    sources_chosen: dict[str, int] = {}
    sources_rejected: dict[str, int] = {}

    for event, pair in work:
        pair = enrich_pair(
            pair,
            event,
            chosen_index,
            cache,
            args.cache,
            generator,
            reject_mode,
        )
        rows.append(pair)
        cs = pair["metadata"]["chosen_rationale_source"]
        rs = pair["metadata"]["rejected_rationale_source"]
        sources_chosen[cs] = sources_chosen.get(cs, 0) + 1
        sources_rejected[rs] = sources_rejected.get(rs, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.train_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_train_row(row), ensure_ascii=False) + "\n")

    manifest = {
        "pairs": len(rows),
        "skips": skips,
        "chosen_rationale_sources": sources_chosen,
        "rejected_rationale_sources": sources_rejected,
        "out": str(args.out),
        "train_out": str(args.train_out),
        "cache": str(args.cache),
        "format": "sharegpt_ranking_dpo_s1b_rationale_json",
        "reject_rationale_mode": reject_mode,
        "dry_run": args.dry_run,
        "parallel_gpus": parallel_gpus,
        "workers_per_gpu": args.workers_per_gpu,
    }
    man_path = args.out.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    register_dataset(args.dataset_info, args.register_name, f"../data/{args.train_out.name}")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.train_out}")
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Export P5 S2 DPO pairs with **s1b-style** chosen/rejected: ``Rationale:`` + ``JSON:``.

Same Pareto pair selection as ``export_p5_s2_dpo_pareto.py`` (pool / worse-than-x4 / mixed sim).

Rough structure
---------------
1. **Pair selection** (unchanged): ``p5_s1b_events.jsonl`` + ``p5_s1a_bc.jsonl`` prompts → (x4, rejected).
2. **Chosen rationale**: lookup from ``p5_s1b_sft_train.jsonl`` by matching the **user** prompt
   (same P5 context as the improvement event).
3. **Rejected rationale**:
   - optional cache file ``p5_s2_dpo_rationale_cache.jsonl`` (key = user_hash + config_key);
   - if missing: **Qwen3.5-9B** generates a short rationale for the fixed rejected config
     (config is not changed — only the explanation text).
4. **Assistant strings** (both arms)::

       Rationale: <text>

       JSON:
       {"config": {...}}

Outputs: ``p5_s2_dpo_rationale.jsonl`` (+ ``_train.jsonl`` for LlamaFactory ranking).

Nothing has been trained on this dataset yet; register as ``p5_s2_dpo_rationale`` for a new DPO run
from ``s1b_ep7p5`` with ``rationale_json`` eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

from export_p5_s2_dpo_pareto import (
    DEFAULT_DATASET_INFO,
    DEFAULT_EVENTS,
    DEFAULT_PROMPTS,
    build_pair,
    cfg_key,
    cfg_only,
    register_dataset,
    to_train_row,
)

DEFAULT_S1B_SFT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s1b_sft_train.jsonl"
)
DEFAULT_OUT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale.jsonl"
)
DEFAULT_TRAIN = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_train.jsonl"
)
DEFAULT_CACHE = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_cache.jsonl"
)
DEFAULT_MODEL = Path("/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")

_RATIONALE_JSON_RE = re.compile(
    r"Rationale:\s*(.*?)\n\nJSON:\s*(\{.*\})\s*$",
    re.DOTALL,
)


def user_hash(user: str) -> str:
    return hashlib.sha256(user.encode("utf-8")).hexdigest()


def parse_s1b_assistant(text: str) -> tuple[str | None, dict | None]:
    m = _RATIONALE_JSON_RE.match(text.strip())
    if not m:
        return None, None
    try:
        body = json.loads(m.group(2))
    except json.JSONDecodeError:
        return m.group(1).strip(), None
    return m.group(1).strip(), body.get("config")


def format_s1b_assistant(rationale: str, cfg: dict[str, Any]) -> str:
    rat = rationale.strip()
    if rat.lower().startswith("rationale:"):
        rat = rat.split(":", 1)[1].strip()
    cfg_json = json.dumps({"config": cfg_only(cfg)}, ensure_ascii=False)
    return f"Rationale: {rat}\n\nJSON:\n{cfg_json}"


def load_chosen_rationale_index(s1b_sft: Path) -> dict[str, str]:
    """Map sha256(user prompt) -> rationale text for the SFT target (x4) row."""
    idx: dict[str, str] = {}
    with s1b_sft.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            if len(msgs) < 3:
                continue
            user = msgs[1]["content"]
            rat, _ = parse_s1b_assistant(msgs[2]["content"])
            if rat:
                idx[user_hash(user)] = rat
    return idx


def load_rationale_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["key"]] = row["rationale"]
    return out


def append_cache(path: Path, key: str, rationale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "rationale": rationale}, ensure_ascii=False) + "\n")


def cache_key(user: str, cfg: dict[str, Any]) -> str:
    return user_hash(user) + "::" + cfg_key(cfg)


def heuristic_rationale(cfg: dict[str, Any], *, worse: bool) -> str:
    parts = [f"{k}={cfg[k]}" for k in sorted(cfg_only(cfg))]
    summary = ", ".join(parts[:6])
    if len(parts) > 6:
        summary += ", ..."
    if worse:
        return (
            f"This configuration adjusts several hyperparameters ({summary}) "
            "and is expected to underperform the current best trial on validation."
        )
    return (
        f"This configuration ({summary}) extends promising regions from recent trials "
        "while exploring complementary trade-offs in the search space."
    )


_TARGET_CFG_RE = re.compile(
    r"(TARGET CONFIGURATION[^\n]*:\n\n)(\{.*?\})(\n\nWrite exactly)",
    re.DOTALL,
)


def rationale_user_for_config(event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> tuple[str, str]:
    """Same generator contract as s1b SFT rationale step (``rationale_messages``)."""
    msgs = event["rationale_messages"]
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    cfg_json = json.dumps(cfg_only(cfg), ensure_ascii=False)
    if _TARGET_CFG_RE.search(user):
        user = _TARGET_CFG_RE.sub(r"\1" + cfg_json + r"\3", user, count=1)
    else:
        user = user + f"\n\nTARGET CONFIGURATION:\n\n{cfg_json}\n"
    if rejected:
        user = user.replace(
            "this config improved best-so-far",
            "this candidate config is proposed for the next trial",
        )
    return system, user


class QwenRationaleGenerator:
    """Generate rationale text for a fixed config (JSON block is appended separately)."""

    def __init__(self, model_path: Path, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _decode_rationale(self, gen: str, cfg: dict[str, Any]) -> str:
        gen = gen.strip()
        if "Rationale:" in gen:
            gen = gen.split("Rationale:", 1)[1].strip()
        if "JSON:" in gen:
            gen = gen.split("JSON:", 1)[0].strip()
        return gen or heuristic_rationale(cfg, worse=True)

    def generate_from_rationale_messages(self, event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> str:
        import torch

        system, user = rationale_user_for_config(event, cfg, rejected=rejected)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self._decode_rationale(gen, cfg)

    def generate_batch_from_events(
        self,
        events: list[dict[str, Any]],
        configs: list[dict[str, Any]],
        *,
        rejected: bool,
    ) -> list[str]:
        import torch

        if len(events) != len(configs):
            raise ValueError("events and configs length mismatch")
        if not events:
            return []
        prompts: list[str] = []
        for event, cfg in zip(events, configs):
            system, user = rationale_user_for_config(event, cfg, rejected=rejected)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        results: list[str] = []
        for i, cfg in enumerate(configs):
            in_len = int(inputs["attention_mask"][i].sum().item())
            gen = self.tokenizer.decode(out[i][in_len:], skip_special_tokens=True)
            results.append(self._decode_rationale(gen, cfg))
        return results


def _gpu_worker_fill_cache(
    gpu: str,
    jobs: list[dict[str, Any]],
    model_path: str,
    max_new_tokens: int,
    batch_size: int,
    cache_part: str,
) -> int:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gen = QwenRationaleGenerator(Path(model_path), max_new_tokens=max_new_tokens)
    part_path = Path(cache_part)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with part_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i : i + batch_size]
            events = [j["event"] for j in chunk]
            configs = [j["cfg"] for j in chunk]
            rats = gen.generate_batch_from_events(events, configs, rejected=True)
            for job, rat in zip(chunk, rats):
                f.write(json.dumps({"key": job["key"], "rationale": rat}, ensure_ascii=False) + "\n")
                written += 1
    return written


def parallel_fill_rejected_cache(
    jobs: list[dict[str, Any]],
    gpus: list[str],
    *,
    workers_per_gpu: int,
    model_path: Path,
    max_new_tokens: int,
    cache_dir: Path,
) -> dict[str, str]:
    """One model per GPU; each GPU runs micro-batches of size ``workers_per_gpu``."""
    if not jobs:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_gpu = len(gpus)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_gpu)]
    for i, job in enumerate(jobs):
        shards[i % n_gpu].append(job)

    ctx = mp.get_context("spawn")
    payloads = []
    for gi, gpu in enumerate(gpus):
        if not shards[gi]:
            continue
        part = cache_dir / f"rejected_part_gpu{gpu}.jsonl"
        payloads.append(
            (
                gpu,
                shards[gi],
                str(model_path),
                max_new_tokens,
                workers_per_gpu,
                str(part),
            )
        )
    with ctx.Pool(processes=len(payloads)) as pool:
        counts = pool.starmap(_gpu_worker_fill_cache, payloads)
    print(f"[export] parallel cache fill: {sum(counts)} rationales on GPUs {gpus} batch={workers_per_gpu}")

    merged: dict[str, str] = {}
    for _, _, _, _, _, part_s in payloads:
        part = Path(part_s)
        if not part.is_file():
            continue
        with part.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                merged[row["key"]] = row["rationale"]
    return merged


def resolve_rationale(
    *,
    user: str,
    cfg: dict[str, Any],
    event: dict[str, Any] | None,
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    mode: str,
    is_chosen: bool,
) -> tuple[str, str]:
    """Returns (rationale, source_tag)."""
    if is_chosen:
        rat = chosen_index.get(user_hash(user))
        if rat:
            return rat, "s1b_sft"
        if generator is not None and event is not None:
            rat = generator.generate_from_rationale_messages(event, cfg, rejected=False)
            return rat, "qwen_chosen_fallback"
        return heuristic_rationale(cfg, worse=False), "heuristic_chosen"

    key = cache_key(user, cfg)
    if key in cache:
        return cache[key], "cache"

    if mode == "heuristic":
        rat = heuristic_rationale(cfg, worse=True)
        cache[key] = rat
        append_cache(cache_path, key, rat)
        return rat, "heuristic_rejected"

    if generator is None:
        rat = heuristic_rationale(cfg, worse=True)
        return rat, "heuristic_rejected_no_model"

    if event is not None:
        rat = generator.generate_from_rationale_messages(event, cfg, rejected=True)
    else:
        rat = heuristic_rationale(cfg, worse=True)
    cache[key] = rat
    append_cache(cache_path, key, rat)
    return rat, "qwen_rejected"


def enrich_pair(
    pair: dict,
    event: dict[str, Any],
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    reject_mode: str,
) -> dict:
    system = event["s1b_system_prompt"]
    user = event["s1b_user_prompt"]
    pair["conversations"][0]["value"] = system
    pair["conversations"][1]["value"] = user
    meta = pair["metadata"]
    x4 = meta["chosen_config"]
    rej = meta["rejected_config"]

    chosen_rat, chosen_src = resolve_rationale(
        user=user,
        cfg=x4,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=True,
    )
    rejected_rat, rejected_src = resolve_rationale(
        user=user,
        cfg=rej,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=False,
    )

    pair["chosen"]["value"] = format_s1b_assistant(chosen_rat, x4)
    pair["rejected"]["value"] = format_s1b_assistant(rejected_rat, rej)
    meta["chosen_rationale_source"] = chosen_src
    meta["rejected_rationale_source"] = rejected_src
    meta["format"] = "s1b_rationale_json"
    return pair


def collect_pair_jobs(
    args: argparse.Namespace,
    chosen_index: dict[str, str],
) -> tuple[list[tuple[dict, dict]], dict[str, int]]:
    prompts: dict[tuple, dict] = {}
    with args.prompts.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = row["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompts[key] = row

    work: list[tuple[dict, dict]] = []
    skips: dict[str, int] = {}
    hist_cache: dict[str, list[dict]] = {}
    with args.events.open(encoding="utf-8") as f:
        for line in f:
            if args.max_pairs and len(work) >= args.max_pairs:
                break
            event = json.loads(line)
            m = event["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompt_row = prompts.get(key)
            if prompt_row is None:
                skips["missing_prompt"] = skips.get("missing_prompt", 0) + 1
                continue
            pair, reason = build_pair(event, prompt_row, hist_cache)
            if pair is None:
                skips[reason] = skips.get(reason, 0) + 1
                continue
            work.append((event, pair))
    return work, skips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--s1b-sft", type=Path, default=DEFAULT_S1B_SFT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--register-name", type=str, default="p5_s2_dpo_rationale")
    parser.add_argument(
        "--reject-rationale-mode",
        choices=("qwen", "heuristic"),
        default="qwen",
        help="How to fill rejected rationales when not in cache.",
    )
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pair selection + chosen lookup only; rejected=heuristic; no Qwen load.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids (e.g. 0,1,2,3). One Qwen per GPU, batched rejected generation.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Micro-batch size per GPU (default 3 → ~12 concurrent seqs on 4 GPUs).",
    )
    args = parser.parse_args()

    chosen_index = load_chosen_rationale_index(args.s1b_sft)
    cache = load_rationale_cache(args.cache)

    reject_mode = "heuristic" if args.dry_run else args.reject_rationale_mode
    work, skips = collect_pair_jobs(args, chosen_index)

    parallel_gpus = [g.strip() for g in args.parallel_gpus.split(",") if g.strip()]
    use_parallel = bool(parallel_gpus) and not args.dry_run and reject_mode == "qwen"

    generator: QwenRationaleGenerator | None = None
    if not use_parallel and not args.dry_run and reject_mode == "qwen":
        print(f"[export] loading Qwen for rejected rationales: {args.qwen_model}")
        generator = QwenRationaleGenerator(args.qwen_model, max_new_tokens=args.max_new_tokens)

    if use_parallel:
        pending: list[dict[str, Any]] = []
        for event, pair in work:
            user = event["s1b_user_prompt"]
            rej = pair["metadata"]["rejected_config"]
            key = cache_key(user, rej)
            if key not in cache:
                pending.append({"key": key, "event": event, "cfg": rej, "user": user})
        print(
            f"[export] parallel: {len(pending)} rejected to generate on GPUs {parallel_gpus} "
            f"(batch {args.workers_per_gpu}/GPU)"
        )
        cache_dir = args.cache.parent / "p5_s2_dpo_rationale_cache_parts"
        new_entries = parallel_fill_rejected_cache(
            pending,
            parallel_gpus,
            workers_per_gpu=args.workers_per_gpu,
            model_path=args.qwen_model,
            max_new_tokens=args.max_new_tokens,
            cache_dir=cache_dir,
        )
        cache.update(new_entries)
        with args.cache.open("a", encoding="utf-8") as f:
            for key, rat in new_entries.items():
                f.write(json.dumps({"key": key, "rationale": rat}, ensure_ascii=False) + "\n")

    rows: list[dict] = []
    sources_chosen: dict[str, int] = {}
    sources_rejected: dict[str, int] = {}

    for event, pair in work:
        pair = enrich_pair(
            pair,
            event,
            chosen_index,
            cache,
            args.cache,
            generator,
            reject_mode,
        )
        rows.append(pair)
        cs = pair["metadata"]["chosen_rationale_source"]
        rs = pair["metadata"]["rejected_rationale_source"]
        sources_chosen[cs] = sources_chosen.get(cs, 0) + 1
        sources_rejected[rs] = sources_rejected.get(rs, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.train_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_train_row(row), ensure_ascii=False) + "\n")

    manifest = {
        "pairs": len(rows),
        "skips": skips,
        "chosen_rationale_sources": sources_chosen,
        "rejected_rationale_sources": sources_rejected,
        "out": str(args.out),
        "train_out": str(args.train_out),
        "cache": str(args.cache),
        "format": "sharegpt_ranking_dpo_s1b_rationale_json",
        "reject_rationale_mode": reject_mode,
        "dry_run": args.dry_run,
        "parallel_gpus": parallel_gpus,
        "workers_per_gpu": args.workers_per_gpu,
    }
    man_path = args.out.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    register_dataset(args.dataset_info, args.register_name, f"../data/{args.train_out.name}")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.train_out}")
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Export P5 S2 DPO pairs with **s1b-style** chosen/rejected: ``Rationale:`` + ``JSON:``.

Same Pareto pair selection as ``export_p5_s2_dpo_pareto.py`` (pool / worse-than-x4 / mixed sim).

Rough structure
---------------
1. **Pair selection** (unchanged): ``p5_s1b_events.jsonl`` + ``p5_s1a_bc.jsonl`` prompts → (x4, rejected).
2. **Chosen rationale**: lookup from ``p5_s1b_sft_train.jsonl`` by matching the **user** prompt
   (same P5 context as the improvement event).
3. **Rejected rationale**:
   - optional cache file ``p5_s2_dpo_rationale_cache.jsonl`` (key = user_hash + config_key);
   - if missing: **Qwen3.5-9B** generates a short rationale for the fixed rejected config
     (config is not changed — only the explanation text).
4. **Assistant strings** (both arms)::

       Rationale: <text>

       JSON:
       {"config": {...}}

Outputs: ``p5_s2_dpo_rationale.jsonl`` (+ ``_train.jsonl`` for LlamaFactory ranking).

Nothing has been trained on this dataset yet; register as ``p5_s2_dpo_rationale`` for a new DPO run
from ``s1b_ep7p5`` with ``rationale_json`` eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
from pathlib import Path
from typing import Any

from export_p5_s2_dpo_pareto import (
    DEFAULT_DATASET_INFO,
    DEFAULT_EVENTS,
    DEFAULT_PROMPTS,
    build_pair,
    cfg_key,
    cfg_only,
    register_dataset,
    to_train_row,
)

DEFAULT_S1B_SFT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s1b_sft_train.jsonl"
)
DEFAULT_OUT = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale.jsonl"
)
DEFAULT_TRAIN = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_train.jsonl"
)
DEFAULT_CACHE = Path(
    "/root/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/p5_s2_dpo_rationale_cache.jsonl"
)
DEFAULT_MODEL = Path("/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")

_RATIONALE_JSON_RE = re.compile(
    r"Rationale:\s*(.*?)\n\nJSON:\s*(\{.*\})\s*$",
    re.DOTALL,
)


def user_hash(user: str) -> str:
    return hashlib.sha256(user.encode("utf-8")).hexdigest()


def parse_s1b_assistant(text: str) -> tuple[str | None, dict | None]:
    m = _RATIONALE_JSON_RE.match(text.strip())
    if not m:
        return None, None
    try:
        body = json.loads(m.group(2))
    except json.JSONDecodeError:
        return m.group(1).strip(), None
    return m.group(1).strip(), body.get("config")


def format_s1b_assistant(rationale: str, cfg: dict[str, Any]) -> str:
    rat = rationale.strip()
    if rat.lower().startswith("rationale:"):
        rat = rat.split(":", 1)[1].strip()
    cfg_json = json.dumps({"config": cfg_only(cfg)}, ensure_ascii=False)
    return f"Rationale: {rat}\n\nJSON:\n{cfg_json}"


def load_chosen_rationale_index(s1b_sft: Path) -> dict[str, str]:
    """Map sha256(user prompt) -> rationale text for the SFT target (x4) row."""
    idx: dict[str, str] = {}
    with s1b_sft.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            if len(msgs) < 3:
                continue
            user = msgs[1]["content"]
            rat, _ = parse_s1b_assistant(msgs[2]["content"])
            if rat:
                idx[user_hash(user)] = rat
    return idx


def load_rationale_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["key"]] = row["rationale"]
    return out


def append_cache(path: Path, key: str, rationale: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "rationale": rationale}, ensure_ascii=False) + "\n")


def cache_key(user: str, cfg: dict[str, Any]) -> str:
    return user_hash(user) + "::" + cfg_key(cfg)


def heuristic_rationale(cfg: dict[str, Any], *, worse: bool) -> str:
    parts = [f"{k}={cfg[k]}" for k in sorted(cfg_only(cfg))]
    summary = ", ".join(parts[:6])
    if len(parts) > 6:
        summary += ", ..."
    if worse:
        return (
            f"This configuration adjusts several hyperparameters ({summary}) "
            "and is expected to underperform the current best trial on validation."
        )
    return (
        f"This configuration ({summary}) extends promising regions from recent trials "
        "while exploring complementary trade-offs in the search space."
    )


_TARGET_CFG_RE = re.compile(
    r"(TARGET CONFIGURATION[^\n]*:\n\n)(\{.*?\})(\n\nWrite exactly)",
    re.DOTALL,
)


def rationale_user_for_config(event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> tuple[str, str]:
    """Same generator contract as s1b SFT rationale step (``rationale_messages``)."""
    msgs = event["rationale_messages"]
    system = msgs[0]["content"]
    user = msgs[1]["content"]
    cfg_json = json.dumps(cfg_only(cfg), ensure_ascii=False)
    if _TARGET_CFG_RE.search(user):
        user = _TARGET_CFG_RE.sub(r"\1" + cfg_json + r"\3", user, count=1)
    else:
        user = user + f"\n\nTARGET CONFIGURATION:\n\n{cfg_json}\n"
    if rejected:
        user = user.replace(
            "this config improved best-so-far",
            "this candidate config is proposed for the next trial",
        )
    return system, user


class QwenRationaleGenerator:
    """Generate rationale text for a fixed config (JSON block is appended separately)."""

    def __init__(self, model_path: Path, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _decode_rationale(self, gen: str, cfg: dict[str, Any]) -> str:
        gen = gen.strip()
        if "Rationale:" in gen:
            gen = gen.split("Rationale:", 1)[1].strip()
        if "JSON:" in gen:
            gen = gen.split("JSON:", 1)[0].strip()
        return gen or heuristic_rationale(cfg, worse=True)

    def generate_from_rationale_messages(self, event: dict[str, Any], cfg: dict[str, Any], *, rejected: bool) -> str:
        import torch

        system, user = rationale_user_for_config(event, cfg, rejected=rejected)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self._decode_rationale(gen, cfg)

    def generate_batch_from_events(
        self,
        events: list[dict[str, Any]],
        configs: list[dict[str, Any]],
        *,
        rejected: bool,
    ) -> list[str]:
        import torch

        if len(events) != len(configs):
            raise ValueError("events and configs length mismatch")
        if not events:
            return []
        prompts: list[str] = []
        for event, cfg in zip(events, configs):
            system, user = rationale_user_for_config(event, cfg, rejected=rejected)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        results: list[str] = []
        for i, cfg in enumerate(configs):
            in_len = int(inputs["attention_mask"][i].sum().item())
            gen = self.tokenizer.decode(out[i][in_len:], skip_special_tokens=True)
            results.append(self._decode_rationale(gen, cfg))
        return results


def _gpu_worker_fill_cache(
    gpu: str,
    jobs: list[dict[str, Any]],
    model_path: str,
    max_new_tokens: int,
    batch_size: int,
    cache_part: str,
) -> int:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gen = QwenRationaleGenerator(Path(model_path), max_new_tokens=max_new_tokens)
    part_path = Path(cache_part)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with part_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i : i + batch_size]
            events = [j["event"] for j in chunk]
            configs = [j["cfg"] for j in chunk]
            rats = gen.generate_batch_from_events(events, configs, rejected=True)
            for job, rat in zip(chunk, rats):
                f.write(json.dumps({"key": job["key"], "rationale": rat}, ensure_ascii=False) + "\n")
                written += 1
    return written


def parallel_fill_rejected_cache(
    jobs: list[dict[str, Any]],
    gpus: list[str],
    *,
    workers_per_gpu: int,
    model_path: Path,
    max_new_tokens: int,
    cache_dir: Path,
) -> dict[str, str]:
    """One model per GPU; each GPU runs micro-batches of size ``workers_per_gpu``."""
    if not jobs:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_gpu = len(gpus)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_gpu)]
    for i, job in enumerate(jobs):
        shards[i % n_gpu].append(job)

    ctx = mp.get_context("spawn")
    payloads = []
    for gi, gpu in enumerate(gpus):
        if not shards[gi]:
            continue
        part = cache_dir / f"rejected_part_gpu{gpu}.jsonl"
        payloads.append(
            (
                gpu,
                shards[gi],
                str(model_path),
                max_new_tokens,
                workers_per_gpu,
                str(part),
            )
        )
    with ctx.Pool(processes=len(payloads)) as pool:
        counts = pool.starmap(_gpu_worker_fill_cache, payloads)
    print(f"[export] parallel cache fill: {sum(counts)} rationales on GPUs {gpus} batch={workers_per_gpu}")

    merged: dict[str, str] = {}
    for _, _, _, _, _, part_s in payloads:
        part = Path(part_s)
        if not part.is_file():
            continue
        with part.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                merged[row["key"]] = row["rationale"]
    return merged


def resolve_rationale(
    *,
    user: str,
    cfg: dict[str, Any],
    event: dict[str, Any] | None,
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    mode: str,
    is_chosen: bool,
) -> tuple[str, str]:
    """Returns (rationale, source_tag)."""
    if is_chosen:
        rat = chosen_index.get(user_hash(user))
        if rat:
            return rat, "s1b_sft"
        if generator is not None and event is not None:
            rat = generator.generate_from_rationale_messages(event, cfg, rejected=False)
            return rat, "qwen_chosen_fallback"
        return heuristic_rationale(cfg, worse=False), "heuristic_chosen"

    key = cache_key(user, cfg)
    if key in cache:
        return cache[key], "cache"

    if mode == "heuristic":
        rat = heuristic_rationale(cfg, worse=True)
        cache[key] = rat
        append_cache(cache_path, key, rat)
        return rat, "heuristic_rejected"

    if generator is None:
        rat = heuristic_rationale(cfg, worse=True)
        return rat, "heuristic_rejected_no_model"

    if event is not None:
        rat = generator.generate_from_rationale_messages(event, cfg, rejected=True)
    else:
        rat = heuristic_rationale(cfg, worse=True)
    cache[key] = rat
    append_cache(cache_path, key, rat)
    return rat, "qwen_rejected"


def enrich_pair(
    pair: dict,
    event: dict[str, Any],
    chosen_index: dict[str, str],
    cache: dict[str, str],
    cache_path: Path,
    generator: QwenRationaleGenerator | None,
    reject_mode: str,
) -> dict:
    system = event["s1b_system_prompt"]
    user = event["s1b_user_prompt"]
    pair["conversations"][0]["value"] = system
    pair["conversations"][1]["value"] = user
    meta = pair["metadata"]
    x4 = meta["chosen_config"]
    rej = meta["rejected_config"]

    chosen_rat, chosen_src = resolve_rationale(
        user=user,
        cfg=x4,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=True,
    )
    rejected_rat, rejected_src = resolve_rationale(
        user=user,
        cfg=rej,
        event=event,
        chosen_index=chosen_index,
        cache=cache,
        cache_path=cache_path,
        generator=generator,
        mode=reject_mode,
        is_chosen=False,
    )

    pair["chosen"]["value"] = format_s1b_assistant(chosen_rat, x4)
    pair["rejected"]["value"] = format_s1b_assistant(rejected_rat, rej)
    meta["chosen_rationale_source"] = chosen_src
    meta["rejected_rationale_source"] = rejected_src
    meta["format"] = "s1b_rationale_json"
    return pair


def collect_pair_jobs(
    args: argparse.Namespace,
    chosen_index: dict[str, str],
) -> tuple[list[tuple[dict, dict]], dict[str, int]]:
    prompts: dict[tuple, dict] = {}
    with args.prompts.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            m = row["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompts[key] = row

    work: list[tuple[dict, dict]] = []
    skips: dict[str, int] = {}
    hist_cache: dict[str, list[dict]] = {}
    with args.events.open(encoding="utf-8") as f:
        for line in f:
            if args.max_pairs and len(work) >= args.max_pairs:
                break
            event = json.loads(line)
            m = event["metadata"]
            key = (m["task_type"], m["dataset"], m["algorithm"], int(m["improvement_trial"]))
            prompt_row = prompts.get(key)
            if prompt_row is None:
                skips["missing_prompt"] = skips.get("missing_prompt", 0) + 1
                continue
            pair, reason = build_pair(event, prompt_row, hist_cache)
            if pair is None:
                skips[reason] = skips.get(reason, 0) + 1
                continue
            work.append((event, pair))
    return work, skips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--s1b-sft", type=Path, default=DEFAULT_S1B_SFT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--register-name", type=str, default="p5_s2_dpo_rationale")
    parser.add_argument(
        "--reject-rationale-mode",
        choices=("qwen", "heuristic"),
        default="qwen",
        help="How to fill rejected rationales when not in cache.",
    )
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pairs", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pair selection + chosen lookup only; rejected=heuristic; no Qwen load.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids (e.g. 0,1,2,3). One Qwen per GPU, batched rejected generation.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=3,
        help="Micro-batch size per GPU (default 3 → ~12 concurrent seqs on 4 GPUs).",
    )
    args = parser.parse_args()

    chosen_index = load_chosen_rationale_index(args.s1b_sft)
    cache = load_rationale_cache(args.cache)

    reject_mode = "heuristic" if args.dry_run else args.reject_rationale_mode
    work, skips = collect_pair_jobs(args, chosen_index)

    parallel_gpus = [g.strip() for g in args.parallel_gpus.split(",") if g.strip()]
    use_parallel = bool(parallel_gpus) and not args.dry_run and reject_mode == "qwen"

    generator: QwenRationaleGenerator | None = None
    if not use_parallel and not args.dry_run and reject_mode == "qwen":
        print(f"[export] loading Qwen for rejected rationales: {args.qwen_model}")
        generator = QwenRationaleGenerator(args.qwen_model, max_new_tokens=args.max_new_tokens)

    if use_parallel:
        pending: list[dict[str, Any]] = []
        for event, pair in work:
            user = event["s1b_user_prompt"]
            rej = pair["metadata"]["rejected_config"]
            key = cache_key(user, rej)
            if key not in cache:
                pending.append({"key": key, "event": event, "cfg": rej, "user": user})
        print(
            f"[export] parallel: {len(pending)} rejected to generate on GPUs {parallel_gpus} "
            f"(batch {args.workers_per_gpu}/GPU)"
        )
        cache_dir = args.cache.parent / "p5_s2_dpo_rationale_cache_parts"
        new_entries = parallel_fill_rejected_cache(
            pending,
            parallel_gpus,
            workers_per_gpu=args.workers_per_gpu,
            model_path=args.qwen_model,
            max_new_tokens=args.max_new_tokens,
            cache_dir=cache_dir,
        )
        cache.update(new_entries)
        with args.cache.open("a", encoding="utf-8") as f:
            for key, rat in new_entries.items():
                f.write(json.dumps({"key": key, "rationale": rat}, ensure_ascii=False) + "\n")

    rows: list[dict] = []
    sources_chosen: dict[str, int] = {}
    sources_rejected: dict[str, int] = {}

    for event, pair in work:
        pair = enrich_pair(
            pair,
            event,
            chosen_index,
            cache,
            args.cache,
            generator,
            reject_mode,
        )
        rows.append(pair)
        cs = pair["metadata"]["chosen_rationale_source"]
        rs = pair["metadata"]["rejected_rationale_source"]
        sources_chosen[cs] = sources_chosen.get(cs, 0) + 1
        sources_rejected[rs] = sources_rejected.get(rs, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.train_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_train_row(row), ensure_ascii=False) + "\n")

    manifest = {
        "pairs": len(rows),
        "skips": skips,
        "chosen_rationale_sources": sources_chosen,
        "rejected_rationale_sources": sources_rejected,
        "out": str(args.out),
        "train_out": str(args.train_out),
        "cache": str(args.cache),
        "format": "sharegpt_ranking_dpo_s1b_rationale_json",
        "reject_rationale_mode": reject_mode,
        "dry_run": args.dry_run,
        "parallel_gpus": parallel_gpus,
        "workers_per_gpu": args.workers_per_gpu,
    }
    man_path = args.out.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    register_dataset(args.dataset_info, args.register_name, f"../data/{args.train_out.name}")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.train_out}")
    print(f"Wrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
