#!/usr/bin/env python3
"""Online GRPO: collect one task with the current adapter, then update weights.

For each task in --dataset-order-file (32 blocks):
  1. Roll out rs_mix v12 schedule (12 prefix jobs) via run_p5_rl_bo_prefix_b.py
  2. GRPO within each job group (4 siblings)
  3. One optimizer step per task using --batch-size groups (default 12 = all jobs)

Arms (via --train-member-policy passed to collect):
  adaptive      — adaptive 3+1 train members (default)
  strict_first4 — first 4 samples only (adaptive_4 style)

Ref policy stays frozen at --ref-adapter (default: --init-adapter).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))

from train_p5_reinforce import (  # noqa: E402
    annotate_ref_logprobs_in_buffer,
    assign_advantages,
    build_batch_ids,
    load_records,
    mean_token_logprob,
    ref_logprob_cache_ready,
)


def load_task_order(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"empty task order: {path}")
    if isinstance(data[0], dict):
        return [
            {
                "task_type": str(t["task_type"]),
                "dataset": str(t["dataset"]),
                "algorithm": str(t["algorithm"]),
            }
            for t in data
        ]
    raise ValueError(f"unsupported order format: {path}")


def _cuda_empty() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _offload_policy_for_collect(policy) -> None:
    """Free GPU memory before the collect subprocess loads its own Qwen copy."""
    policy.cpu()
    _cuda_empty()
    print("[ONLINE] offloaded policy to CPU for collect")


def _reload_policy_for_update(policy, device: torch.device) -> None:
    policy.to(device)
    _cuda_empty()
    print("[ONLINE] reloaded policy on GPU for update")


def _precompute_ref_logprobs(
    *,
    collect_dir: Path,
    policy,
    base_model: str,
    ref_adapter: Path,
    tokenizer,
    device: torch.device,
    max_seq_len: int,
) -> None:
    if ref_logprob_cache_ready(
        collect_dir,
        ref_adapter=ref_adapter,
        max_seq_len=max_seq_len,
        train_members_only=True,
    ):
        print(f"[ONLINE] ref logprobs cached -> {collect_dir}")
        return
    print(f"[ONLINE] precomputing ref logprobs on {device} -> {collect_dir}")
    policy.cpu()
    _cuda_empty()
    ref = load_peft(base_model, ref_adapter, device, False)
    annotate_ref_logprobs_in_buffer(
        collect_dir,
        ref=ref,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        ref_adapter=ref_adapter,
        train_members_only=True,
    )
    del ref
    _cuda_empty()
    print("[ONLINE] ref logprobs done; ref unloaded")


def collect_task_online(
    *,
    py_eval: str,
    task: dict[str, str],
    collect_dir: Path,
    adapter_path: Path,
    rs_version: str,
    train_member_policy: str,
    n_samples: int,
    n_samples_max: int,
    seed: int,
    base_model: str,
    bo_history_root: str,
    time_limit: int,
) -> Path:
    if collect_dir.exists():
        shutil.rmtree(collect_dir)
    collect_dir.mkdir(parents=True, exist_ok=True)
    tt, ds, algo = task["task_type"], task["dataset"], task["algorithm"]
    metric = "bal_acc" if tt == "cls" else "mse"
    cmd = [
        py_eval,
        str(REPO_ROOT / "examples/benchmark/run_p5_rl_bo_prefix_b.py"),
        "--task-type",
        tt,
        "--dataset",
        ds,
        "--algorithm",
        algo,
        "--metric",
        metric,
        "--collect-schedule",
        "rs_mix",
        "--train-member-policy",
        train_member_policy,
        "--rs-version",
        rs_version,
        "--n-samples",
        str(n_samples),
        "--n-samples-max",
        str(n_samples_max),
        "--bo-history-root",
        bo_history_root,
        "--seed",
        str(seed),
        "--time-limit",
        str(time_limit),
        "--output-dir",
        str(collect_dir),
        "--qwen-backend",
        "hf",
        "--qwen-model-path",
        base_model,
        "--qwen-adapter-path",
        str(adapter_path),
        "--llm-response-format",
        "json_only",
        "--llm-max-tokens",
        "768",
        "--llm-temperature",
        "1.0",
    ]
    print(f"[ONLINE] collect {tt}/{ds}/{algo} adapter={adapter_path} policy={train_member_policy}")
    env = os.environ.copy()
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)
    jsonls = list(collect_dir.rglob("rollouts_*.jsonl"))
    if not jsonls:
        raise RuntimeError(f"no rollouts written under {collect_dir}")
    return collect_dir


def group_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_g: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_g[str(row["group_id"])].append(row)
    return [by_g[k] for k in sorted(by_g.keys())]


def update_one_dataset(
    *,
    policy,
    opt,
    tokenizer,
    records: list[dict[str, Any]],
    args,
) -> tuple[int, int]:
    assign_advantages(records, mode="grpo_group")
    groups = group_records(records)
    if not groups:
        return 0, 0
    n_groups = len(groups)
    expected = int(args.batch_size)
    if n_groups != expected:
        print(f"[ONLINE] WARN expected {expected} job groups, got {n_groups}")

    device = next(policy.parameters()).device
    opt.zero_grad(set_to_none=True)
    skipped = 0
    running = 0.0
    denom = max(n_groups, 1)

    for group in groups:
        group_losses: list[torch.Tensor] = []
        for row in group:
            try:
                batch = build_batch_ids(
                    tokenizer, row["messages"], row["completion"], device, args.max_seq_len
                )
            except Exception as e:
                skipped += 1
                print(f"[skip tokenize] {row.get('dataset')}: {e}")
                continue
            if batch is None:
                skipped += 1
                continue
            input_ids, labels = batch
            logp = mean_token_logprob(policy, input_ids, labels)
            ref_lp = row.get("ref_logprob")
            if ref_lp is None:
                skipped += 1
                print(f"[skip missing ref_logprob] {row.get('dataset')} group={row.get('group_id')}")
                continue
            logp_ref = torch.tensor(float(ref_lp), device=device, dtype=logp.dtype)
            kl = logp.detach() - logp_ref
            pg = -(row["advantage"] * logp)
            group_losses.append(pg + args.kl_beta * kl)
        if not group_losses:
            continue
        loss = torch.stack(group_losses).mean() / denom
        loss.backward()
        running += float(loss.detach().item()) * denom

    clip_grad_norm_([x for x in policy.parameters() if x.requires_grad], args.max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
    print(f"[ONLINE] dataset step groups={n_groups} loss~{running:.6f} skipped={skipped}")
    return 1, skipped


def load_peft(base_model: str, adapter: Path, device: torch.device, trainable: bool) -> PeftModel:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base.to(device)
    m = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    if trainable:
        m.train()
        if hasattr(m, "enable_input_require_grads"):
            m.enable_input_require_grads()
        m.gradient_checkpointing_enable()
    else:
        m.eval()
        for param in m.parameters():
            param.requires_grad = False
    return m


def save_adapter(policy, tokenizer, out_dir: Path, meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "online_grpo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ONLINE] saved -> {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-adapter", type=Path, required=True)
    p.add_argument("--ref-adapter", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=None, help="Per-block collect dirs (default: output-dir/online_collect)")
    p.add_argument("--dataset-order-file", type=Path, required=True)
    p.add_argument("--base-model", default="/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")
    p.add_argument("--py-eval", default="/root/blockdata/miniconda3/envs/openbox_qwen_eval/bin/python")
    p.add_argument("--rs-version", choices=["v3", "v4", "v5"], default="v3")
    p.add_argument(
        "--train-member-policy",
        choices=["adaptive", "strict_first4"],
        default="adaptive",
    )
    p.add_argument("--batch-size", type=int, default=12, help="Job groups per optimizer step (= one v12 task).")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--n-samples-max", type=int, default=10)
    p.add_argument("--bo-history-root", default="history_data/history_data")
    p.add_argument("--collect-time-limit", type=int, default=21600)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint-at", type=str, default="8,16,32")
    p.add_argument("--checkpoint-root", type=Path, default=None)
    p.add_argument("--start-block", type=int, default=1, help="Resume from this block (1-based).")
    p.add_argument(
        "--reuse-collect",
        action="store_true",
        help="If block collect dir already has rollouts, skip re-collect.",
    )
    args = p.parse_args()

    ref_adapter = args.ref_adapter or args.init_adapter
    work_dir = args.work_dir or (args.output_dir / "online_collect")
    work_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = load_task_order(args.dataset_order_file)
    ckpt_blocks = {int(x.strip()) for x in str(args.checkpoint_at).split(",") if x.strip()}
    ckpt_root = args.checkpoint_root or args.output_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[ONLINE] init={args.init_adapter} ref={ref_adapter} tasks={len(tasks)} batch_size={args.batch_size}")
    print(f"[ONLINE] collect_policy={args.train_member_policy} rs={args.rs_version}")

    adapter_scratch = args.output_dir / "_adapter_current"
    resume_adapter = args.init_adapter
    if int(args.start_block) > 1 and adapter_scratch.is_dir() and (
        adapter_scratch / "adapter_model.safetensors"
    ).is_file():
        resume_adapter = adapter_scratch
        print(f"[ONLINE] resume from {resume_adapter} at block>={args.start_block}")
    elif int(args.start_block) > 1:
        raise SystemExit(f"--start-block={args.start_block} but missing {adapter_scratch}")

    policy = load_peft(args.base_model, resume_adapter, device, True)
    print(f"[ONLINE] policy on {device}; ref logprobs precomputed at collect time (not loaded during update)")
    opt = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=args.lr)

    if int(args.start_block) <= 1:
        if adapter_scratch.exists():
            shutil.rmtree(adapter_scratch)
        save_adapter(
            policy,
            tokenizer,
            adapter_scratch,
            {"note": "scratch copy for online collect handoff"},
        )
    elif not adapter_scratch.is_dir():
        save_adapter(
            policy,
            tokenizer,
            adapter_scratch,
            {"note": "restored scratch adapter on resume"},
        )

    total_steps = 0
    total_skipped = 0
    saved_ckpts: set[int] = set()

    for block_i, task in enumerate(tasks, start=1):
        if block_i < int(args.start_block):
            print(f"[ONLINE] skip block {block_i} (< start_block={args.start_block})")
            continue
        ds = task["dataset"]
        collect_dir = work_dir / f"block_{block_i:02d}_{task['task_type']}_{ds}_{task['algorithm']}"
        has_rollouts = bool(list(collect_dir.rglob("rollouts_*.jsonl"))) if collect_dir.is_dir() else False
        if args.reuse_collect and has_rollouts:
            print(f"[ONLINE] reuse collect block {block_i} {ds} -> {collect_dir}")
        else:
            _offload_policy_for_collect(policy)
            collect_task_online(
                py_eval=args.py_eval,
                task=task,
                collect_dir=collect_dir,
                adapter_path=adapter_scratch,
                rs_version=args.rs_version,
                train_member_policy=args.train_member_policy,
                n_samples=args.n_samples,
                n_samples_max=args.n_samples_max,
                seed=args.seed,
                base_model=args.base_model,
                bo_history_root=args.bo_history_root,
                time_limit=args.collect_time_limit,
            )
        _precompute_ref_logprobs(
            collect_dir=collect_dir,
            policy=policy,
            base_model=args.base_model,
            ref_adapter=ref_adapter,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=args.max_seq_len,
        )
        _reload_policy_for_update(policy, device)
        records = load_records(collect_dir, train_members_only=True)
        if not records:
            raise RuntimeError(f"no train_member records after collect block {block_i} {ds}")

        steps, skipped = update_one_dataset(
            policy=policy,
            opt=opt,
            tokenizer=tokenizer,
            records=records,
            args=args,
        )
        total_steps += steps
        total_skipped += skipped

        if adapter_scratch.exists():
            shutil.rmtree(adapter_scratch)
        save_adapter(
            policy,
            tokenizer,
            adapter_scratch,
            {
                "block": block_i,
                "dataset": ds,
                "n_records": len(records),
                "train_member_policy": args.train_member_policy,
            },
        )

        if block_i in ckpt_blocks and block_i not in saved_ckpts:
            save_adapter(
                policy,
                tokenizer,
                ckpt_root / f"ckpt{block_i}",
                {
                    "checkpoint_block": block_i,
                    "datasets_so_far": [t["dataset"] for t in tasks[:block_i]],
                    "train_member_policy": args.train_member_policy,
                    "recipe": "online_grpo_b12",
                },
            )
            saved_ckpts.add(block_i)

    save_adapter(
        policy,
        tokenizer,
        args.output_dir,
        {
            "recipe": "online_grpo_b12",
            "train_member_policy": args.train_member_policy,
            "batch_size": args.batch_size,
            "n_tasks": len(tasks),
            "total_steps": total_steps,
            "total_skipped": total_skipped,
            "init_adapter": str(args.init_adapter),
            "ref_adapter": str(ref_adapter),
        },
    )
    print(f"[ONLINE] done steps={total_steps} skipped={total_skipped} out={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
