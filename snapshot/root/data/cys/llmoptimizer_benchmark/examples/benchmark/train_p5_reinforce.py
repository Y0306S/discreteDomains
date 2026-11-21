#!/usr/bin/env python3
"""REINFORCE / GRPO-style updates from P5 RL rollout JSONL buffers.

Continues LoRA from --init-adapter for one pass:
  loss = -A * mean_log π(completion|prompt) + β * KL_proxy

Advantage modes:
  global         — standardize rewards over the full buffer (R0 default)
  per_dataset    — standardize within each dataset; optionally train
                   dataset-by-dataset (--dataset-sequential)
  grpo_group     — standardize within each group_id (Setup B: 4 siblings)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_records(
    buffer_root: Path,
    *,
    train_members_only: bool = True,
    strict_first_n: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_skipped_non_train = 0
    n_skipped_strict = 0
    for path in sorted(buffer_root.rglob("rollouts_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                messages = row.get("messages")
                completion = row.get("completion")
                reward = row.get("reward") or {}
                if not isinstance(messages, list) or not messages:
                    continue
                if not isinstance(completion, str) or not completion.strip():
                    continue
                if not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
                    continue
                try:
                    r = float(reward.get("reward"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(r):
                    continue
                if strict_first_n is not None:
                    try:
                        sample_i = int(row.get("sample_i", 0))
                    except (TypeError, ValueError):
                        sample_i = 0
                    if sample_i >= int(strict_first_n):
                        n_skipped_strict += 1
                        continue
                elif train_members_only and "train_member" in row and not bool(row.get("train_member")):
                    n_skipped_non_train += 1
                    continue
                dataset = row.get("dataset") or "unknown"
                group_id = row.get("group_id") or f"{dataset}:{row.get('trial')}"
                rec: dict[str, Any] = {
                    "messages": messages,
                    "completion": completion,
                    "reward": r,
                    "dataset": dataset,
                    "trial": row.get("trial"),
                    "group_id": group_id,
                }
                ref_lp = row.get("ref_logprob")
                if ref_lp is not None:
                    try:
                        rec["ref_logprob"] = float(ref_lp)
                    except (TypeError, ValueError):
                        pass
                rows.append(rec)
    if n_skipped_non_train:
        print(f"[REINFORCE] skipped {n_skipped_non_train} non-train_member records")
    if n_skipped_strict:
        print(f"[REINFORCE] skipped {n_skipped_strict} records outside strict_first_n={strict_first_n}")
    return rows


def _standardize(vals: torch.Tensor) -> torch.Tensor:
    adv = vals - vals.mean()
    std = adv.std(unbiased=False).clamp_min(1e-6)
    return adv / std


def assign_advantages(
    records: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    if not records:
        return
    if mode == "global":
        rewards = torch.tensor([r["reward"] for r in records], dtype=torch.float32)
        adv = _standardize(rewards)
        for i, row in enumerate(records):
            row["advantage"] = float(adv[i].item())
        return

    if mode == "per_dataset":
        by_ds: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(records):
            by_ds[str(row["dataset"])].append(i)
        for idxs in by_ds.values():
            rewards = torch.tensor([records[i]["reward"] for i in idxs], dtype=torch.float32)
            adv = _standardize(rewards)
            for j, i in enumerate(idxs):
                records[i]["advantage"] = float(adv[j].item())
        return

    if mode == "grpo_group":
        by_g: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(records):
            by_g[str(row["group_id"])].append(i)
        for idxs in by_g.values():
            rewards = torch.tensor([records[i]["reward"] for i in idxs], dtype=torch.float32)
            if len(idxs) == 1:
                records[idxs[0]]["advantage"] = 0.0
                continue
            adv = _standardize(rewards)
            for j, i in enumerate(idxs):
                records[i]["advantage"] = float(adv[j].item())
        return

    raise ValueError(f"Unknown advantage mode: {mode}")


def ref_logprob_cache_ready(
    buffer_root: Path,
    *,
    ref_adapter: Path,
    max_seq_len: int,
    train_members_only: bool = True,
) -> bool:
    meta_path = buffer_root / "ref_logprob_meta.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if str(meta.get("ref_adapter")) != str(ref_adapter):
        return False
    if int(meta.get("max_seq_len", -1)) != int(max_seq_len):
        return False
    n_needed = 0
    n_have = 0
    for path in sorted(buffer_root.rglob("rollouts_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if train_members_only and "train_member" in row and not bool(row.get("train_member")):
                    continue
                n_needed += 1
                if row.get("ref_logprob") is not None:
                    n_have += 1
    return n_needed > 0 and n_have == n_needed


def annotate_ref_logprobs_in_buffer(
    buffer_root: Path,
    *,
    ref,
    tokenizer,
    max_seq_len: int,
    ref_adapter: Path,
    train_members_only: bool = True,
) -> int:
    """Write ref_logprob into rollout JSONL for train_member rows."""
    device = next(ref.parameters()).device
    n_annotated = 0
    n_skipped = 0
    for path in sorted(buffer_root.rglob("rollouts_*.jsonl")):
        rows_out: list[str] = []
        changed = False
        with path.open(encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                if train_members_only and "train_member" in row and not bool(row.get("train_member")):
                    rows_out.append(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                if row.get("ref_logprob") is not None:
                    rows_out.append(json.dumps(row, ensure_ascii=False) + "\n")
                    continue
                messages = row.get("messages")
                completion = row.get("completion")
                if not isinstance(messages, list) or not isinstance(completion, str):
                    rows_out.append(json.dumps(row, ensure_ascii=False) + "\n")
                    n_skipped += 1
                    continue
                try:
                    batch = build_batch_ids(tokenizer, messages, completion, device, max_seq_len)
                except Exception as e:
                    print(f"[ref_logprob] skip tokenize {row.get('dataset')}: {e}")
                    rows_out.append(json.dumps(row, ensure_ascii=False) + "\n")
                    n_skipped += 1
                    continue
                if batch is None:
                    rows_out.append(json.dumps(row, ensure_ascii=False) + "\n")
                    n_skipped += 1
                    continue
                input_ids, labels = batch
                with torch.no_grad():
                    row["ref_logprob"] = float(mean_token_logprob(ref, input_ids, labels).item())
                row["ref_logprob_adapter"] = str(ref_adapter)
                rows_out.append(json.dumps(row, ensure_ascii=False) + "\n")
                changed = True
                n_annotated += 1
        if changed:
            path.write_text("".join(rows_out), encoding="utf-8")
    meta = {
        "ref_adapter": str(ref_adapter),
        "max_seq_len": int(max_seq_len),
        "n_annotated": n_annotated,
        "n_skipped": n_skipped,
    }
    (buffer_root / "ref_logprob_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ref_logprob] annotated={n_annotated} skipped={n_skipped} root={buffer_root}")
    return n_annotated


def build_batch_ids(tokenizer, messages: list[dict], completion: str, device: torch.device, max_seq_len: int):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full = prompt + completion
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        full_ids = prompt_ids + tokenizer(completion, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len:
        return None
    input_ids = torch.tensor([full_ids], device=device)
    labels = input_ids.clone()
    labels[0, : len(prompt_ids)] = -100
    return input_ids, labels


def mean_token_logprob(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, labels=labels)
    return -out.loss


def train_on_order(
    *,
    policy,
    ref,
    opt,
    tokenizer,
    records: list[dict[str, Any]],
    order: list[int],
    args,
    wandb_run=None,
) -> tuple[int, int]:
    device = next(policy.parameters()).device
    opt.zero_grad(set_to_none=True)
    step = 0
    running = 0.0
    skipped = 0
    for i, idx in enumerate(order, start=1):
        row = records[idx]
        try:
            batch = build_batch_ids(
                tokenizer, row["messages"], row["completion"], device, args.max_seq_len
            )
        except Exception as e:
            skipped += 1
            print(f"[skip tokenize] {row.get('dataset')} trial={row.get('trial')}: {e}")
            continue
        if batch is None:
            skipped += 1
            continue
        input_ids, labels = batch
        logp = mean_token_logprob(policy, input_ids, labels)
        with torch.no_grad():
            logp_ref = mean_token_logprob(ref, input_ids, labels)
        kl = logp.detach() - logp_ref
        a = row["advantage"]
        pg = -(a * logp)
        loss = (pg + args.kl_beta * kl) / args.grad_accum
        loss.backward()
        running += float(loss.detach().item()) * args.grad_accum

        if i % args.grad_accum == 0 or i == len(order):
            clip_grad_norm_([x for x in policy.parameters() if x.requires_grad], args.max_grad_norm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            print(f"[REINFORCE] step={step} i={i}/{len(order)} loss~{running:.6f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": running,
                        "train/progress": i / max(len(order), 1),
                    },
                    step=step,
                )
            running = 0.0
    return step, skipped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer-root", type=Path, required=True)
    p.add_argument("--init-adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model", default="/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--advantage-mode",
        choices=["global", "per_dataset", "grpo_group"],
        default="global",
    )
    p.add_argument(
        "--dataset-sequential",
        action="store_true",
        help="Train one dataset fully before the next (works with per_dataset and grpo_group).",
    )
    p.add_argument(
        "--dataset-order-file",
        type=Path,
        default=None,
        help="Optional text/JSON file listing dataset names in train order (overrides alpha sort).",
    )
    p.add_argument(
        "--mid-checkpoint-dir",
        type=Path,
        default=None,
        help="If set with --dataset-sequential, save adapter here after --mid-checkpoint-after datasets.",
    )
    p.add_argument(
        "--mid-checkpoint-after",
        type=int,
        default=8,
        help="Number of dataset blocks before writing --mid-checkpoint-dir (default 8).",
    )
    p.add_argument(
        "--checkpoint-at",
        type=str,
        default=None,
        help="Comma-separated dataset-block indices to save (e.g. 12,14,16). Requires --checkpoint-root.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="Root dir for multi mid-checkpoints; writes {root}/ckpt{N}/ for each --checkpoint-at N.",
    )
    p.add_argument(
        "--checkpoint-label-offset",
        type=int,
        default=0,
        help="Add to block index when naming ckpt dirs (Phase2: train 8 blocks labeled 18..24 with offset=16).",
    )
    p.add_argument(
        "--include-non-train-members",
        action="store_true",
        help="If set, use every JSONL record. Default: drop rows with train_member=false.",
    )
    p.add_argument(
        "--strict-first-n",
        type=int,
        default=None,
        help="Keep only records with sample_i < N (ignores train_member flags).",
    )
    p.add_argument(
        "--strict-dataset-order",
        action="store_true",
        help="With --dataset-order-file: do not append leftover datasets missing from the order file.",
    )
    p.add_argument("--wandb-project", default=None, help="W&B project (e.g. llmoptimizer). Omit to disable.")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (optional).")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name (optional).")
    p.add_argument("--wandb-group", default=None, help="W&B group, e.g. sweep id (optional).")
    p.add_argument(
        "--wandb-tags",
        default=None,
        help="Comma-separated W&B tags (optional).",
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_records(
        args.buffer_root,
        train_members_only=not args.include_non_train_members,
        strict_first_n=args.strict_first_n,
    )
    if not records:
        raise SystemExit(f"No usable rollout records under {args.buffer_root}")

    assign_advantages(records, mode=args.advantage_mode)
    rewards = torch.tensor([r["reward"] for r in records], dtype=torch.float32)
    print(
        f"[REINFORCE] records={len(records)} mode={args.advantage_mode} "
        f"seq_by_ds={args.dataset_sequential} "
        f"reward_mean={float(rewards.mean()):.4f} reward_std={float(rewards.std()):.4f}"
    )

    wandb_run = None
    if args.wandb_project:
        import wandb

        tags = [t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()]
        if wandb.run is None:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                group=args.wandb_group,
                tags=tags or None,
                config={
                    "lr": args.lr,
                    "kl_beta": args.kl_beta,
                    "grad_accum": args.grad_accum,
                    "max_grad_norm": args.max_grad_norm,
                    "seed": args.seed,
                    "advantage_mode": args.advantage_mode,
                    "dataset_sequential": bool(args.dataset_sequential),
                    "buffer_root": str(args.buffer_root),
                    "init_adapter": str(args.init_adapter),
                    "n_records": len(records),
                    "reward_mean": float(rewards.mean()),
                    "reward_std": float(rewards.std()),
                },
            )
        else:
            wandb_run = wandb.run
        wandb_run.log(
            {
                "data/n_records": len(records),
                "data/reward_mean": float(rewards.mean()),
                "data/reward_std": float(rewards.std()),
            }
        )
    else:
        try:
            import wandb

            if wandb.run is not None:
                wandb_run = wandb.run
                wandb_run.config.update(
                    {
                        "lr": args.lr,
                        "kl_beta": args.kl_beta,
                        "grad_accum": args.grad_accum,
                        "max_grad_norm": args.max_grad_norm,
                        "seed": args.seed,
                        "advantage_mode": args.advantage_mode,
                        "dataset_sequential": bool(args.dataset_sequential),
                        "buffer_root": str(args.buffer_root),
                        "init_adapter": str(args.init_adapter),
                        "n_records": len(records),
                        "reward_mean": float(rewards.mean()),
                        "reward_std": float(rewards.std()),
                    },
                    allow_val_change=True,
                )
                wandb_run.log(
                    {
                        "data/n_records": len(records),
                        "data/reward_mean": float(rewards.mean()),
                        "data/reward_std": float(rewards.std()),
                    }
                )
        except ImportError:
            pass

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def load_peft(trainable: bool) -> PeftModel:
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        base.to(device)
        m = PeftModel.from_pretrained(base, str(args.init_adapter), is_trainable=trainable)
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

    def save_adapter(out_dir: Path, *, note: str, extra_meta: dict[str, Any] | None = None) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        meta = {
            "init_adapter": str(args.init_adapter),
            "buffer_root": str(args.buffer_root),
            "n_records": len(records),
            "n_skipped": total_skipped,
            "lr": args.lr,
            "kl_beta": args.kl_beta,
            "grad_accum": args.grad_accum,
            "seed": args.seed,
            "advantage_mode": args.advantage_mode,
            "dataset_sequential": bool(args.dataset_sequential),
            "note": note,
        }
        if extra_meta:
            meta.update(extra_meta)
        (out_dir / "reinforce_train_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[REINFORCE] saved adapter -> {out_dir} ({note})")

    def load_dataset_order(path: Path | None, available: list[str]) -> list[str]:
        if path is None:
            return sorted(available)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return sorted(available)
        if path.suffix == ".json":
            data = json.loads(text)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                ordered = [str(x["dataset"]) for x in data]
            else:
                ordered = [str(x) for x in data]
        else:
            ordered = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        avail_set = set(available)
        out = [d for d in ordered if d in avail_set]
        missing = [d for d in ordered if d not in avail_set]
        extra = sorted(avail_set - set(out))
        if missing:
            print(f"[REINFORCE] WARN order-file datasets missing from buffer: {missing}")
        if extra and not args.strict_dataset_order:
            print(f"[REINFORCE] appending unordered leftover datasets: {extra}")
            out.extend(extra)
        elif extra and args.strict_dataset_order:
            print(f"[REINFORCE] strict order: ignoring leftover datasets: {extra}")
        return out

    ckpt_blocks: set[int] = set()
    if args.checkpoint_at:
        ckpt_blocks = {int(x.strip()) for x in str(args.checkpoint_at).split(",") if x.strip()}
        if args.checkpoint_root is None:
            raise SystemExit("--checkpoint-at requires --checkpoint-root")
    if args.checkpoint_root is not None and not ckpt_blocks:
        raise SystemExit("--checkpoint-root requires --checkpoint-at")

    print("[REINFORCE] loading policy + frozen ref (same init adapter)...")
    policy = load_peft(True)
    ref = load_peft(False)
    opt = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=args.lr)

    total_skipped = 0
    mid_saved = False
    saved_ckpt_labels: set[int] = set()
    if args.dataset_sequential:
        by_ds: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(records):
            by_ds[str(row["dataset"])].append(i)
        ds_order = load_dataset_order(args.dataset_order_file, list(by_ds.keys()))
        print(f"[REINFORCE] dataset order ({len(ds_order)}): {ds_order}")
        for block_i, ds in enumerate(ds_order, start=1):
            idxs = by_ds[ds]
            order = list(idxs)
            random.shuffle(order)
            print(f"[REINFORCE] dataset-sequential block {block_i}/{len(ds_order)} dataset={ds} n={len(order)}")
            _, skipped = train_on_order(
                policy=policy,
                ref=ref,
                opt=opt,
                tokenizer=tokenizer,
                records=records,
                order=order,
                args=args,
                wandb_run=wandb_run,
            )
            total_skipped += skipped
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/dataset_block": block_i,
                        "train/dataset_name": ds,
                        "train/skipped_total": total_skipped,
                    },
                    step=block_i,
                )
            if (
                args.mid_checkpoint_dir is not None
                and not mid_saved
                and block_i >= int(args.mid_checkpoint_after)
            ):
                save_adapter(
                    args.mid_checkpoint_dir,
                    note=f"mid-checkpoint after {block_i} datasets",
                    extra_meta={
                        "mid_checkpoint_after": int(args.mid_checkpoint_after),
                        "datasets_so_far": ds_order[:block_i],
                    },
                )
                mid_saved = True
            label = block_i + int(args.checkpoint_label_offset)
            if ckpt_blocks and label in ckpt_blocks and label not in saved_ckpt_labels:
                assert args.checkpoint_root is not None
                save_adapter(
                    args.checkpoint_root / f"ckpt{label}",
                    note=f"checkpoint after block {block_i} (label={label})",
                    extra_meta={
                        "checkpoint_label": label,
                        "block_index": block_i,
                        "checkpoint_label_offset": int(args.checkpoint_label_offset),
                        "datasets_so_far": ds_order[:block_i],
                    },
                )
                saved_ckpt_labels.add(label)
    else:
        order = list(range(len(records)))
        random.shuffle(order)
        _, total_skipped = train_on_order(
            policy=policy,
            ref=ref,
            opt=opt,
            tokenizer=tokenizer,
            records=records,
            order=order,
            args=args,
            wandb_run=wandb_run,
        )

    save_adapter(args.output_dir, note="final")
    if wandb_run is not None:
        wandb_run.summary["train/n_skipped"] = total_skipped
        wandb_run.summary["train/n_records"] = len(records)
        if not args.wandb_project:
            history = wandb_run.history(keys=["train/loss"], pandas=False)
            if history:
                last_loss = history[-1].get("train/loss")
                if last_loss is not None:
                    wandb_run.summary["train/final_loss"] = float(last_loss)
        if args.wandb_project:
            wandb_run.finish()
    if args.mid_checkpoint_dir is not None and not mid_saved:
        print(
            f"[REINFORCE] WARN mid-checkpoint requested after "
            f"{args.mid_checkpoint_after} datasets but only finished fewer blocks"
        )
    if ckpt_blocks:
        missing = sorted(ckpt_blocks - saved_ckpt_labels)
        if missing:
            print(f"[REINFORCE] WARN requested checkpoints not saved: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
