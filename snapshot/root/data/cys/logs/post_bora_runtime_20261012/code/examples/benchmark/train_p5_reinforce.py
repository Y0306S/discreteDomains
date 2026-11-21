#!/usr/bin/env python3
"""Offline group-normalized REINFORCE from P5 RL rollout JSONL buffers.

Continues LoRA from --init-adapter for one shuffled pass:
  loss = -A * mean_t log π(answer_t | prefix) + β * mean_t KL(π_ref(·|prefix) || π(·|prefix))
The KL runs over the full vocabulary at each recorded answer prefix (``--kl-mode
full_vocab``); ``legacy_proxy`` keeps the old detached proxy for old buffers only.
This is not on-policy clipped GRPO: there is no old-policy ratio.

Advantage modes:
  global         — standardize rewards over the full buffer (R0 default)
  per_dataset    — standardize within each dataset; optionally train
                   dataset-by-dataset (--dataset-sequential)
  grpo_group     — standardize within each group_id (Setup B: 4 siblings)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import shutil
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch.nn.utils import clip_grad_norm_
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

try:
    from .hf_qwen_client import load_verified_adapter
    from .p5_response_format import CHAT_TEMPLATE_VERSION, FORMAT_VERSION, RESPONSE_FORMAT, render_s1b_prompt, validate_rollout_format
except ImportError:
    from hf_qwen_client import load_verified_adapter
    from p5_response_format import CHAT_TEMPLATE_VERSION, FORMAT_VERSION, RESPONSE_FORMAT, render_s1b_prompt, validate_rollout_format

TOKENIZATION_VERSION = "s1b_nothink_answer_eos_v2"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_provenance(args, tokenizer, records):
    files = list(args.buffer_root.rglob("rollouts_*.jsonl"))
    files += [Path(__file__), Path(__file__).with_name("hf_qwen_client.py"),
              Path(__file__).with_name("p5_response_format.py"), args.train_manifest]
    if args.dataset_order_file:
        files.append(args.dataset_order_file)
    for extra in (getattr(args, "task_allowlist", None), getattr(args, "test_manifest", None)):
        if extra:
            files.append(Path(extra))
    for root in (Path(args.base_model), args.init_adapter):
        files.extend(p for p in root.iterdir() if p.is_file() and
                     (p.suffix in (".json", ".safetensors", ".jinja") or p.name == "merges.txt"))
    return {"format_version": 1, "objective_family": "offline_group_normalized_reinforce",
            "kl_objective": "legacy_detached_proxy_not_a_valid_regularizer",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "files_sha256": {str(p.resolve()): file_hash(p) for p in sorted(set(files))},
            "effective_records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "python": platform.python_version(), "torch_cuda": torch.version.cuda,
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "peft", "tokenizers", "safetensors", "numpy")},
            "tokenizer_source": str(tokenizer.name_or_path), "chat_template": tokenizer.chat_template,
            "tokenization_version": TOKENIZATION_VERSION,
            "response_format": getattr(args, "response_format", RESPONSE_FORMAT),
            "response_format_version": FORMAT_VERSION,
            "chat_template_contract": CHAT_TEMPLATE_VERSION,
            "enable_thinking": False, "answer_eos_token_id": tokenizer.eos_token_id,
            "gradient_accumulation": "mean over successful backward passes; flush every partial block"}


def freeze_training_inputs(args, provenance):
    members = [(p, 'buffer/' + str(p.relative_to(args.buffer_root)))
               for p in sorted(args.buffer_root.rglob('rollouts_*.jsonl'))]
    members += [(args.train_manifest, 'train_manifest.jsonl'),
                (Path(__file__), 'code/train_p5_reinforce.py'),
                (Path(__file__).with_name('hf_qwen_client.py'), 'code/hf_qwen_client.py'),
                (Path(__file__).with_name('p5_response_format.py'), 'code/p5_response_format.py')]
    if args.dataset_order_file:
        members.append((args.dataset_order_file, 'dataset_order' + args.dataset_order_file.suffix))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output_dir, prefix='training_inputs_', suffix='.tar.gz', delete=False) as stream:
        archive = Path(stream.name)
    with tarfile.open(archive, 'w:gz') as bundle:
        for path, name in members:
            expected = provenance['files_sha256'][str(path.resolve())]
            if file_hash(path) != expected:
                raise RuntimeError(f'Training input changed before archiving: {path}')
            bundle.add(path, arcname=name, recursive=False)
            if file_hash(path) != expected:
                raise RuntimeError(f'Training input changed while archiving: {path}')
    provenance['input_archive'] = {'path': str(archive.resolve()), 'sha256': file_hash(archive),
                                    'members': [name for _, name in members]}


def load_records(
    buffer_root: Path,
    *,
    train_members_only: bool = True,
    strict_first_n: int | None = None,
    expected_response_format: str | None = None,
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
                validate_rollout_format(row, expected_response_format)
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


TRAINABLE_V14_STATUSES = {"SUCCESS", "PARSE_FAILURE", "DUPLICATE"}
_SPACE_NUMERIC = re.compile(r"^- ([\w:]+): (float|integer), range=\[([-+\d.eE]+), ([-+\d.eE]+)\]", re.M)
_SPACE_CATEGORICAL = re.compile(r"^- ([\w:]+): categorical, choices=(\[.*?\])$", re.M)


def invalid_proposal_reason(row) -> str | None:
    """Why the written configuration violates the search space shown in its own prompt, if it does.

    The live optimizer silently clips or replaces such values before evaluating, so the
    measured score belongs to a different configuration than the text. These count as
    parse failures (fixed penalty), as agreed for v14.
    """
    import ast
    from p5_response_format import parse_response

    user = next((m["content"] for m in row["messages"] if m.get("role") == "user"), "")
    space = {m.group(1): (m.group(2), float(m.group(3)), float(m.group(4))) for m in _SPACE_NUMERIC.finditer(user)}
    space.update({m.group(1): ("categorical", [str(c) for c in ast.literal_eval(m.group(2))])
                  for m in _SPACE_CATEGORICAL.finditer(user)})
    if not space:
        raise SystemExit(f"Search space not found in the prompt of {row.get('group_id')} s{row.get('sample_i')}")
    try:
        _rationale, payload = parse_response(row["completion"])
    except (ValueError, TypeError) as exc:
        return f"unparsable: {exc}"
    config = payload.get("config", payload)
    for key, (kind, *rest) in space.items():
        if key not in config:
            return f"missing {key}"
        value = config[key]
        if kind == "categorical":
            if str(value) not in rest[0]:
                return f"{key}={value!r} not in {rest[0]}"
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key}={value!r} is not numeric"
        if kind == "integer" and float(value) != int(value):
            return f"{key}={value!r} is not an integer"
        if not rest[0] <= float(value) <= rest[1]:
            return f"{key}={value!r} outside [{rest[0]}, {rest[1]}]"
    unknown = [k for k in config if k not in space and k != "algorithm"]
    return f"unknown keys {unknown}" if unknown else None


def _split_triples(path: Path) -> set:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        rows = [json.loads(line)["metadata"] for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    return {(str(r["task_type"]), str(r["dataset"]).lower(), str(r["algorithm"])) for r in rows}


def load_records_v14(
    buffer_root: Path,
    *,
    allowlist: Path,
    train_manifest: Path,
    test_manifest: Path,
    expected_response_format: str,
    parse_fail_penalty: float = 1.0,
    success_reward_floor: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load v14 rollouts, refusing anything that is not provably a clean train member.

    Every row must be the v14 schema, an allow-listed train task absent from the
    test split, with a Dataset-profile label matching its task type. Only marked
    train members with a trainable outcome are kept. Groups need >= 2 members with
    different rewards (group-normalized advantages are zero otherwise).
    """
    from p5_response_format import parse_response

    allowed, train, test = _split_triples(allowlist), _split_triples(train_manifest), _split_triples(test_manifest)
    rows: list[dict[str, Any]] = []
    seen: set = set()
    stats: dict[str, Any] = defaultdict(int)
    files = sorted(buffer_root.rglob("rollouts_*.jsonl"))
    if not files:
        raise SystemExit(f"No rollouts_*.jsonl under {buffer_root}")
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats["rows_read"] += 1
            if row.get("schema") != "rl_rollout.v14":
                raise SystemExit(f"{path}: non-v14 row {row.get('group_id')} {row.get('trial')}")
            key = (str(row["task_type"]), str(row["dataset"]).lower(), str(row["algorithm"]))
            if key not in allowed or key not in train or key in test:
                raise SystemExit(f"{path}: task {key} violates the train/test split")
            ident = (row["group_id"], int(row["sample_i"]))
            if ident in seen:
                raise SystemExit(f"{path}: duplicate rollout {ident}")
            seen.add(ident)
            user = next((m["content"] for m in row["messages"] if m.get("role") == "user"), "")
            labels = set(re.findall(r"\"?task_type\"?\s*:\s*\"?(classification|regression)", user))
            expected = "classification" if row["task_type"] == "cls" else "regression"
            if labels != {expected}:
                raise SystemExit(f"{path}: Dataset profile says {sorted(labels)} for a {expected} task ({ident})")
            validate_rollout_format(row, expected_response_format)
            stats[f"status_{row['evaluation_status']}"] += 1
            if row["evaluation_status"] in ("SUCCESS", "DUPLICATE"):
                reason = invalid_proposal_reason(row)
                if reason is not None:
                    stats["relabelled_invalid_to_parse_failure"] += 1
                    print(f"[REINFORCE] {row['group_id']} s{row['sample_i']}: {reason} -> PARSE_FAILURE")
                    row = dict(row, evaluation_status="PARSE_FAILURE", collected_status=row["evaluation_status"],
                               reward=dict(row["reward"], reward=-float(parse_fail_penalty), relabelled=reason))
            if not row.get("train_member") or row.get("excluded_from_training"):
                continue
            if row["evaluation_status"] not in TRAINABLE_V14_STATUSES:
                raise SystemExit(f"{path}: train member with untrainable status {row['evaluation_status']} ({ident})")
            completion = row.get("completion")
            if not isinstance(completion, str) or not completion.strip():
                raise SystemExit(f"{path}: train member without model text ({ident})")
            if row["evaluation_status"] != "PARSE_FAILURE":
                parse_response(completion)
            reward = float(row["reward"]["reward"])
            if not math.isfinite(reward):
                raise SystemExit(f"{path}: nonfinite reward ({ident})")
            if success_reward_floor is not None and row["evaluation_status"] == "SUCCESS" and reward < success_reward_floor:
                # A valid, evaluated configuration never scores below a parse failure.
                stats["success_rewards_clipped"] += 1
                reward = float(success_reward_floor)
            rows.append({
                "messages": row["messages"],
                "completion": completion,
                "reward": reward,
                "dataset": row["dataset"],
                "task_type": row["task_type"],
                "trial": row.get("trial"),
                "group_id": row["group_id"],
                "status": row["evaluation_status"],
                "generation_limit": row.get("llm_max_tokens"),
            })
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(rows):
        groups[rec["group_id"]].append(i)
    keep: list[int] = []
    for gid, idxs in groups.items():
        distinct = {round(rows[i]["reward"], 12) for i in idxs}
        if len(idxs) < 2 or len(distinct) < 2:
            stats["groups_dropped_no_signal"] += 1
            continue
        keep.extend(idxs)
    stats["groups_kept"] = len(groups) - stats["groups_dropped_no_signal"]
    out = [rows[i] for i in sorted(keep)]
    for rec in out:
        stats[f"train_{rec['status']}"] += 1
    stats["records_kept"] = len(out)
    return out, dict(stats)


def _standardize(vals: torch.Tensor, std_floor: float = 0.0) -> torch.Tensor:
    """(r - mean) / max(std, std_floor). A floor keeps near-tied groups from being
    blown up to full +-1 advantages by tiny (often noise-level) reward differences."""
    adv = vals - vals.mean()
    std = adv.std(unbiased=False).clamp_min(max(1e-6, float(std_floor)))
    return adv / std


def assign_advantages(
    records: list[dict[str, Any]],
    *,
    mode: str,
    std_floor: float = 0.0,
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
            adv = _standardize(rewards, std_floor)
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
    if meta.get("tokenization_version") != TOKENIZATION_VERSION:
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
                if (row.get("ref_logprob") is not None
                        and row.get("ref_logprob_tokenization") == TOKENIZATION_VERSION
                        and row.get("ref_logprob_adapter") == str(ref_adapter)):
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
                if (row.get("ref_logprob") is not None
                        and row.get("ref_logprob_tokenization") == TOKENIZATION_VERSION
                        and row.get("ref_logprob_adapter") == str(ref_adapter)):
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
                row["ref_logprob_tokenization"] = TOKENIZATION_VERSION
                rows_out.append(json.dumps(row, ensure_ascii=False) + "\n")
                changed = True
                n_annotated += 1
        if changed:
            path.write_text("".join(rows_out), encoding="utf-8")
    meta = {
        "ref_adapter": str(ref_adapter),
        "tokenization_version": TOKENIZATION_VERSION,
        "max_seq_len": int(max_seq_len),
        "n_annotated": n_annotated,
        "n_skipped": n_skipped,
    }
    (buffer_root / "ref_logprob_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ref_logprob] annotated={n_annotated} skipped={n_skipped} root={buffer_root}")
    return n_annotated


def build_batch_ids(tokenizer, messages: list[dict], completion: str, device: torch.device, max_seq_len: int,
                    generation_limit: int | None = None):
    """Tokenize prompt + answer. A completion that used the whole generation budget
    was cut off, so it gets no EOS: the model never learns to stop where it was cut."""
    if not completion.strip() or tokenizer.eos_token_id is None:
        raise ValueError("Training requires a nonempty completion and an EOS token")
    prompt_ids = tokenizer(render_s1b_prompt(tokenizer, messages), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    truncated = generation_limit is not None and len(answer_ids) >= int(generation_limit)
    if not truncated and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id):
        answer_ids.append(tokenizer.eos_token_id)
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > max_seq_len:
        return None
    input_ids = torch.tensor([full_ids], device=device)
    labels = input_ids.clone()
    labels[0, : len(prompt_ids)] = -100
    return input_ids, labels


def mean_token_logprob(model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, labels=labels)
    return -out.loss


def _on(x, device):
    """Move tensors to device; leave anything else (e.g. test placeholders) unchanged."""
    return x.to(device) if isinstance(x, torch.Tensor) else x


def _answer_logits(model, input_ids: torch.Tensor, n_answer: int) -> torch.Tensor:
    """Logits that predict the last n_answer tokens: positions L-n-1 .. L-2."""
    keep = n_answer + 1
    try:
        out = model(input_ids=input_ids, logits_to_keep=keep)
    except TypeError:
        out = model(input_ids=input_ids)
    return out.logits[0, -keep:-1]


def answer_logprob_and_kl(policy, ref, input_ids: torch.Tensor, labels: torch.Tensor, chunk: int = 64):
    """Mean answer-token log-prob under the policy and mean per-token KL(ref || policy).

    The KL sums over the full vocabulary at every recorded answer prefix. Reference
    probabilities are frozen; policy log-probabilities stay differentiable, so the
    penalty pulls the policy distribution back toward the reference.
    """
    n_answer = int((labels[0] != -100).sum().item())
    if n_answer <= 0 or not bool((labels[0, -n_answer:] != -100).all()):
        raise ValueError("Answer tokens must be a nonempty suffix")
    targets = input_ids[0, -n_answer:]
    pol_logits = _answer_logits(policy, input_ids, n_answer)
    ref_device = next(ref.parameters()).device
    with torch.no_grad():
        # The frozen reference may live on its own GPU; bring its logits to the policy's device.
        ref_logits = _answer_logits(ref, input_ids.to(ref_device), n_answer).to(pol_logits.device)
    logp_sum = kl_sum = 0.0
    ref_logp_sum = 0.0
    for start in range(0, n_answer, chunk):
        stop = min(start + chunk, n_answer)
        lp = torch.log_softmax(pol_logits[start:stop].float(), dim=-1)
        with torch.no_grad():
            lr = torch.log_softmax(ref_logits[start:stop].float(), dim=-1)
        t = targets[start:stop].unsqueeze(-1)
        logp_sum = logp_sum + lp.gather(-1, t).sum()
        kl_sum = kl_sum + (lr.exp() * (lr - lp)).sum()
        ref_logp_sum += float(lr.gather(-1, t).sum())
    return logp_sum / n_answer, kl_sum / n_answer, ref_logp_sum / n_answer


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
    running_kl = 0.0
    running_dlogp = 0.0
    skipped = 0
    pending = 0
    nonfinite = 0
    if args.grad_accum < 1:
        raise ValueError("grad_accum must be positive")
    scheduler = getattr(args, "_scheduler", None)
    kl_mode = getattr(args, "kl_mode", "legacy_proxy")
    max_nonfinite = int(getattr(args, "max_nonfinite", 0))

    def finish_batch(progress_i):
        nonlocal step, pending, running, running_kl, running_dlogp, nonfinite
        if not pending:
            return
        trainable = [x for x in policy.parameters() if x.requires_grad]
        for parameter in trainable:
            if parameter.grad is not None:
                parameter.grad.div_(pending)
        grad_norm = clip_grad_norm_(trainable, args.max_grad_norm)
        lr_now = opt.param_groups[0]["lr"]
        if not torch.isfinite(grad_norm):
            nonfinite += 1
            print(f"[REINFORCE] WARN nonfinite grad norm at i={progress_i}; update skipped ({nonfinite})")
            if nonfinite > max_nonfinite:
                raise RuntimeError("Too many nonfinite updates")
        else:
            opt.step()
            step += 1
            if scheduler is not None:
                scheduler.step()
        opt.zero_grad(set_to_none=True)
        mean_loss, mean_kl, mean_dlogp = running / pending, running_kl / pending, running_dlogp / pending
        print(f"[REINFORCE] step={step} i={progress_i}/{len(order)} batch={pending} loss={mean_loss:.6f} "
              f"kl={mean_kl:.6f} logp_minus_ref={mean_dlogp:.6f} grad_norm={float(grad_norm):.4f} lr={lr_now:.3e}")
        kl_alert = getattr(args, "kl_alert", None)
        if kl_alert is not None and mean_kl > kl_alert:
            print(f"[REINFORCE] KL ALERT step={step} kl={mean_kl:.6f} > {kl_alert}")
        if wandb_run is not None:
            wandb_run.log({"train/loss": mean_loss, "train/kl": mean_kl, "train/logp_minus_ref": mean_dlogp,
                           "train/grad_norm": float(grad_norm), "train/lr": lr_now,
                           "train/progress": progress_i / max(len(order), 1)})
        pending, running, running_kl, running_dlogp = 0, 0.0, 0.0, 0.0

    for i, idx in enumerate(order, start=1):
        row = records[idx]
        try:
            limit = row.get("generation_limit")
            batch = build_batch_ids(
                tokenizer, row["messages"], row["completion"], device, args.max_seq_len,
                **({"generation_limit": limit} if limit is not None else {}),
            )
        except Exception as e:
            if kl_mode == "full_vocab":
                raise RuntimeError(f"Tokenization failed for {row.get('group_id')} {row.get('trial')}: {e}") from e
            skipped += 1
            print(f"[skip tokenize] {row.get('dataset')} trial={row.get('trial')}: {e}")
            continue
        if batch is None:
            if kl_mode == "full_vocab":
                raise RuntimeError(f"Row exceeds max_seq_len={args.max_seq_len}: {row.get('group_id')} {row.get('trial')}")
            skipped += 1
            continue
        input_ids, labels = batch
        a = row["advantage"]
        if kl_mode == "full_vocab":
            logp, kl, ref_logp = answer_logprob_and_kl(policy, ref, input_ids, labels)
            dlogp = float(logp.detach()) - ref_logp
        else:
            logp = mean_token_logprob(policy, input_ids, labels)
            with torch.no_grad():
                ref_device = next(ref.parameters()).device
                logp_ref = mean_token_logprob(ref, _on(input_ids, ref_device), _on(labels, ref_device))
                logp_ref = _on(logp_ref, logp.device)
            kl = logp.detach() - logp_ref
            dlogp = float(kl)
        loss = -(a * logp) + args.kl_beta * kl
        if not torch.isfinite(loss):
            nonfinite += 1
            print(f"[REINFORCE] WARN nonfinite loss {row.get('group_id')} {row.get('trial')}; row skipped ({nonfinite})")
            if nonfinite > max_nonfinite:
                raise RuntimeError("Too many nonfinite losses")
            continue
        loss.backward()
        pending += 1
        running += float(loss.detach().item())
        running_kl += float(kl.detach().item())
        running_dlogp += dlogp
        if pending == args.grad_accum:
            finish_batch(i)
    finish_batch(len(order))
    return step, skipped


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer-root", type=Path, required=True)
    p.add_argument("--init-adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model", default="/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")
    p.add_argument("--train-manifest", type=Path, default=Path(
        "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_train_80_fixed.jsonl"))
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--kl-beta", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Default: measured longest row. Rows never get silently skipped in v14.")
    p.add_argument("--max-seq-len-cap", type=int, default=32768)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-nonfinite", type=int, default=5)
    p.add_argument("--kl-mode", choices=["full_vocab", "legacy_proxy"], default="full_vocab")
    p.add_argument("--keep-policy-dropout", action="store_true",
                   help="Default disables LoRA dropout so policy and reference agree at step 0.")
    p.add_argument("--rollout-schema", choices=["v14", "legacy"], default="v14")
    p.add_argument("--advantage-std-floor", type=float, default=0.0,
                   help="grpo_group: divide by max(group std, floor). 0 reproduces v14.")
    p.add_argument("--success-reward-floor", type=float, default=None,
                   help="Clip SUCCESS rewards from below (v15: -1, the parse-failure penalty).")
    p.add_argument("--kl-alert", type=float, default=None,
                   help="Log a KL ALERT line whenever a batch's mean KL exceeds this value.")
    p.add_argument("--task-allowlist", type=Path, default=None)
    p.add_argument("--test-manifest", type=Path, default=Path(
        "/root/data/cys/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_48h_20260925/tasks_132.json"))
    p.add_argument("--device", default="cuda:0", help="Policy (trained) device.")
    p.add_argument("--ref-device", default=None, help="Frozen reference device; default: --device.")
    p.add_argument(
        "--advantage-mode",
        choices=["global", "per_dataset", "grpo_group"],
        default="grpo_group",
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
    p.add_argument(
        "--response-format", choices=["json_only", "rationale_json"], default=RESPONSE_FORMAT,
        help="Require rollout metadata and prompts to match this output contract.",
    )
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    load_stats: dict[str, Any] = {}
    if args.rollout_schema == "v14":
        if args.task_allowlist is None:
            raise SystemExit("--rollout-schema v14 requires --task-allowlist")
        if args.dataset_sequential or args.strict_first_n is not None or args.include_non_train_members:
            raise SystemExit("v14 trains shuffled train members only")
        if args.advantage_mode != "grpo_group" or args.kl_mode != "full_vocab":
            raise SystemExit("v14 requires --advantage-mode grpo_group and --kl-mode full_vocab")
        records, load_stats = load_records_v14(
            args.buffer_root,
            allowlist=args.task_allowlist,
            train_manifest=args.train_manifest,
            test_manifest=args.test_manifest,
            expected_response_format=args.response_format,
            success_reward_floor=args.success_reward_floor,
        )
        print(f"[REINFORCE] v14 load: {json.dumps(load_stats, sort_keys=True)}")
    else:
        records = load_records(
            args.buffer_root,
            train_members_only=not args.include_non_train_members,
            strict_first_n=args.strict_first_n,
            expected_response_format=args.response_format,
        )
    if not records:
        raise SystemExit(f"No usable rollout records under {args.buffer_root}")

    assign_advantages(records, mode=args.advantage_mode, std_floor=args.advantage_std_floor)
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
        if getattr(wandb, "run", None) is None:
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

            if getattr(wandb, "run", None) is not None:
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

    tokenizer_source = args.init_adapter if (args.init_adapter / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Measure every row up front: nothing valid is skipped for length.
    lengths = []
    for row in records:
        ids = build_batch_ids(tokenizer, row["messages"], row["completion"], torch.device("cpu"),
                              10 ** 9, generation_limit=row.get("generation_limit"))
        lengths.append(int(ids[0].shape[1]))
    longest = max(lengths)
    if args.max_seq_len is None:
        args.max_seq_len = longest
    if longest > args.max_seq_len or longest > args.max_seq_len_cap:
        raise SystemExit(f"Longest row has {longest} tokens; max_seq_len={args.max_seq_len} cap={args.max_seq_len_cap}")
    load_stats.update(max_tokens=longest, mean_tokens=sum(lengths) / len(lengths))
    print(f"[REINFORCE] token lengths: max={longest} mean={sum(lengths) / len(lengths):.0f} -> max_seq_len={args.max_seq_len}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ref_device = torch.device((args.ref_device or args.device) if torch.cuda.is_available() else "cpu")
    provenance = training_provenance(args, tokenizer, records)
    provenance["load_stats"] = load_stats
    if args.kl_mode == "full_vocab":
        provenance["kl_objective"] = "full_vocabulary_kl_reference_to_policy_on_answer_tokens"
    provenance["lr_schedule"] = f"linear warmup {args.warmup_steps} updates then constant {args.lr}"
    provenance["advantage_std_floor"] = args.advantage_std_floor
    provenance["success_reward_floor"] = args.success_reward_floor
    provenance["policy_lora_dropout_disabled"] = not args.keep_policy_dropout
    freeze_training_inputs(args, provenance)
    verifications = {}

    def load_peft(trainable: bool, device: torch.device) -> PeftModel:
        config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
        model_class = (AutoModelForImageTextToText if config.model_type in ("qwen3_5", "qwen3_5_moe")
                       else AutoModelForCausalLM)
        base = model_class.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        base.to(device)
        m, verification = load_verified_adapter(base, str(args.init_adapter), is_trainable=trainable)
        role = "policy" if trainable else "reference"
        verifications[role] = dict(verification, architecture=type(base).__name__)
        if trainable:
            m.train()
            if not args.keep_policy_dropout:
                for module in m.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0.0
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
        for name in (Path(__file__).name, "hf_qwen_client.py"):
            shutil.copy2(Path(__file__).with_name(name), out_dir / name)
        checkpoint_provenance = dict(provenance, adapter_verification=verifications,
            checkpoint_adapter_sha256=file_hash(out_dir / "adapter_model.safetensors"))
        (out_dir / "training_provenance.json").write_text(json.dumps(checkpoint_provenance, indent=2))
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
    policy = load_peft(True, device)
    ref = load_peft(False, ref_device)
    print(f"[REINFORCE] policy on {device}, frozen reference on {ref_device}")
    opt = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=args.lr)
    warmup = max(1, int(args.warmup_steps))
    args._scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (step + 1) / warmup))

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
