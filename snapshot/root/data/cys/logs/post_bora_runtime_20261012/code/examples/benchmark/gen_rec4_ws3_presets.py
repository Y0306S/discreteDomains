#!/usr/bin/env python3
"""Generate 3 lora_recommended4 warm-start configs per ml_test_20_fixed slot (temp 0.7)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


EVAL_CLI = Path(
    "/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_config_cli.py"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shard_records(records: list[dict], n_shards: int, shard_dir: Path) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards: list[list[dict]] = [[] for _ in range(n_shards)]
    for i, rec in enumerate(records):
        rec = dict(rec)
        rec["_global_index"] = i
        shards[i % n_shards].append(rec)
    paths = []
    for i, rows in enumerate(shards):
        p = shard_dir / f"shard_{i:02d}.jsonl"
        write_jsonl(p, rows)
        paths.append(p)
    return paths


def gpu_free_mb(gpu: int) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(out.strip().split()[0])


def launch_shards(args: argparse.Namespace, shard_paths: list[Path], pred_dir: Path) -> list[Path]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    n_gpus = int(args.num_gpus)
    jobs_per_gpu = int(args.jobs_per_gpu)
    n_slots = n_gpus * jobs_per_gpu
    slot_pid: dict[int, subprocess.Popen | None] = {i: None for i in range(n_slots)}
    slot_pred: dict[int, Path] = {}
    queue = list(enumerate(shard_paths))
    pred_paths: list[Path] = []
    launched = 0
    total = len(queue)

    def slot_alive(slot: int) -> bool:
        proc = slot_pid.get(slot)
        return proc is not None and proc.poll() is None

    while queue or any(slot_alive(s) for s in slot_pid):
        for slot in range(n_slots):
            proc = slot_pid[slot]
            if proc is not None and proc.poll() is not None:
                rc = proc.returncode
                if rc != 0:
                    raise SystemExit(f"shard worker slot={slot} failed rc={rc}")
                slot_pid[slot] = None
            if slot_pid[slot] is not None:
                continue
            if not queue:
                continue
            gpu = slot // jobs_per_gpu
            occupied = sum(
                1
                for s in range(gpu * jobs_per_gpu, (gpu + 1) * jobs_per_gpu)
                if slot_alive(s)
            )
            if occupied >= 1 and gpu_free_mb(gpu) < int(args.pack_min_free_mb):
                continue
            shard_i, shard_path = queue.pop(0)
            pred_path = pred_dir / f"pred_shard_{shard_i:02d}.jsonl"
            log_path = log_dir / f"rec4_gen_shard_{shard_i:02d}.log"
            cmd = [
                args.python,
                str(EVAL_CLI),
                "--generate",
                "qwen",
                "--dataset-jsonl",
                str(shard_path),
                "--predictions-out",
                str(pred_path),
                "--benchmark-root",
                args.benchmark_root,
                "--base-model",
                args.base_model,
                "--adapter-dir",
                args.adapter_dir,
                "--model-label",
                "lora_recommended4",
                "--num-samples",
                str(args.num_samples),
                "--temperature",
                str(args.temperature),
                "--top-p",
                str(args.top_p),
                "--max-new-tokens",
                str(args.max_new_tokens),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            print(
                f"[gen] launch shard={shard_i} gpu={gpu} slot={slot} "
                f"records={sum(1 for _ in shard_path.open() if _.strip())}",
                flush=True,
            )
            lf = open(log_path, "w")
            slot_pid[slot] = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            slot_pred[slot] = pred_path
            pred_paths.append(pred_path)
            launched += 1
        time.sleep(5)

    print(f"[gen] finished {launched}/{total} shards", flush=True)
    return pred_paths


def configs_from_preds(pred_paths: list[Path], records: list[dict]) -> dict[str, list[dict]]:
    by_key: dict[str, list[dict]] = {}
    n_parse_fail = 0
    for path in pred_paths:
        if not path.exists():
            continue
        for pred in load_jsonl(path):
            idx = int(pred["index"])
            rec = records[idx]
            meta = rec["metadata"]
            key = f"{meta['dataset']}:{meta['algorithm']}"
            cfg = pred.get("config")
            if not isinstance(cfg, dict):
                n_parse_fail += 1
                continue
            cfg = dict(cfg)
            cfg.setdefault("algorithm", meta["algorithm"])
            by_key.setdefault(key, []).append(cfg)
    print(f"[gen] parse_fail={n_parse_fail} keys={len(by_key)}", flush=True)
    return by_key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path("/root/data/cys/LLMOptimizer/LLMOptimizer/bolt_qwen9b_finetune/data/ml_test_20_fixed.jsonl"),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("/root/blockdata/llmoptimizer_benchmark/benchmark_data/ml_test_20_fixed_ckpt32_rec4_ws3/rec4_ws3_presets.json"),
    )
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--python", default="/root/blockdata/miniconda3/envs/openbox_qwen_eval/bin/python")
    p.add_argument("--benchmark-root", default="/root/blockdata/llmoptimizer_benchmark")
    p.add_argument("--base-model", default="/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")
    p.add_argument(
        "--adapter-dir",
        default="/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_recommended4",
    )
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--jobs-per-gpu", type=int, default=3)
    p.add_argument("--pack-min-free-mb", type=int, default=18000)
    args = p.parse_args()

    work = args.work_dir or args.out_json.parent / "rec4_ws3_gen"
    args.log_dir = args.log_dir or work / "logs"
    records = load_jsonl(args.dataset_jsonl)
    print(f"[gen] records={len(records)} samples={args.num_samples} temp={args.temperature}", flush=True)
    n_shards = int(args.num_gpus) * int(args.jobs_per_gpu)
    shard_paths = shard_records(records, n_shards, work / "shards")
    pred_paths = launch_shards(args, shard_paths, work / "preds")
    by_key = configs_from_preds(pred_paths, records)

    missing = []
    short = []
    payload = {}
    expected_keys = [f"{r['metadata']['dataset']}:{r['metadata']['algorithm']}" for r in records]
    for key in expected_keys:
        cfgs = by_key.get(key) or []
        if not cfgs:
            missing.append(key)
            continue
        if len(cfgs) < args.num_samples:
            short.append((key, len(cfgs)))
        payload[key] = cfgs[: args.num_samples]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "n_records": len(records),
        "n_keys": len(payload),
        "n_missing": len(missing),
        "n_short": len(short),
        "missing": missing,
        "short": short,
        "out_json": str(args.out_json),
    }
    (args.out_json.parent / "rec4_ws3_presets_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if missing:
        raise SystemExit(f"missing presets for {len(missing)} slots")


if __name__ == "__main__":
    main()
