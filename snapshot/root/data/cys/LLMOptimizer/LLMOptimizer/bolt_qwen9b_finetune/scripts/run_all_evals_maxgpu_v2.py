#!/usr/bin/env python3
"""Parallel DL taskset evaluation for trained Qwen3.5-9B models.

1. Generate predictions on dl_test_15_modified.jsonl (3 runs x n=1, temp=0.3, top_p=0.9).
2. Evaluate with task_set train_inner using paper §3.3 scoring:
   mean normalized VALID_OUTER loss, 5 replicas, 10k steps.
3. Also run API models: gpt_5.5, claude_opus_4_8, gemini_3.1_pro_preview, gemini_31_flash.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "scripts" / "eval"))
from taskset_paper_scoring import (  # noqa: E402
    PAPER_BASELINE_TASK_WORKERS,
    PAPER_EVAL_EVERY_N,
    PAPER_NUM_REPLICAS,
    PAPER_REPLICA_AGGREGATION,
    PAPER_TASK_WORKERS,
    PAPER_TRAINING_STEPS,
    get_task_best_val,
    load_baseline_cache,
)
EVAL_CLI = BASE_DIR / "scripts/eval/evaluate_config_cli.py"
TASKSET_EVAL = BASE_DIR / "scripts/eval/run_taskset_dl_eval_optimized.py"
TEST_DATA = BASE_DIR / "data/dl_test_15_modified.jsonl"
EVAL_DIR = BASE_DIR / "eval_runs/dl_test_15"
OUTPUT_BASE = Path("/root/data/cys/blockdata_archive/model-weights")
BASE_MODEL = Path("/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B")

NUM_GPUS = 4
WORKERS_PER_GPU = 1  # safe default: one model copy per GPU (same as non-v2)
NUM_RUNS = 3
MAX_NEW_TOKENS = 512
# Reasoning models (e.g. gemini-3.1-pro-preview) spend output tokens on internal
# thinking before the visible JSON answer; 512 truncates or yields empty responses.
API_MODEL_MAX_NEW_TOKENS: dict[str, int] = {
    "gemini_3.1_pro_preview": 2048,
}
HARD_CODED_NUM_SAMPLES = 1
HARD_CODED_TEMPERATURE = 0.3
HARD_CODED_TOP_P = 0.9
HARD_CODED_TOP_K = 50
BASELINE_CACHE = EVAL_DIR / "task_baselines.json"

GENERATION_PYTHON = Path("/root/miniconda3/envs/llamafactory_qwen35/bin/python")
TASKSET_PYTHON = Path("/root/miniconda3/envs/taskset_eval/bin/python")

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault(
    "OPENAI_HK_API_KEY",
    "hk-tur55o100006045155343feb9008ac12174f4f260efc7ae4",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def split_chunks(data: list[dict], n_chunks: int) -> list[list[dict]]:
    if n_chunks <= 0:
        return [data]
    chunk_size = max(1, len(data) // n_chunks)
    chunks: list[list[dict]] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(data)
        if start >= len(data):
            break
        chunk = []
        for j, record in enumerate(data[start:end]):
            record = dict(record)
            record["_global_index"] = start + j
            chunk.append(record)
        chunks.append(chunk)
    return chunks


def merge_jsonl_files(output_path: Path, input_paths: list[Path]) -> None:
    rows: list[dict] = []
    for path in input_paths:
        if path.exists():
            rows.extend(load_jsonl(path))
    write_jsonl(output_path, rows)
    print(f"  merged {len(rows)} rows into {output_path}")


def launch_worker(
    gpu_id: int,
    model_name: str,
    base_model: Path,
    adapter_dir: Path | None,
    chunk_idx: int,
    chunk_data: list[dict],
    out_jsonl: Path,
    run_suffix: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> subprocess.Popen:
    chunk_file = EVAL_DIR / f"chunk_{model_name}{run_suffix}_{chunk_idx}.jsonl"
    write_jsonl(chunk_file, chunk_data)

    cmd = [
        str(GENERATION_PYTHON),
        str(EVAL_CLI),
        "--generate",
        "qwen",
        "--dataset-jsonl",
        str(chunk_file),
        "--predictions-out",
        str(out_jsonl),
        "--base-model",
        str(base_model),
        "--model-label",
        model_name,
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--num-samples",
        str(num_samples),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
    ]
    if adapter_dir is not None:
        cmd += ["--adapter-dir", str(adapter_dir)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = EVAL_DIR / f"gen_{model_name}{run_suffix}_gpu{gpu_id}_chunk{chunk_idx}.log"
    with log_file.open("w") as log:
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def merge_chunk_predictions(model_name: str, out_jsonl: Path, num_chunks: int, run_suffix: str = "") -> None:
    rows: list[dict] = []
    for i in range(num_chunks):
        chunk_pred = EVAL_DIR / f"pred_{model_name}{run_suffix}_chunk{i}.jsonl"
        if chunk_pred.exists():
            rows.extend(load_jsonl(chunk_pred))
    write_jsonl(out_jsonl, rows)
    print(f"  {model_name}{run_suffix}: merged {len(rows)} predictions")


def run_model_generation(
    model_name: str,
    base_model: Path,
    adapter_dir: Path | None,
    num_workers: int,
    num_gpus: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    resume: bool = False,
) -> Path:
    print(
        f"\n=== Generating: {model_name} "
        f"({num_workers} workers, {num_samples} samples each, {NUM_RUNS} runs)"
        f"{' [resume]' if resume else ''} ==="
    )

    test_data = load_jsonl(TEST_DATA)
    chunks = split_chunks(test_data, num_workers)
    expected_per_chunk = [len(chunk) for chunk in chunks]
    final_out = EVAL_DIR / f"{model_name}_test_predictions.jsonl"

    if resume and final_out.exists():
        print(f"  {model_name}: existing merged predictions found, skipping generation.")
        return final_out

    run_files: list[Path] = []
    for run_id in range(1, NUM_RUNS + 1):
        run_suffix = f"_run{run_id}"
        run_out = EVAL_DIR / f"{model_name}_test_predictions{run_suffix}.jsonl"
        if resume and run_out.exists() and count_jsonl_lines(run_out) == len(test_data):
            print(f"  {model_name} run {run_id}: already complete, skipping.")
            run_files.append(run_out)
            continue

        print(f"  {model_name} run {run_id}: starting")
        procs = []
        for i, chunk in enumerate(chunks):
            out_jsonl = EVAL_DIR / f"pred_{model_name}{run_suffix}_chunk{i}.jsonl"
            if resume and count_jsonl_lines(out_jsonl) == expected_per_chunk[i]:
                print(f"    Chunk {i}: already complete ({expected_per_chunk[i]} predictions)")
                continue
            gpu_id = i % num_gpus
            proc = launch_worker(
                gpu_id,
                model_name,
                base_model,
                adapter_dir,
                i,
                chunk,
                out_jsonl,
                run_suffix,
                num_samples,
                temperature,
                top_p,
                top_k,
            )
            procs.append((i, proc, out_jsonl, expected_per_chunk[i]))
            print(
                f"    Chunk {i} -> GPU {gpu_id} "
                f"({len(chunk)} tasks, {num_samples} samples/task)"
            )

        for i, proc, out_jsonl, expected in procs:
            ret = proc.wait()
            lines = count_jsonl_lines(out_jsonl)
            status = "ok" if ret == 0 and lines == expected else "FAILED"
            print(f"    Chunk {i}: {status} ({lines}/{expected}, exit {ret})")

        merge_chunk_predictions(model_name, run_out, len(chunks), run_suffix=run_suffix)
        run_files.append(run_out)

    merge_jsonl_files(final_out, run_files)
    return final_out


def run_api_generation(
    model_label: str,
    api_model_name: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    resume: bool = False,
) -> Path:
    print(f"\n=== Generating: {model_label} (API) ===")
    test_data = load_jsonl(TEST_DATA)
    final_out = EVAL_DIR / f"{model_label}_test_predictions.jsonl"
    if resume and final_out.exists():
        print(f"  {model_label}: existing API predictions found, skipping.")
        return final_out

    run_files: list[Path] = []
    for run_id in range(1, NUM_RUNS + 1):
        run_out = EVAL_DIR / f"{model_label}_test_predictions_run{run_id}.jsonl"
        if resume and run_out.exists() and count_jsonl_lines(run_out) == len(test_data):
            print(f"  {model_label} run {run_id}: already complete, skipping.")
            run_files.append(run_out)
            continue
        log_file = EVAL_DIR / f"gen_{model_label}_run{run_id}.log"
        cmd = [
            sys.executable,
            str(EVAL_CLI),
            "--generate",
            "openai",
            "--dataset-jsonl",
            str(TEST_DATA),
            "--predictions-out",
            str(run_out),
            "--api-model",
            api_model_name,
            "--api-base-url",
            "https://api.openai-hk.com/v1",
            "--model-label",
            model_label,
            "--max-new-tokens",
            str(API_MODEL_MAX_NEW_TOKENS.get(model_label, MAX_NEW_TOKENS)),
            "--num-samples",
            str(num_samples),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
        ]
        api_key = os.environ.get("OPENAI_HK_API_KEY")
        if api_key:
            cmd += ["--api-key", api_key]

        with log_file.open("w") as log:
            api_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        api_proc.wait()
        lines = count_jsonl_lines(run_out)
        print(f"  {model_label} run {run_id}: done ({lines} predictions)")
        run_files.append(run_out)

    merge_jsonl_files(final_out, run_files)
    return final_out


def count_missing_baselines() -> tuple[int, int]:
    """Return (missing, total_resolvable) baseline entries for dl_test_15."""
    sys.path.insert(0, str(BASE_DIR / "scripts" / "eval"))
    from run_taskset_dl_eval_optimized import find_task_seed, load_jsonl  # noqa: WPS433

    gold_records = load_jsonl(TEST_DATA)
    baseline_cache = load_baseline_cache(BASELINE_CACHE)
    total = 0
    missing = 0
    for gold_record in gold_records:
        user_msg = json.loads(gold_record["messages"][1]["content"])
        task_name = find_task_seed(user_msg["task_family"], user_msg["task"])
        if not task_name:
            continue
        total += 1
        if get_task_best_val(task_name, baseline_cache) is None:
            missing += 1
    return missing, total


def build_all_baselines(resume: bool = False, baseline_task_workers: int | None = None) -> bool:
    workers = baseline_task_workers if baseline_task_workers is not None else PAPER_BASELINE_TASK_WORKERS
    missing, total = count_missing_baselines()
    if missing == 0:
        print(f"Baseline cache complete ({total}/{total} tasks): {BASELINE_CACHE}")
        return True
    if resume and BASELINE_CACHE.exists() and missing < total:
        print(f"Baseline cache partial ({total - missing}/{total} cached); building {missing} remaining...")
    else:
        print(f"Building baselines for {missing}/{total} tasks -> {BASELINE_CACHE}")

    cmd = [
        str(TASKSET_PYTHON),
        str(TASKSET_EVAL),
        "--build-baselines-only",
        "--test-data",
        str(TEST_DATA),
        "--baseline-cache",
        str(BASELINE_CACHE),
        "--work-dir",
        str(EVAL_DIR / "baseline_build"),
    ]
    if baseline_task_workers is not None:
        cmd.extend(["--baseline-task-workers", str(workers)])
    log_file = EVAL_DIR / "build_baselines.log"
    print(
        f"  workers: {workers} parallel tasks | "
        f"progress: {EVAL_DIR / 'baseline_build' / 'progress.json'}"
    )
    with log_file.open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"  baseline build exit code {result.returncode} (log: {log_file})")
    return result.returncode == 0


def run_taskset_eval(model_name: str, pred_jsonl: Path, resume: bool = False, debug: bool = False) -> bool:
    out_json = EVAL_DIR / f"{model_name}_dl_taskset_eval.json"
    if resume and out_json.exists():
        print(f"  {model_name}: taskset eval output exists, skipping.")
        return True

    print(
        f"\n=== Taskset eval: {model_name} "
        f"(paper §3.3, {PAPER_NUM_REPLICAS} replicas, {PAPER_REPLICA_AGGREGATION} agg, "
        f"{PAPER_TRAINING_STEPS} steps, eval every {PAPER_EVAL_EVERY_N}, "
        f"{PAPER_TASK_WORKERS} task workers) ==="
    )
    cmd = [
        str(TASKSET_PYTHON),
        str(TASKSET_EVAL),
        "--predictions",
        str(pred_jsonl),
        "--test-data",
        str(TEST_DATA),
        "--model-label",
        model_name,
        "--baseline-cache",
        str(BASELINE_CACHE),
        "--skip-baseline-build",
        "--output-json",
        str(out_json),
        "--work-dir",
        str(EVAL_DIR / f"taskset_work_{model_name}"),
    ]
    if debug:
        cmd.append("--debug")
    log_file = EVAL_DIR / f"eval_{model_name}.log"
    with log_file.open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"  {model_name}: exit code {result.returncode}")
    return result.returncode == 0


def dl_models() -> list[tuple[str, Path, Path | None]]:
    return [
        ("qwen_base", BASE_MODEL, None),
        (
            "qwen_full_sft_4_dl",
            OUTPUT_BASE / "llamafactory_qwen35_9b_base_full_sft_4_dl",
            None,
        ),
        (
            "qwen_full_sft_4_dl_rec",
            OUTPUT_BASE / "llamafactory_qwen35_9b_base_full_sft_4_dl_rec",
            None,
        ),
        ("qwen_base_lora_sft_dl", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_dl"),
        ("qwen_base_lora_sft_2_dl", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_2_dl"),
        ("qwen_base_lora_sft_3_dl", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_3_dl"),
        ("qwen_base_lora_sft_4_dl", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_4_dl"),
        ("qwen_base_lora_sft_5_dl", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_5_dl"),
        ("qwen_base_lora_sft_6_dl", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_6_dl"),
        ("qwen_base_lora_sft_7_dl", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_7_dl"),
    ]


# (label, api_model_slug) — labels match prior ML eval naming for comparison CSVs.
API_MODELS = [
    ("gpt_5.5", "gpt-5.5"),
    ("claude_opus_4_8", "claude-opus-4-8"),
    ("gemini_3.1_pro_preview", "gemini-3.1-pro-preview"),
    ("gemini_31_flash", "gemini-3.1-flash"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DL taskset prediction + evaluation pipeline.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Ignored; pipeline uses HARD_CODED_NUM_SAMPLES.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=NUM_GPUS,
        help="Number of GPUs available for model generation.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=WORKERS_PER_GPU,
        help="Workers per GPU. Use 1 for safest execution (same as non-v2).",
    )
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--debug-eval",
        action="store_true",
        help="Pass --debug to taskset eval (per-replica train_inner logs under work-dir/debug/).",
    )
    parser.add_argument("--skip-api", action="store_true", help="Skip API model generation/eval.")
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Only run API models (skip local Qwen checkpoints).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping completed chunks and eval outputs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun: delete existing predictions and eval outputs.",
    )
    parser.add_argument(
        "--build-baselines-first",
        action="store_true",
        help="Build shared task_baselines.json for all test tasks before model eval.",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Only build task_baselines.json and exit (no generation or model eval).",
    )
    parser.add_argument(
        "--baseline-task-workers",
        type=int,
        default=None,
        help=f"Parallel CPU tasks for baseline grid build (default {PAPER_BASELINE_TASK_WORKERS}).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Optional subset of model labels to run (default: all local + API).",
    )
    return parser.parse_args()


def clean_eval_dir(force: bool = False, resume: bool = False) -> None:
    if force:
        print("=== Force mode: removing existing predictions and eval outputs ===")
        patterns = [
            "*_test_predictions.jsonl",
            "*_test_predictions_run*.jsonl",
            "*_dl_taskset_eval.json",
            "dl_test_15_all_taskset_eval.json",
            "chunk_*.jsonl",
            "pred_*.jsonl",
            "gen_*.log",
            "eval_*.log",
        ]
        for pattern in patterns:
            for path in EVAL_DIR.glob(pattern):
                path.unlink()
        return

    if resume:
        print("=== Resume mode: preserving existing eval outputs and chunk files ===")
        return

    print("=== Cleaning old eval files ===")
    for pattern in ("*_test_predictions*.jsonl", "*_dl_taskset_eval.json", "chunk_*.jsonl", "pred_*.jsonl"):
        for path in EVAL_DIR.glob(pattern):
            path.unlink()


def main() -> int:
    args = parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    clean_eval_dir(force=args.force, resume=args.resume)

    num_workers = max(1, args.num_gpus * args.workers_per_gpu)
    selected = set(args.models) if args.models else None

    print("=== DL taskset eval pipeline ===")
    print(
        f"=== Running with {args.num_gpus} GPUs x {args.workers_per_gpu} workers/GPU "
        f"= {num_workers} total workers ==="
    )
    print(f"Test data : {TEST_DATA}")
    print(f"Output dir: {EVAL_DIR}")
    print(
        f"Gen params: {NUM_RUNS} runs x n={HARD_CODED_NUM_SAMPLES}, "
        f"temp={HARD_CODED_TEMPERATURE}, top_p={HARD_CODED_TOP_P}, "
        f"top_k={HARD_CODED_TOP_K}, max_new_tokens={MAX_NEW_TOKENS}"
    )
    print(
        f"Eval params: paper §3.3 — {PAPER_NUM_REPLICAS} replicas, "
        f"{PAPER_REPLICA_AGGREGATION} replica agg, mean norm VALID_OUTER, "
        f"{PAPER_TRAINING_STEPS} steps, eval every {PAPER_EVAL_EVERY_N}, "
        f"{PAPER_TASK_WORKERS} eval task workers, baseline cache: {BASELINE_CACHE}"
    )
    if not args.skip_api and not args.api_only:
        print("API models:", ", ".join(label for label, _ in API_MODELS))
    elif args.api_only:
        print("Mode: API models only ->", ", ".join(label for label, _ in API_MODELS))
    print()

    if args.baselines_only or args.build_baselines_first:
        ok = build_all_baselines(resume=args.resume, baseline_task_workers=args.baseline_task_workers)
        if args.baselines_only:
            return 0 if ok else 1
        if not ok:
            print("WARNING: baseline build failed; model eval may fail for uncached tasks.")

    pred_files: dict[str, Path] = {}

    if not args.skip_gen:
        if not args.api_only:
            for model_name, base_model, adapter_dir in dl_models():
                if selected and model_name not in selected:
                    continue
                if model_name == "qwen_base" or (
                    base_model.exists() and (adapter_dir is None or adapter_dir.exists())
                ):
                    pred_files[model_name] = run_model_generation(
                        model_name,
                        base_model,
                        adapter_dir,
                        num_workers,
                        args.num_gpus,
                        HARD_CODED_NUM_SAMPLES,
                        HARD_CODED_TEMPERATURE,
                        HARD_CODED_TOP_P,
                        HARD_CODED_TOP_K,
                        resume=args.resume,
                    )
                else:
                    print(
                        f"  SKIP {model_name}: weights not found at "
                        f"{base_model if adapter_dir is None else adapter_dir}"
                    )

        if not args.skip_api:
            print("\n=== API models ===")
            for model_label, api_model_name in API_MODELS:
                if selected and model_label not in selected:
                    continue
                pred_files[model_label] = run_api_generation(
                    model_label,
                    api_model_name,
                    HARD_CODED_NUM_SAMPLES,
                    HARD_CODED_TEMPERATURE,
                    HARD_CODED_TOP_P,
                    resume=args.resume,
                )
    else:
        for path in EVAL_DIR.glob("*_test_predictions.jsonl"):
            pred_files[path.name.replace("_test_predictions.jsonl", "")] = path

    if not args.skip_eval:
        eval_results: dict[str, bool] = {}
        for model_name, pred_jsonl in pred_files.items():
            if pred_jsonl.exists():
                eval_results[model_name] = run_taskset_eval(
                    model_name, pred_jsonl, resume=args.resume, debug=args.debug_eval
                )
            else:
                eval_results[model_name] = False

        combined = {"summary": {}, "results": []}
        for model_name, ok in eval_results.items():
            out_json = EVAL_DIR / f"{model_name}_dl_taskset_eval.json"
            if out_json.exists():
                data = json.loads(out_json.read_text())
                combined["summary"][model_name] = data.get("summary", {})
                combined["results"].extend(data.get("results", []))
            combined["summary"].setdefault(model_name, {})["eval_ok"] = ok

        final_out = EVAL_DIR / "dl_test_15_all_taskset_eval.json"
        final_out.write_text(json.dumps(combined, indent=2, sort_keys=True))
        print(f"\nCombined results: {final_out}")
        print(json.dumps(combined["summary"], indent=2))

    print("\n========================================")
    print("DL TASKSET EVALUATION COMPLETE")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
