#!/usr/bin/env python3
"""Parallel evaluation maximizing GPU utilization.

Splits test data into chunks, runs multiple workers per model across all GPUs.
Each GPU can hold 4x 9B model copies (75GB used / 96GB available).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_CLI = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/scripts/eval/evaluate_config_cli.py")
TEST_DATA = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data/ml_test_20.jsonl")
EVAL_DIR = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs")
OUTPUT_BASE = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/outputs")
BASE_MODEL = Path("/jizhicfs/pkuhetu/bcxu/models/Qwen3.5-9B-Base")
FULL_SFT = OUTPUT_BASE / "llamafactory_qwen35_9b_base_full_sft_4/checkpoint-45"

NUM_GPUS = 8
WORKERS_PER_GPU = 1  # safe default: one model copy per GPU to reduce OOM risk
TOTAL_WORKERS = NUM_GPUS * WORKERS_PER_GPU  # 8 workers
NUM_RUNS = 3
HARD_CODED_NUM_SAMPLES = 1
HARD_CODED_TEMPERATURE = 0.3
HARD_CODED_TOP_P = 0.9

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OPENAI_HK_API_KEY"] = "hk-tur55o100006045155343feb9008ac12174f4f260efc7ae4"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def count_jsonl_lines(path):
    if not path.exists():
        return 0
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())


def split_chunks(data, n_chunks):
    chunk_size = len(data) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else len(data)
        chunk = []
        for j, record in enumerate(data[start:end]):
            record = dict(record)  # shallow copy
            record["_global_index"] = start + j
            chunk.append(record)
        chunks.append(chunk)
    return chunks


def is_chunk_complete(out_jsonl, expected_lines):
    if not out_jsonl.exists():
        return False
    return count_jsonl_lines(out_jsonl) == expected_lines


def launch_worker(gpu_id, model_name, base_model, adapter_dir, chunk_idx, chunk_data, out_jsonl, num_samples, temperature, top_p, top_k):
    """Launch a single generation worker on a specific GPU."""
    chunk_file = EVAL_DIR / f"chunk_{model_name}_{chunk_idx}.jsonl"
    write_jsonl(chunk_file, chunk_data)

    cmd = [
        sys.executable, str(EVAL_CLI),
        "--generate", "qwen",
        "--dataset-jsonl", str(chunk_file),
        "--predictions-out", str(out_jsonl),
        "--base-model", str(base_model),
        "--model-label", model_name,
        "--max-new-tokens", "512",
        "--num-samples", str(num_samples),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
    ]
    if adapter_dir is not None:
        cmd += ["--adapter-dir", str(adapter_dir)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_file = EVAL_DIR / f"gen_{model_name}_gpu{gpu_id}_chunk{chunk_idx}.log"
    with open(log_file, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    return proc


def merge_predictions(model_name, out_jsonl, num_chunks, run_id: int | None = None):
    """Merge chunk predictions into final file."""
    all_preds = []
    run_suffix = f"_run{run_id}" if run_id is not None else ""
    for i in range(num_chunks):
        chunk_pred = EVAL_DIR / f"pred_{model_name}{run_suffix}_chunk{i}.jsonl"
        if chunk_pred.exists():
            all_preds.extend(load_jsonl(chunk_pred))
    write_jsonl(out_jsonl, all_preds)
    print(f"  {model_name}{run_suffix}: merged {len(all_preds)} predictions")


def merge_jsonl_files(output_path, input_paths):
    """Merge multiple JSONL files into one output file."""
    all_rows = []
    for path in input_paths:
        if path.exists():
            all_rows.extend(load_jsonl(path))
    write_jsonl(output_path, all_rows)
    print(f"  merged {len(all_rows)} rows into {output_path}")


def run_model_generation(model_name, base_model, adapter_dir, num_workers, num_gpus, num_samples, temperature, top_p, top_k, resume=False):
    """Generate predictions for a model using multiple workers and multiple n=1 runs."""
    print(f"\n=== Generating: {model_name} ({num_workers} workers, {num_samples} samples each, {NUM_RUNS} runs){' [resume]' if resume else ''} ===")

    # Load and split test data
    test_data = load_jsonl(TEST_DATA)
    chunks = split_chunks(test_data, num_workers)
    expected_lines = [len(chunk) for chunk in chunks]

    final_out = EVAL_DIR / f"{model_name}_test_predictions.jsonl"
    if resume and final_out.exists():
        print(f"  {model_name}: existing merged predictions found, skipping generation.")
        return final_out

    run_files = []
    for run_id in range(1, NUM_RUNS + 1):
        run_out = EVAL_DIR / f"{model_name}_test_predictions_run{run_id}.jsonl"
        if resume and run_out.exists() and count_jsonl_lines(run_out) == len(test_data):
            print(f"  {model_name} run {run_id}: existing run predictions found, skipping.")
            run_files.append(run_out)
            continue

        print(f"  {model_name} run {run_id}: starting")
        procs = []
        for i, chunk in enumerate(chunks):
            out_jsonl = EVAL_DIR / f"pred_{model_name}_run{run_id}_chunk{i}.jsonl"
            if resume and is_chunk_complete(out_jsonl, expected_lines[i]):
                print(f"    Chunk {i}: already complete ({expected_lines[i]} predictions). Skipping.")
                continue
            gpu_id = i % num_gpus
            proc = launch_worker(gpu_id, model_name, base_model, adapter_dir, i, chunk, out_jsonl, num_samples, temperature, top_p, top_k)
            procs.append((i, proc, out_jsonl, expected_lines[i]))
            print(f"    Chunk {i} -> GPU {gpu_id} ({len(chunk)} tasks, {num_samples} samples/task)")

        failed_chunks = []
        for i, proc, out_jsonl, expected in procs:
            ret = proc.wait()
            lines = count_jsonl_lines(out_jsonl)
            status = "ok" if ret == 0 and lines == expected else "FAILED"
            if ret != 0 or lines != expected:
                failed_chunks.append((i, ret, lines, expected, str(out_jsonl)))
            print(f"    Chunk {i}: {status} ({lines}/{expected} predictions, exit code {ret})")

        if failed_chunks:
            print(f"  WARNING: {len(failed_chunks)} chunks failed for {model_name} run {run_id}.")
            for idx, ret, lines, expected, path in failed_chunks:
                print(f"      chunk {idx}: exit {ret}, predictions {lines}/{expected}, file {path}")

        merge_predictions(model_name, run_out, num_workers, run_id=run_id)
        run_files.append(run_out)

    merge_jsonl_files(final_out, run_files)
    return final_out


def run_openbox_eval(model_name, pred_jsonl, out_json, resume=False, force_openbox=False):
    """Run OpenBox evaluation."""
    if not force_openbox and resume and out_json.exists():
        print(f"=== OpenBox eval: {model_name} [resume] ===")
        print(f"  {model_name}: existing eval {out_json} found, skipping.")
        return True

    if force_openbox and out_json.exists():
        print(f"=== OpenBox eval: {model_name} [force] ===")
        out_json.unlink()

    print(f"\n=== OpenBox eval: {model_name} ===")
    cmd = [
        sys.executable, str(EVAL_CLI),
        "--dataset-jsonl", str(TEST_DATA),
        "--predictions-jsonl", str(pred_jsonl),
        "--output-json", str(out_json),
    ]
    log_file = EVAL_DIR / f"eval_{model_name}.log"
    with open(log_file, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    print(f"  {model_name}: exit code {result.returncode}")
    return result.returncode == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run parallel SFT model evaluations.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of responses to generate per task. Unique configs are evaluated and the best is kept.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling.",
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
        help="Number of workers to launch per GPU. Use 1 for the safest execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous outputs by skipping already completed chunks and evaluations.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun all prediction outputs and ignore any existing model prediction files.",
    )
    parser.add_argument(
        "--force-openbox",
        action="store_true",
        help="Force rerun all OpenBox evals even if eval output files already exist.",
    )
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.force:
        print("=== Force mode: removing existing predictions and eval outputs ===")
        for f in EVAL_DIR.glob("*_test_predictions.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("*_test_predictions_run*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("*_test_openbox_eval.json"):
            f.unlink()
        for f in EVAL_DIR.glob("chunk_*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("pred_*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("gen_*.log"):
            f.unlink()
        for f in EVAL_DIR.glob("eval_*.log"):
            f.unlink()
    elif args.resume:
        print("=== Resume mode: preserving existing eval outputs and chunk files ===")
    else:
        # Clean old files
        print("=== Cleaning old eval files ===")
        for f in EVAL_DIR.glob("qwen_*_test_*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("qwen_*_test_*.json"):
            f.unlink()
        for f in EVAL_DIR.glob("lora_*_test_*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("lora_*_test_*.json"):
            f.unlink()
        for f in EVAL_DIR.glob("full_*_test_*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("full_*_test_*.json"):
            f.unlink()
        for f in EVAL_DIR.glob("chunk_*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("pred_*.jsonl"):
            f.unlink()
        for f in EVAL_DIR.glob("gen_*.log"):
            f.unlink()
        for f in EVAL_DIR.glob("eval_*.log"):
            f.unlink()

    global NUM_GPUS
    NUM_GPUS = args.num_gpus
    num_workers = args.num_gpus * args.workers_per_gpu

    # Models to evaluate (all from model_comparison_final.csv)
    # Format: (model_name, base_model_path, adapter_dir_or_none)
    # For full SFT models, base_model_path is the merged model dir, adapter_dir=None
    # For LoRA models, base_model_path=BASE_MODEL, adapter_dir=adapter_path
    models = [
        ("qwen_base", BASE_MODEL, None),
        ("qwen_base_lora_sft", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft"),
        ("qwen_base_lora_sft_3", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_3"),
        ("qwen_base_lora_sft_recommended", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_base_lora_sft_recommended"),
        ("qwen_lora_recommended5", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_lora_recommended5"),
        ("lora_recommended2", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_lora_recommended2"),
        ("lora_recommended3", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_lora_recommended3"),
        ("lora_recommended4", BASE_MODEL, OUTPUT_BASE / "llamafactory_qwen35_9b_lora_recommended4"),
        ("qwen_base_full_sft_4", OUTPUT_BASE / "llamafactory_qwen35_9b_base_full_sft_4", None),
        ("qwen_full_sft_recommended", OUTPUT_BASE / "llamafactory_qwen35_9b_full_sft_recommended", None),
        ("full_sft_5", OUTPUT_BASE / "llamafactory_qwen35_9b_full_sft_5", None),
        ("full_sft_6", OUTPUT_BASE / "llamafactory_qwen35_9b_full_sft_6", None),
    ]

    # Generate predictions for each model using the configured worker count
    print(f"=== Running with {args.num_gpus} GPUs x {args.workers_per_gpu} workers per GPU = {num_workers} total workers ===")
    pred_files = {}
    for model_name, base_model, adapter_dir in models:
        pred_file = run_model_generation(
            model_name,
            base_model,
            adapter_dir,
            num_workers,
            args.num_gpus,
            HARD_CODED_NUM_SAMPLES,
            HARD_CODED_TEMPERATURE,
            HARD_CODED_TOP_P,
            args.top_k,
            resume=args.resume,
        )
        pred_files[model_name] = pred_file

    # API models evaluation (CPU-based, sequential)
    api_models = [
        ("gemini_31_flash", "gemini-3.1-flash"),
        ("gemini_3.1_pro_preview", "gemini-3.1-pro-preview"),
        ("gpt_5.5", "gpt-5.5"),
        ("claude_opus_4_8", "claude-opus-4-8"),
    ]
    
    for model_label, api_model_name in api_models:
        print(f"\n=== Generating: {model_label} (API) ===")
        api_pred = EVAL_DIR / f"{model_label}_test_predictions.jsonl"
        run_files = []
        for run_id in range(1, 4):
            run_pred = EVAL_DIR / f"{model_label}_test_predictions_run{run_id}.jsonl"
            run_log = EVAL_DIR / f"gen_{model_label}_run{run_id}.log"
            api_cmd = [
                sys.executable, str(EVAL_CLI),
                "--generate", "openai",
                "--dataset-jsonl", str(TEST_DATA),
                "--predictions-out", str(run_pred),
                "--api-model", api_model_name,
                "--api-base-url", "https://api.openai-hk.com/v1",
                "--api-key", __import__("os").environ["OPENAI_HK_API_KEY"],
                "--model-label", model_label,
                "--max-new-tokens", "512",
                "--num-samples", "1",
                "--temperature", "0.3",
                "--top-p", "0.9",
            ]
            with open(run_log, "w") as log:
                api_proc = subprocess.Popen(api_cmd, stdout=log, stderr=subprocess.STDOUT)
            api_proc.wait()
            lines = len(load_jsonl(run_pred)) if run_pred.exists() else 0
            print(f"  {model_label} run {run_id}: done ({lines} predictions)")
            run_files.append(run_pred)
        merge_jsonl_files(api_pred, run_files)
        api_lines = len(load_jsonl(api_pred)) if api_pred.exists() else 0
        print(f"  {model_label}: merged {api_lines} predictions from 3 runs")
        pred_files[model_label] = api_pred

    # Run OpenBox evaluations
    eval_results = {}
    for model_name in list(pred_files.keys()):
        pred_jsonl = pred_files[model_name]
        out_json = EVAL_DIR / f"{model_name}_test_openbox_eval.json"
        ok = run_openbox_eval(model_name, pred_jsonl, out_json, resume=args.resume, force_openbox=args.force_openbox)
        eval_results[model_name] = ok

    # Combine all
    print("\n=== Combining all evaluations ===")
    combined = {"summary": {}, "results": []}
    for model_name in eval_results:
        out_json = EVAL_DIR / f"{model_name}_test_openbox_eval.json"
        if out_json.exists():
            data = json.loads(out_json.read_text())
            combined["summary"].update(data.get("summary", {}))
            combined["results"].extend(data.get("results", []))

    final_out = EVAL_DIR / "qwen_all_test_openbox_eval.json"
    final_out.write_text(json.dumps(combined, indent=2, sort_keys=True))
    print(f"\nCombined: {final_out}")
    print(json.dumps(combined["summary"], indent=2))

    print("\n========================================")
    print("ALL EVALUATIONS COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
