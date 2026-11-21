"""
run_batch_ml.py
---------------
Batch execution for ML benchmarks across multiple LLM backends.
"""



import csv
import json
import os
import argparse
import glob
import statistics
import time
import traceback
from datetime import datetime

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"


from src.benchmarks.benchmark_runner import run_benchmark
from src.config.model_registry import MODEL_CONFIGS, require_ollama_server


API_KEY = os.getenv("OPENAI_HK_API_KEY")
TASK_IDS = [359955, 190411, 359977, 359956, 359970]
TASKS = ["lr", "rf", "xgboost", "nn"]
RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results_2")
LEGACY_RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results")

# Build TARGET_MODELS dynamically from MODEL_CONFIGS
API_MODELS = [
    # OpenAI
    "gpt-5.4",
    "gpt-5-mini",
    "gpt-3.5",
    # Google
    "gemini-3.1-pro",
    "gemini-3-flash",
    # Anthropic (PKU lab server)
    "claude-opus-4-6",
    "claude-opus-4-6-thinking",
    "claude-sonnet-4-6",
    "claude-sonnet-4-6-thinking",
    # DeepSeek
    "deepseek-v3.2",
    "deepseek-r1",
    # Zhipu AI (PKU lab server)
    "glm-5.1",
]

# Only include models that exist in MODEL_CONFIGS
TARGET_MODELS = [m for m in API_MODELS if m in MODEL_CONFIGS]

# Add Ollama models (now the only local model provider)
OLLAMA_MODEL_PRIORITY = ["llama-3.2-3b", "llama-3.2-vision", "qwen-3.5-4b", "qwen-3.5-9b"]
for ollama_model in OLLAMA_MODEL_PRIORITY:
    if ollama_model in MODEL_CONFIGS:
        TARGET_MODELS.append(ollama_model)
        print(f"[INFO] Ollama model available: {ollama_model}")


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

def result_exists(task_name: str, task_id: int, llm_model: str) -> bool:
    """Return True if this benchmark/model/task result artifact already exists."""
    safe_task = str(task_name).upper().replace("/", "_").replace(":", "_")
    safe_llm_model = str(llm_model).replace("/", "_").replace(":", "_")
    for root in (RESULTS_DIR, LEGACY_RESULTS_DIR):
        out_dir = os.path.join(root, safe_task, str(task_id), safe_llm_model)
        if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "hpo_*.json")):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML benchmark batch across multiple LLM backends.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip run combinations that already have saved result artifacts.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run (e.g., 'llama-3.2-vision-8k'). If not set, runs all available models.",
    )
    return parser.parse_args()



def result_exists(task_name: str, task_id: int, llm_model: str) -> bool:
    """Return True if this benchmark/model/task result artifact already exists."""
    safe_task = str(task_name).upper().replace("/", "_").replace(":", "_")
    safe_llm_model = str(llm_model).replace("/", "_").replace(":", "_")
    out_dir = os.path.join(RESULTS_DIR, safe_task, str(task_id), safe_llm_model)
    if not os.path.isdir(out_dir):
        return False
    return bool(glob.glob(os.path.join(out_dir, "hpo_*.json")))


def _pick_best_metrics(artifact: dict):
    best_name = artifact.get("best_performer")
    default = artifact.get("default_baseline", {}) or {}
    recs = artifact.get("recommendations", []) or []

    if best_name == "Default":
        return {
            "best_label": "Default",
            "best_loss": _safe_float(default.get("loss")),
            "best_accuracy": _safe_float(default.get("accuracy")),
            "best_cost_sec": _safe_float(default.get("cost")),
        }

    if isinstance(best_name, str) and best_name.startswith("Config"):
        try:
            idx = int(best_name.split()[-1])
            match = next((r for r in recs if r.get("config_num") == idx), None)
            if match is not None:
                return {
                    "best_label": best_name,
                    "best_loss": _safe_float(match.get("loss")),
                    "best_accuracy": _safe_float(match.get("accuracy")),
                    "best_cost_sec": _safe_float(match.get("cost")),
                }
        except Exception:
            pass

    # Fallback: compute best available among default + recommendations.
    candidates = []
    d_loss = _safe_float(default.get("loss"))
    if d_loss is not None:
        candidates.append(
            ("Default", d_loss, _safe_float(default.get("accuracy")), _safe_float(default.get("cost")))
        )
    for r in recs:
        r_loss = _safe_float(r.get("loss"))
        if r_loss is not None:
            candidates.append(
                (
                    f"Config {r.get('config_num')}",
                    r_loss,
                    _safe_float(r.get("accuracy")),
                    _safe_float(r.get("cost")),
                )
            )
    if not candidates:
        return {"best_label": None, "best_loss": None, "best_accuracy": None, "best_cost_sec": None}
    best = min(candidates, key=lambda x: x[1])
    return {
        "best_label": best[0],
        "best_loss": best[1],
        "best_accuracy": best[2],
        "best_cost_sec": best[3],
    }


def _tie_break_key(row):
    # Primary: lower loss, secondary lower cost_usd, tertiary lower total_tokens
    return (
        row.get("best_loss", float("inf")),
        row.get("cost_usd") if row.get("cost_usd") is not None else float("inf"),
        row.get("total_tokens") if row.get("total_tokens") is not None else float("inf"),
    )


def build_best_by_benchmark(rows):
    out = []
    for benchmark in TASKS:
        success = [r for r in rows if r.get("status") == "SUCCESS" and r.get("task") == benchmark and r.get("best_loss") is not None]
        if not success:
            out.append(
                {
                    "benchmark": benchmark,
                    "winner_llm": None,
                    "best_error": None,
                    "task_id_of_best": None,
                    "avg_error_all_models": None,
                    "runs_count": 0,
                }
            )
            continue
        winner = min(success, key=_tie_break_key)
        losses = [r["best_loss"] for r in success if r.get("best_loss") is not None]
        out.append(
            {
                "benchmark": benchmark,
                "winner_llm": winner["model_key"],
                "best_error": winner["best_loss"],
                "task_id_of_best": winner["task_id"],
                "avg_error_all_models": statistics.mean(losses) if losses else None,
                "runs_count": len(success),
            }
        )
    return out


def build_best_by_task_benchmark(rows):
    out = []
    for task_id in TASK_IDS:
        for benchmark in TASKS:
            success = [
                r
                for r in rows
                if r.get("status") == "SUCCESS"
                and r.get("task_id") == task_id
                and r.get("task") == benchmark
                and r.get("best_loss") is not None
            ]
            if not success:
                out.append(
                    {
                        "task_id": task_id,
                        "benchmark": benchmark,
                        "winner_llm": None,
                        "best_error": None,
                        "best_accuracy": None,
                        "winner_cost_usd": None,
                    }
                )
                continue
            winner = min(success, key=_tie_break_key)
            out.append(
                {
                    "task_id": task_id,
                    "benchmark": benchmark,
                    "winner_llm": winner["model_key"],
                    "best_error": winner["best_loss"],
                    "best_accuracy": winner.get("best_accuracy"),
                    "winner_cost_usd": winner.get("cost_usd"),
                }
            )
    return out


def build_llm_scorecard(rows):
    out = []
    for model_key in TARGET_MODELS:
        model_rows = [r for r in rows if r.get("model_key") == model_key]
        success = [r for r in model_rows if r.get("status") == "SUCCESS" and r.get("best_loss") is not None]
        losses = [r["best_loss"] for r in success]
        total_tokens = sum((r.get("total_tokens") or 0) for r in model_rows if r.get("status") == "SUCCESS")
        total_cost = sum((r.get("cost_usd") or 0.0) for r in model_rows if r.get("status") == "SUCCESS")
        out.append(
            {
                "llm": model_key,
                "success_runs": len(success),
                "failed_runs": len(model_rows) - len(success),
                "avg_error": statistics.mean(losses) if losses else None,
                "median_error": statistics.median(losses) if losses else None,
                "best_error": min(losses) if losses else None,
                "total_tokens": int(total_tokens),
                "total_cost_usd": round(total_cost, 6),
            }
        )
    return out


def _fmt_float(value, digits=5):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def _print_table(title, headers, rows):
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    print(" | ".join(h.ljust(20) for h in headers))
    print("-" * 100)
    for row in rows:
        print(" | ".join(str(row.get(h, "N/A")).ljust(20) for h in headers))


def save_summary_artifacts(summary_rows, best_by_benchmark, best_by_task_benchmark, llm_scorecard):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "metadata": {
            "timestamp": ts,
            "task_ids": TASK_IDS,
            "benchmarks": TASKS,
            "target_models": TARGET_MODELS,
        },
        "raw_rows": summary_rows,
        "tables": {
            "best_by_benchmark": best_by_benchmark,
            "best_by_task_and_benchmark": best_by_task_benchmark,
            "llm_scorecard": llm_scorecard,
        },
        "totals": {
            "run_count": len(summary_rows),
            "success_count": sum(1 for r in summary_rows if r.get("status") == "SUCCESS"),
            "failure_count": sum(1 for r in summary_rows if r.get("status") != "SUCCESS"),
            "total_cost_usd": round(sum((r.get("cost_usd") or 0.0) for r in summary_rows), 6),
        },
    }

    json_path = os.path.join(RESULTS_DIR, f"ml_batch_summary_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            if not rows:
                f.write("")
                return
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    csv_benchmark = os.path.join(RESULTS_DIR, f"ml_best_by_benchmark_{ts}.csv")
    csv_task_benchmark = os.path.join(RESULTS_DIR, f"ml_best_by_task_benchmark_{ts}.csv")
    csv_scorecard = os.path.join(RESULTS_DIR, f"ml_llm_scorecard_{ts}.csv")
    write_csv(csv_benchmark, best_by_benchmark)
    write_csv(csv_task_benchmark, best_by_task_benchmark)
    write_csv(csv_scorecard, llm_scorecard)

    md_path = os.path.join(RESULTS_DIR, f"ml_batch_summary_{ts}.md")
    with open(md_path, "w") as f:
        f.write("# ML Batch Summary\n\n")
        f.write(f"- Timestamp: {ts}\n")
        f.write(f"- Runs: {payload['totals']['run_count']}\n")
        f.write(f"- Success: {payload['totals']['success_count']}\n")
        f.write(f"- Failed: {payload['totals']['failure_count']}\n")
        f.write(f"- Total Cost USD: {payload['totals']['total_cost_usd']}\n\n")
        f.write("## Best by Benchmark\n\n")
        f.write(json.dumps(best_by_benchmark, indent=2))
        f.write("\n\n## Best by Task and Benchmark\n\n")
        f.write(json.dumps(best_by_task_benchmark, indent=2))
        f.write("\n\n## LLM Scorecard\n\n")
        f.write(json.dumps(llm_scorecard, indent=2))

    return {
        "json": json_path,
        "csv_best_by_benchmark": csv_benchmark,
        "csv_best_by_task_benchmark": csv_task_benchmark,
        "csv_llm_scorecard": csv_scorecard,
        "md": md_path,
    }


if __name__ == "__main__":
    args = parse_args()
    should_resume = args.resume
    
    # Filter models if --models specified
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",")]
        TARGET_MODELS = [m for m in TARGET_MODELS if m in selected_models]
        print(f"Filtered to {len(TARGET_MODELS)} model(s): {TARGET_MODELS}")

    require_ollama_server(TARGET_MODELS)
    
    if not API_KEY:
        raise EnvironmentError("OPENAI_HK_API_KEY is not set. Export it before running batch scripts.")

    print(f"Starting batch ML run for {len(TARGET_MODELS)} LLM models...")
    print(f"Resume mode: {'ON' if should_resume else 'OFF'}")
    summary = []
    for model_key in TARGET_MODELS:
        cfg = MODEL_CONFIGS[model_key]
        for task_id in TASK_IDS:
            for task in TASKS:
                if should_resume and result_exists(task_name=task, task_id=task_id, llm_model=cfg["id"]):
                    print(f"\n{'#' * 70}")
                    print(
                        f"SKIP (resume): LLM={model_key} | Benchmark={task.upper()} | TaskID={task_id} "
                        f"(existing result found)"
                    )
                    print(f"{'#' * 70}")
                    continue
                print(f"\n{'#' * 70}")
                print(
                    f"LLM={model_key} | Benchmark={task.upper()} | TaskID={task_id}"
                )
                print(f"{'#' * 70}")
                try:
                    artifact = run_benchmark(
                        model_name=task,
                        task_id=task_id,
                        api_key=cfg.get("api_key_override", API_KEY),
                        llm_model=cfg["id"],
                        base_url=cfg["base_url"],
                        model_key=model_key,
                    )
                    best_metrics = _pick_best_metrics(artifact)
                    token_usage = artifact.get("token_usage", {}) or {}
                    summary.append(
                        {
                            "model_key": model_key,
                            "task": task,
                            "task_id": task_id,
                            "status": "SUCCESS",
                            "best": artifact.get("best_performer"),
                            "best_label": best_metrics.get("best_label"),
                            "best_loss": best_metrics.get("best_loss"),
                            "best_accuracy": best_metrics.get("best_accuracy"),
                            "best_cost_sec": best_metrics.get("best_cost_sec"),
                            "prompt_tokens": token_usage.get("prompt_tokens"),
                            "completion_tokens": token_usage.get("completion_tokens"),
                            "total_tokens": token_usage.get("total_tokens"),
                            "cost_usd": artifact.get("total_cost_usd"),
                        }
                    )
                except Exception as e:
                    traceback.print_exc()
                    summary.append(
                        {
                            "model_key": model_key,
                            "task": task,
                            "task_id": task_id,
                            "status": f"FAILED: {e}",
                            "best": None,
                            "best_label": None,
                            "best_loss": None,
                            "best_accuracy": None,
                            "best_cost_sec": None,
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                            "cost_usd": None,
                        }
                    )
                print("Sleeping for 5 seconds before the next benchmark...")
                time.sleep(5)

    print(f"\n{'=' * 100}")
    print("BATCH ML SUMMARY")
    print(f"{'=' * 100}")
    for row in summary:
        print(
            f"{row['model_key']:<20} | {row['task']:<8} | {row['task_id']:<8} | "
            f"{row['status']:<30} | best={row['best']} | "
            f"err={_fmt_float(row.get('best_loss'))} | cost={row['cost_usd']}"
        )
    print(f"{'=' * 100}\n")

    best_by_benchmark = build_best_by_benchmark(summary)
    best_by_task_benchmark = build_best_by_task_benchmark(summary)
    llm_scorecard = build_llm_scorecard(summary)

    _print_table(
        "BEST LLM BY BENCHMARK",
        ["benchmark", "winner_llm", "best_error", "task_id_of_best", "avg_error_all_models", "runs_count"],
        best_by_benchmark,
    )
    _print_table(
        "BEST LLM BY TASK + BENCHMARK",
        ["task_id", "benchmark", "winner_llm", "best_error", "best_accuracy", "winner_cost_usd"],
        best_by_task_benchmark,
    )
    _print_table(
        "LLM SCORECARD",
        ["llm", "success_runs", "failed_runs", "avg_error", "median_error", "best_error", "total_tokens", "total_cost_usd"],
        llm_scorecard,
    )

    paths = save_summary_artifacts(summary, best_by_benchmark, best_by_task_benchmark, llm_scorecard)
    print(f"\nSaved ML batch summary artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
