"""
run_batch_nas_201.py
--------------------
Batch execution script for NASBench-201 benchmarks (local, no containers).

These benchmarks work locally without containers since the data files
have been extracted to ~/.local/share/hpobench/nasbench_201/
"""

# Set OpenMP thread limits to prevent over-subscription
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"

import argparse
import csv
import glob
import json
import os
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


from src.config.model_registry import MODEL_CONFIGS, is_ollama_model, require_ollama_server
from src.benchmarks.nas_runner import run_nas_benchmark

API_KEY = os.getenv("OPENAI_HK_API_KEY")
RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results_2")

TARGET_MODELS = [
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
    # Ollama models (local)
    "llama-3.2-3b",
    "llama-3.2-vision",
    "qwen-3.5-4b",
    "qwen-3.5-9b",
]

# NASBench-201 benchmarks - work locally with data files
TARGET_BENCHMARKS = [
    "Cifar10ValidNasBench201",
    "Cifar100NasBench201",
    "ImageNetNasBench201",
]


def result_exists(benchmark_name: str, llm_model: str) -> bool:
    """Return True if this benchmark/model result artifact already exists."""
    safe_benchmark = str(benchmark_name).replace("/", "_").replace(":", "_")
    safe_llm_model = str(llm_model).replace("/", "_").replace(":", "_")
    out_dir = os.path.join(RESULTS_DIR, safe_benchmark, safe_llm_model)
    if not os.path.isdir(out_dir):
        return False
    return bool(glob.glob(os.path.join(out_dir, "nas_*.json")))


def parse_args():
    parser = argparse.ArgumentParser(description="Run NASBench-201 benchmark batch across multiple LLM backends.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip run combinations that already have saved result artifacts.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to run (e.g., 'llama-3.2-3b,gpt-5-mini'). If not set, runs all available models.",
    )
    return parser.parse_args()


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _pick_best_metrics(artifact: dict):
    best_name = artifact.get("best_performer")
    default = artifact.get("default_baseline", {}) or {}
    recs = artifact.get("recommendations", []) or []

    default_loss = _safe_float(default.get("loss"))
    default_acc = _safe_float(default.get("accuracy"))
    if default_acc is None and default_loss is not None:
        default_acc = (1 - default_loss) * 100

    if best_name == "Default":
        return {
            "best_label": "Default",
            "best_loss": default_loss,
            "best_accuracy": default_acc,
            "best_cost_sec": _safe_float(default.get("cost")),
        }

    if isinstance(best_name, str) and best_name.startswith("Config"):
        try:
            idx = int(best_name.split()[-1])
            match = next((r for r in recs if r.get("config_num") == idx), None)
            if match is not None:
                m_loss = _safe_float(match.get("loss"))
                m_acc = _safe_float(match.get("accuracy"))
                if m_acc is None and m_loss is not None:
                    m_acc = (1 - m_loss) * 100
                return {
                    "best_label": best_name,
                    "best_loss": m_loss,
                    "best_accuracy": m_acc,
                    "best_cost_sec": _safe_float(match.get("cost")),
                }
        except Exception:
            pass

    candidates = []
    if default_loss is not None:
        candidates.append(("Default", default_loss, default_acc, _safe_float(default.get("cost"))))
    for r in recs:
        r_loss = _safe_float(r.get("loss"))
        if r_loss is None:
            continue
        r_acc = _safe_float(r.get("accuracy"))
        if r_acc is None:
            r_acc = (1 - r_loss) * 100
        candidates.append((f"Config {r.get('config_num')}", r_loss, r_acc, _safe_float(r.get("cost"))))

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
    return (
        row.get("best_loss", float("inf")),
        row.get("cost_usd") if row.get("cost_usd") is not None else float("inf"),
        row.get("total_tokens") if row.get("total_tokens") is not None else float("inf"),
    )


def build_best_by_benchmark(rows):
    out = []
    for benchmark in TARGET_BENCHMARKS:
        success = [
            r for r in rows
            if r.get("status") == "SUCCESS"
            and r.get("benchmark") == benchmark
            and r.get("best_loss") is not None
        ]
        if not success:
            out.append(
                {
                    "benchmark": benchmark,
                    "winner_llm": None,
                    "best_error": None,
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
                "avg_error_all_models": statistics.mean(losses) if losses else None,
                "runs_count": len(success),
            }
        )
    return out


def build_llm_scorecard(rows, model_keys=None):
    out = []
    if model_keys is None:
        model_keys = TARGET_MODELS
    for model_key in model_keys:
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


def save_summary_artifacts(summary_rows, best_by_benchmark, llm_scorecard):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    payload = {
        "metadata": {
            "timestamp": ts,
            "benchmarks": TARGET_BENCHMARKS,
            "target_models": TARGET_MODELS,
        },
        "raw_rows": summary_rows,
        "tables": {
            "best_by_benchmark": best_by_benchmark,
            "llm_scorecard": llm_scorecard,
        },
        "totals": {
            "run_count": len(summary_rows),
            "success_count": sum(1 for r in summary_rows if r.get("status") == "SUCCESS"),
            "failure_count": sum(1 for r in summary_rows if r.get("status") != "SUCCESS"),
            "total_cost_usd": round(sum((r.get("cost_usd") or 0.0) for r in summary_rows), 6),
        },
    }

    json_path = os.path.join(RESULTS_DIR, f"nas201_batch_summary_{ts}.json")
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

    csv_benchmark = os.path.join(RESULTS_DIR, f"nas201_best_by_benchmark_{ts}.csv")
    csv_scorecard = os.path.join(RESULTS_DIR, f"nas201_llm_scorecard_{ts}.csv")
    write_csv(csv_benchmark, best_by_benchmark)
    write_csv(csv_scorecard, llm_scorecard)

    md_path = os.path.join(RESULTS_DIR, f"nas201_batch_summary_{ts}.md")
    with open(md_path, "w") as f:
        f.write("# NASBench-201 Batch Summary\n\n")
        f.write(f"- Timestamp: {ts}\n")
        f.write(f"- Runs: {payload['totals']['run_count']}\n")
        f.write(f"- Success: {payload['totals']['success_count']}\n")
        f.write(f"- Failed: {payload['totals']['failure_count']}\n")
        f.write(f"- Total Cost USD: {payload['totals']['total_cost_usd']}\n\n")
        f.write("## Best by Benchmark\n\n")
        f.write(json.dumps(best_by_benchmark, indent=2))
        f.write("\n\n## LLM Scorecard\n\n")
        f.write(json.dumps(llm_scorecard, indent=2))

    return {
        "json": json_path,
        "csv_best_by_benchmark": csv_benchmark,
        "csv_llm_scorecard": csv_scorecard,
        "md": md_path,
    }


if __name__ == "__main__":
    args = parse_args()
    should_resume = args.resume

    if not API_KEY:
        raise EnvironmentError("OPENAI_HK_API_KEY is not set. Export it before running batch scripts.")

    # Filter models if --models specified
    target_models = TARGET_MODELS
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",")]
        target_models = [m for m in target_models if m in selected_models]
        print(f"Filtered to {len(target_models)} model(s): {target_models}")

    ollama_models = [m for m in target_models if is_ollama_model(m)]
    if ollama_models:
        try:
            require_ollama_server(ollama_models)
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
            print("[WARN] Skipping local Ollama models and continuing with API-backed models.")
            target_models = [m for m in target_models if not is_ollama_model(m)]

    if not target_models:
        raise RuntimeError("No runnable models remain after filtering unavailable Ollama backends.")

    print(
        f"Starting NASBench-201 Batch Evaluation for {len(TARGET_BENCHMARKS)} benchmarks "
        f"across {len(target_models)} LLM models...\n"
    )
    print(f"Resume mode: {'ON' if should_resume else 'OFF'}")

    summary = []
    for model_key in target_models:
        cfg = MODEL_CONFIGS[model_key]
        for benchmark_name in TARGET_BENCHMARKS:
            cfg = MODEL_CONFIGS[model_key]
            if should_resume and result_exists(benchmark_name=benchmark_name, llm_model=cfg["id"]):
                print(f"\n{'#' * 70}")
                print(
                    f"SKIP (resume): LLM={model_key} | Benchmark={benchmark_name} "
                    f"(existing result found)"
                )
                print(f"{'#' * 70}")
                continue

            print(f"\n\n{'#' * 70}")
            print(f"INITIATING PIPELINE FOR LLM={model_key}, BENCHMARK={benchmark_name}")
            print(f"{'#' * 70}")

            try:
                result = run_nas_benchmark(
                    benchmark_name=benchmark_name,
                    api_key=cfg.get("api_key_override", API_KEY),
                    task_id=None,
                    llm_model=cfg["id"],
                    base_url=cfg["base_url"],
                    model_key=model_key,
                )
                token_usage = result.get("token_usage", {}) or {}
                best_metrics = _pick_best_metrics(result)
                summary.append(
                    {
                        "model_key": model_key,
                        "benchmark": benchmark_name,
                        "status": "SUCCESS",
                        "best": result.get("best_performer"),
                        "best_label": best_metrics.get("best_label"),
                        "best_loss": best_metrics.get("best_loss"),
                        "best_accuracy": best_metrics.get("best_accuracy"),
                        "best_cost_sec": best_metrics.get("best_cost_sec"),
                        "prompt_tokens": token_usage.get("prompt_tokens"),
                        "completion_tokens": token_usage.get("completion_tokens"),
                        "total_tokens": token_usage.get("total_tokens"),
                        "cost_usd": result.get("total_cost_usd"),
                    }
                )
            except Exception as e:
                print(f"\nCRITICAL FAILURE for {benchmark_name}:")
                traceback.print_exc()
                summary.append(
                    {
                        "model_key": model_key,
                        "benchmark": benchmark_name,
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
    print("BATCH NAS SUMMARY")
    print(f"{'=' * 100}")
    for row in summary:
        print(
            f"{row['model_key']:<20} | {row['benchmark']:<24} | {row['status']:<30} | "
            f"best={row['best']} | err={_fmt_float(row.get('best_loss'))} | "
            f"acc={_fmt_float(row.get('best_accuracy'), 2)} | cost={row['cost_usd']}"
        )
    print(f"{'=' * 100}\n")

    # Build target models list for scorecard (use filtered list if --models was specified)
    scorecard_models = target_models

    best_by_benchmark = build_best_by_benchmark(summary)
    llm_scorecard = build_llm_scorecard(summary, scorecard_models)

    _print_table(
        "BEST LLM BY BENCHMARK",
        ["benchmark", "winner_llm", "best_error", "avg_error_all_models", "runs_count"],
        best_by_benchmark,
    )
    _print_table(
        "LLM SCORECARD",
        ["llm", "success_runs", "failed_runs", "avg_error", "median_error", "best_error", "total_tokens", "total_cost_usd"],
        llm_scorecard,
    )

    paths = save_summary_artifacts(summary, best_by_benchmark, llm_scorecard)
    print("\nSaved NASBench-201 batch summary artifacts:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
