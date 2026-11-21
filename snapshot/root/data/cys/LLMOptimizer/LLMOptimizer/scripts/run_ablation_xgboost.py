"""
run_ablation_xgboost.py
-----------------------
Ablation study for XGBoost task 190411:
- baseline prompt
- prompt with "Let's think step by step."
Across selected LLM models.
"""

# Set XGBoost/OpenMP thread limits to prevent over-subscription
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"
os.environ["XGBOOST_NUM_THREADS"] = "16"

import traceback

# Override RESULTS_DIR before importing benchmark_runner
from src.benchmarks import benchmark_runner
benchmark_runner.RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results" / "XGBOOST_ABLATION")

from src.benchmarks.benchmark_runner import run_benchmark
from src.config.model_registry import MODEL_CONFIGS, require_ollama_server

API_KEY = os.getenv("OPENAI_HK_API_KEY")
TASK_ID = 190411
BENCHMARK = "xgboost"

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

PROMPT_VARIANTS = {
    "baseline": "",
    "step_by_step": "Let's think step by step.",
}


if __name__ == "__main__":
    if not API_KEY:
        raise EnvironmentError("OPENAI_HK_API_KEY is not set. Export it before running ablation.")

    TARGET_MODELS = [model_key for model_key in TARGET_MODELS if model_key in MODEL_CONFIGS]
    require_ollama_server(TARGET_MODELS)

    print(
        f"Running XGBoost ablation on task_id={TASK_ID} "
        f"for {len(TARGET_MODELS)} models and {len(PROMPT_VARIANTS)} prompt variants..."
    )
    summary = []

    for model_key in TARGET_MODELS:
        cfg = MODEL_CONFIGS[model_key]
        for variant, cot_prompt in PROMPT_VARIANTS.items():
            print(f"\n{'#' * 80}")
            print(f"MODEL={model_key} | VARIANT={variant} | BENCHMARK={BENCHMARK} | TASK_ID={TASK_ID}")
            print(f"{'#' * 80}")
            try:
                artifact = run_benchmark(
                    model_name=BENCHMARK,
                    task_id=TASK_ID,
                    api_key=cfg.get("api_key_override", API_KEY),
                    llm_model=cfg["id"],
                    base_url=cfg["base_url"],
                    model_key=model_key,
                    cot_prompt=cot_prompt,
                    prompt_variant=variant,
                )
                summary.append(
                    {
                        "model": model_key,
                        "variant": variant,
                        "status": "SUCCESS",
                        "best_performer": artifact.get("best_performer"),
                        "cost_usd": artifact.get("total_cost_usd"),
                    }
                )
            except Exception as e:
                traceback.print_exc()
                summary.append(
                    {
                        "model": model_key,
                        "variant": variant,
                        "status": f"FAILED: {e}",
                        "best_performer": None,
                        "cost_usd": None,
                    }
                )

    print(f"\n{'=' * 100}")
    print("XGBOOST ABLATION SUMMARY")
    print(f"{'=' * 100}")
    for row in summary:
        print(
            f"{row['model']:<20} | {row['variant']:<15} | {row['status']:<28} | "
            f"best={row['best_performer']} | cost={row['cost_usd']}"
        )
    print(f"{'=' * 100}\n")
