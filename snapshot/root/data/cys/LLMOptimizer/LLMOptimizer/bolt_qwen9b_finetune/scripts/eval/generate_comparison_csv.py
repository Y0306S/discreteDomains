#!/usr/bin/env python3
"""Generate model comparison CSV from OpenBox evaluation results."""

import json
import sys
from pathlib import Path

EVAL_DIR = Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/eval_runs")
REPORTS_DIR = EVAL_DIR / "reports"

# Models to include (must match output JSON filenames)
MODELS = [
    ("claude_opus_4_8", "claude_opus_4_8_test_openbox_eval.json"),
    ("gemini_3.1_pro_preview", "gemini_3.1_pro_preview_test_openbox_eval.json"),
    ("gemini_31_flash", "gemini_31_flash_test_openbox_eval.json"),
    ("gpt_5.5", "gpt_5.5_test_openbox_eval.json"),
    ("qwen_base", "qwen_base_test_openbox_eval.json"),
    ("qwen_base_full_sft_4", "qwen_base_full_sft_4_test_openbox_eval.json"),
    ("qwen_base_lora_sft", "qwen_base_lora_sft_test_openbox_eval.json"),
    ("qwen_base_lora_sft_3", "qwen_base_lora_sft_3_test_openbox_eval.json"),
    ("qwen_base_lora_sft_2_dl", "qwen_base_lora_sft_2_dl_test_openbox_eval.json"),
    ("qwen_base_lora_sft_3_dl", "qwen_base_lora_sft_3_dl_test_openbox_eval.json"),
    ("qwen_base_lora_sft_4_dl", "qwen_base_lora_sft_4_dl_test_openbox_eval.json"),
    ("qwen_base_lora_sft_5_dl", "qwen_base_lora_sft_5_dl_test_openbox_eval.json"),
    ("qwen_base_lora_sft_6_dl", "qwen_base_lora_sft_6_dl_test_openbox_eval.json"),
    ("qwen_base_lora_sft_dl", "qwen_base_lora_sft_dl_test_openbox_eval.json"),
    ("qwen_full_sft_recommended", "qwen_full_sft_recommended_test_openbox_eval.json"),
    ("qwen_lora_recommended5", "qwen_lora_recommended5_test_openbox_eval.json"),
    ("lora_recommended3", "lora_recommended3_test_openbox_eval.json"),
]


def load_eval(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    results = []
    all_task_ids = set()
    model_scores = {}

    for model_name, filename in MODELS:
        data = load_eval(EVAL_DIR / filename)
        if data is None:
            print(f"Missing: {filename}", file=sys.stderr)
            continue

        task_results = data.get("results", [])
        ok = sum(1 for r in task_results if r.get("status") == "ok")
        failed = sum(1 for r in task_results if r.get("status") != "ok")
        nan_failures = sum(
            1
            for r in task_results
            if r.get("status") != "ok" and "nan" in str(r.get("error", "")).lower()
        )

        scores = {}
        for r in task_results:
            if r.get("status") == "ok":
                tid = r.get("task_id") or r.get("dataset", "")
                scores[tid] = r.get("normalized_score", r.get("objective"))
                all_task_ids.add(tid)

        model_scores[model_name] = scores
        results.append({
            "model": model_name,
            "ok": ok,
            "failed": failed,
            "nan_failures": nan_failures,
            "scores": scores,
        })

    if not results:
        print("No evaluation results found.", file=sys.stderr)
        sys.exit(1)

    # Find best model per task
    best_per_task = {}
    for tid in all_task_ids:
        best_score = float("inf")
        best_model = None
        for model_name, scores in model_scores.items():
            if tid in scores:
                # Lower normalized score is better (it's like a distance metric)
                if scores[tid] < best_score:
                    best_score = scores[tid]
                    best_model = model_name
        best_per_task[tid] = (best_model, best_score)

    # Compute mean normalized score for each model
    for r in results:
        scores = r["scores"]
        if scores:
            r["mean_score"] = sum(scores.values()) / len(scores)
        else:
            r["mean_score"] = float("inf")

    # Sort by mean score (ascending = better)
    results.sort(key=lambda x: x["mean_score"])

    # Find the best overall model
    best_overall = results[0]["model"] if results else None
    best_overall_score = results[0]["mean_score"] if results else float("inf")

    # Write CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / "model_comparison_final.csv"

    with open(csv_path, "w") as f:
        f.write("Model,OK/Failed,NaN Failures,Intersection Size,VS Best Model,Mean Normalized Score\n")
        for r in results:
            model_name = r["model"]
            ok_failed = f"{r['ok']}/{r['failed']}"
            nan_f = str(r["nan_failures"])
            intersection = len(r["scores"])

            if best_overall and best_overall != model_name and r["mean_score"] != float("inf"):
                vs_best = f"{(r['mean_score'] - best_overall_score) / best_overall_score * 100:+.2f}%"
            else:
                vs_best = "/"

            mean_score = str(r["mean_score"])
            f.write(f"{model_name},{ok_failed},{nan_f},{intersection},{vs_best},{mean_score}\n")

    print(f"CSV written to: {csv_path}")
    print(f"Best model: {best_overall} (score: {best_overall_score:.6f})")

    # Also print table
    print("\n" + "=" * 80)
    print(f"{'Model':<30} {'OK/Fail':<10} {'Mean Score':<15} {'vs Best':<12}")
    print("=" * 80)
    for r in results:
        model_name = r["model"]
        ok_failed = f"{r['ok']}/{r['failed']}"
        mean_score = f"{r['mean_score']:.6f}" if r["mean_score"] != float("inf") else "N/A"
        if best_overall and best_overall != model_name and r["mean_score"] != float("inf"):
            vs_best = f"{(r['mean_score'] - best_overall_score) / best_overall_score * 100:+.2f}%"
        else:
            vs_best = "/"
        print(f"{model_name:<30} {ok_failed:<10} {mean_score:<15} {vs_best:<12}")


if __name__ == "__main__":
    main()
