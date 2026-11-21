#!/usr/bin/env python3
"""Report mltest20 Option B completion and list pending jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "examples" / "benchmark"))

from experiment_utils import default_llm_max_tokens  # noqa: E402
from run_toy_categorical_optimizers import variant_is_complete  # noqa: E402

N_TRIALS_DEFAULT = 80
OUTPUT_ROOT_DEFAULT = "benchmark_data/real_hpo_v2_mltest20_algoB"

# dataset:algorithm (Option B)
TASKS = [
    ("yeast", "lightgbm", "cls"),
    ("car", "extra_trees", "cls"),
    ("PhishingWebsites", "xgboost", "cls"),
    ("segment", "catboost", "cls"),
    ("boston", "extra_trees", "rgs"),
    ("tecator", "extra_trees", "rgs"),
    ("socmob", "extra_trees", "rgs"),
    ("pol", "xgboost", "rgs"),
]

CLASSICAL = ("smac", "tpe", "cmaes")
P16_VARIANT = "ptllm_tpe__P16__H2"

LORA_MODELS = {
    "s0": "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s0",
    "s1a_ep2p5": "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep2p5",
    "s1b_ep2p5": "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep2p5",
    "s1a_ep7p5": "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1a_ep7p5",
    "s1b_ep7p5": "/root/blockdata/model-weights/llamafactory_qwen35_9b_lora_p5_s1b_ep7p5",
}

LORA_RESPFMT = {
    "s0": "json_only",
    "s1a_ep2p5": "json_only",
    "s1a_ep7p5": "json_only",
    "s1b_ep2p5": "rationale_json",
    "s1b_ep7p5": "rationale_json",
}

def _lora_maxtok(model: str) -> int:
    return default_llm_max_tokens(LORA_RESPFMT[model])


def adapter_ready(path: str) -> bool:
    return (Path(path) / "adapter_model.safetensors").is_file()


def dataset_dir(output_root: Path, task_type: str, dataset: str) -> Path:
    return output_root / task_type / dataset


def job_done(output_root: Path, task_type: str, dataset: str, variant: str, n_trials: int) -> bool:
    d = dataset_dir(output_root, task_type, dataset)
    return variant_is_complete(d, variant, n_trials)


def summarize_group(
    output_root: Path,
    variant: str,
    n_trials: int,
    label: str,
) -> dict:
    done = 0
    missing = []
    for ds, _algo, tt in TASKS:
        if job_done(output_root, tt, ds, variant, n_trials):
            done += 1
        else:
            missing.append(ds)
    return {"label": label, "variant": variant, "done": done, "total": len(TASKS), "missing": missing}


def build_pending(n_trials: int, output_root: Path, lora_models: dict[str, str]) -> list[dict]:
    pending: list[dict] = []

    for opt in CLASSICAL:
        for ds, algo, tt in TASKS:
            if not job_done(output_root, tt, ds, opt, n_trials):
                pending.append(
                    {
                        "kind": "classical",
                        "optimizer": opt,
                        "dataset": ds,
                        "algorithm": algo,
                        "task_type": tt,
                        "output_root": str(output_root),
                    }
                )

    for ds, algo, tt in TASKS:
        if not job_done(output_root, tt, ds, P16_VARIANT, n_trials):
            pending.append(
                {
                    "kind": "p16",
                    "variant": P16_VARIANT,
                    "dataset": ds,
                    "algorithm": algo,
                    "task_type": tt,
                    "output_root": str(output_root),
                }
            )

    for model, adapter in lora_models.items():
        if not adapter_ready(adapter):
            continue
        out = output_root.parent / f"{output_root.name}_p5_{model}"
        for hybrid in ("H0", "H4"):
            variant = f"ptllm_tpe_qwen__P5__{hybrid}"
            for ds, algo, tt in TASKS:
                if not job_done(out, tt, ds, variant, n_trials):
                    pending.append(
                        {
                            "kind": "lora",
                            "model": model,
                            "hybrid": hybrid,
                            "variant": variant,
                            "dataset": ds,
                            "algorithm": algo,
                            "task_type": tt,
                            "adapter": adapter,
                            "output_root": str(out),
                            "llm_response_format": LORA_RESPFMT[model],
                            "llm_max_tokens": _lora_maxtok(model),
                        }
                    )
    return pending


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=Path(OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--json", action="store_true", help="Emit pending jobs as JSON lines")
    parser.add_argument("--pending-only", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = REPO / output_root

    lora_models = {k: v for k, v in LORA_MODELS.items() if adapter_ready(v)}
    missing_adapters = [k for k in LORA_MODELS if k not in lora_models]

    if not args.pending_only:
        print(f"mltest20 Option B status (n_trials={args.n_trials})")
        print(f"output_root={output_root}")
        print()
        print("Classical baselines:")
        for opt in CLASSICAL:
            s = summarize_group(output_root, opt, args.n_trials, opt)
            print(f"  {opt}: {s['done']}/{s['total']}", "OK" if not s["missing"] else f"missing {s['missing']}")

        s = summarize_group(output_root, P16_VARIANT, args.n_trials, "P16/H2 gpt")
        print(f"\nP16/H2 (gpt-4o-mini): {s['done']}/{s['total']}", "OK" if not s["missing"] else f"missing {s['missing']}")

        print("\nLoRA P5/H0 + P5/H4:")
        for model in LORA_MODELS:
            adapter = LORA_MODELS[model]
            if model not in lora_models:
                print(f"  {model}: SKIP (no adapter at {adapter})")
                continue
            out = output_root.parent / f"{output_root.name}_p5_{model}"
            for hybrid in ("H0", "H4"):
                variant = f"ptllm_tpe_qwen__P5__{hybrid}"
                s = summarize_group(out, variant, args.n_trials, f"{model} P5/{hybrid}")
                print(
                    f"  {model} P5/{hybrid}: {s['done']}/{s['total']}",
                    "OK" if not s["missing"] else f"missing {s['missing']}",
                )

        if missing_adapters:
            print(f"\nAdapters not yet trained: {', '.join(missing_adapters)}")

    pending = build_pending(args.n_trials, output_root, LORA_MODELS)
    if args.json:
        for job in pending:
            print(json.dumps(job))
    elif not args.pending_only:
        print(f"\nPending jobs: {len(pending)}")
        by_kind: dict[str, int] = {}
        for job in pending:
            by_kind[job["kind"]] = by_kind.get(job["kind"], 0) + 1
        for kind, count in sorted(by_kind.items()):
            print(f"  {kind}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
