"""Pick best prompt or hybrid from pilot-dataset results."""

import argparse
import csv
from pathlib import Path

from experiment_utils import best_run_dir, list_v2_dataset_jobs, score_higher_is_better


def fnum(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_variant_scores(base_dir: Path, name_prefix: str, jobs=None):
    scores = {}
    if jobs is None:
        jobs = list_v2_dataset_jobs("pilot")
    for task_type, dataset in jobs:
        key = f"{task_type}/{dataset}"
        ds_dir = base_dir / task_type / dataset
        if not ds_dir.is_dir():
            continue
        for opt_dir in sorted(ds_dir.iterdir()):
            if not opt_dir.is_dir() or not opt_dir.name.startswith(name_prefix):
                continue
            run_dir = best_run_dir(opt_dir)
            if run_dir is None:
                continue
            trajs = sorted(run_dir.glob("*_traj.csv"))
            if not trajs:
                continue
            rows = list(csv.DictReader(trajs[-1].open()))
            if not rows:
                continue
            higher = score_higher_is_better(task_type)
            best = max(fnum(r["best_so_far"]) for r in rows) if higher else min(fnum(r["best_so_far"]) for r in rows)
            scores.setdefault(opt_dir.name, {})[key] = best
    return scores


def avg_rank(scores_by_variant: dict[str, dict[str, float]]) -> dict[str, float]:
    datasets = sorted({ds for m in scores_by_variant.values() for ds in m})
    ranks = {}
    for variant, per_ds in scores_by_variant.items():
        rank_sum = 0.0
        n = 0
        for ds in datasets:
            if ds not in per_ds:
                continue
            task_type = ds.split("/", 1)[0]
            higher = score_higher_is_better(task_type)
            vals = [(v, m[ds]) for v, m in scores_by_variant.items() if ds in m]
            ordered = sorted(vals, key=lambda x: x[1], reverse=higher)
            i = 0
            while i < len(ordered):
                j = i
                while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
                    j += 1
                avg = (i + 1 + j + 1) / 2
                for k in range(i, j + 1):
                    if ordered[k][0] == variant:
                        rank_sum += avg
                        n += 1
                i = j + 1
        ranks[variant] = rank_sum / n if n else float("nan")
    return ranks


def prompt_scores_from_ptllm(ptllm_scores):
    """Stage 1 ranks prompts from pure-ptllm runs only."""
    by_prompt = {}
    for name, per_ds in ptllm_scores.items():
        if not name.startswith("ptllm__"):
            continue
        prompt = name.split("__", 1)[1]
        by_prompt[prompt] = dict(per_ds)
    return by_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default="benchmark_data/real_hpo_v2")
    parser.add_argument("--mode", choices=["prompt", "hybrid"], default="prompt")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--exclude-datasets",
        default="",
        help="Comma-separated dataset names to skip (e.g. covertype).",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="For hybrid mode: only rank ptllm_tpe variants with this prompt (e.g. P5).",
    )
    args = parser.parse_args()

    exclude = {x.strip() for x in args.exclude_datasets.split(",") if x.strip()}

    def pilot_jobs():
        for task_type, dataset in list_v2_dataset_jobs("pilot"):
            if dataset in exclude:
                continue
            yield task_type, dataset

    base = Path(args.base_dir).resolve()
    jobs = list(pilot_jobs())

    if args.mode == "prompt":
        merged = prompt_scores_from_ptllm(collect_variant_scores(base, "ptllm__", jobs))
        ranks = avg_rank(merged)
        ordered = sorted(ranks.items(), key=lambda kv: kv[1])
        print("Best prompts on pilot (lower avg_rank better):")
        for prompt, rank in ordered[: args.top]:
            print(f"  {prompt}: avg_rank={rank:.3f}")
        if ordered:
            print(
                f"\nRollout: STAGE=stage1_rollout PROMPT={ordered[0][0]} "
                f"bash examples/benchmark/run_option2_v2.sh"
            )
        return

    scores = collect_variant_scores(base, "ptllm_tpe__", jobs)
    if args.prompt:
        prefix = f"ptllm_tpe__{args.prompt}__"
        scores = {k: v for k, v in scores.items() if k.startswith(prefix)}
    ranks = avg_rank(scores)
    ordered = sorted(ranks.items(), key=lambda kv: kv[1])
    print("Best ptllm_tpe hybrids on pilot (lower avg_rank better):")
    for name, rank in ordered[: args.top]:
        parts = name.split("__")
        prompt = parts[1] if len(parts) > 1 else "?"
        hybrid = parts[2] if len(parts) > 2 else "H0"
        print(f"  {name}: avg_rank={rank:.3f}  (prompt={prompt} hybrid={hybrid})")
    if ordered:
        parts = ordered[0][0].split("__")
        prompt = parts[1] if len(parts) > 1 else "P0"
        hybrid = parts[2] if len(parts) > 2 else "H0"
        print(
            f"\nRollout: STAGE=stage2_rollout PROMPT={prompt} HYBRID={hybrid} "
            f"bash examples/benchmark/run_option2_v2.sh"
        )


if __name__ == "__main__":
    main()
