#!/usr/bin/env python3
"""Dump P5+ GRPO prompt-only examples from frozen BO prefixes (no LLM, no eval).

Default mix schedule (one prompt per prefix):
  k=0,5,10      rs_warm  (P5plusRS / v4 / v5)
  k=20,30,40    normal   (P5)
  k=50,70,90    explore  (P5plusA5Explore)
  k=60,80,100   exploit  (P5plusA5Exploit)

Completions and rewards stay null for a later 4-sibling sampler.

  python examples/benchmark/export_p5plus_rl_prefix_prompts.py --dry-run
  python examples/benchmark/export_p5plus_rl_prefix_prompts.py --rs-version v4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmark"))

from bo_history_utils import (  # noqa: E402
    DEFAULT_BO_HISTORY_ROOT,
    bo_history_path,
    load_bo_observations,
    load_light16_tasks,
    seed_optimizer_with_bo,
)
from experiment_utils import apply_prompt_settings, default_llm_max_tokens  # noqa: E402
from mindware import CASH  # noqa: E402
from mindware.components.optimizers.experience_gate import ExperienceGate, ExplorePolicy  # noqa: E402
from mindware.components.optimizers.pt4opt_llm_optimizer import PT4OptLLMOptimizer  # noqa: E402
from mindware.components.utils.constants import CLASSIFICATION, REGRESSION  # noqa: E402
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase  # noqa: E402
from mindware.modules.configs import CONFIGS  # noqa: E402
from p5plus_rl_collect_schedule import (  # noqa: E402
    ALL_SCENARIOS,
    MIX_PREFIXES,
    PREFIX_SCENARIO,
    RS_WARM_PROMPT,
    apply_prompt_to_optimizer,
    canonical_scenario,
    default_output_root,
    mix_jobs,
    prompt_id_for_scenario,
)
from run_toy_categorical_optimizers import (  # noqa: E402
    build_optimizer_params,
    expand_optimizer_variants,
    history_scope_dir,
    load_real_data,
    resolve_variant,
)

SCENARIOS = {
    "rs_warm": "P5plusRS",
    "similar": "P5plusRS",
    "normal": "P5",
    "exploit": "P5plusA5Exploit",
    "explore": "P5plusA5Explore",
}

MARKER_NEEDLE = {
    "has_retrieve": "Similar task:",
    "has_self_experience": "Self-run experience",
    "has_coverage_map": "Coverage map:",
}

DEFAULT_OUT = default_output_root("v3")
DEFAULT_TASKS = REPO_ROOT / "benchmark_data/rl_task_selection/light_16.json"
PREFIXES = list(MIX_PREFIXES)


def prompt_markers(user: str) -> Dict[str, bool]:
    text = user or ""
    marks = {key: needle in text for key, needle in MARKER_NEEDLE.items()}
    if "Retrieved experiences from similar past tasks" in text:
        marks["has_retrieve"] = True
    return marks


def markers_ok(scenario: str, marks: Dict[str, bool], *, n_hits: int = 0) -> bool:
    sc = canonical_scenario(scenario)
    if sc in ("similar", "rs_warm"):
        # Empty retrieve is a valid rs_warm example (usable/Spearman gate).
        return (not marks["has_self_experience"]) and (not marks["has_coverage_map"]) and (
            marks["has_retrieve"] or n_hits == 0
        )
    if sc == "exploit":
        return marks["has_self_experience"] and not marks["has_retrieve"] and not marks["has_coverage_map"]
    if sc == "explore":
        return marks["has_coverage_map"] and not marks["has_retrieve"] and not marks["has_self_experience"]
    raise ValueError(f"unknown scenario {scenario!r}")


def hit_stats(meta: Dict[str, Any]) -> Dict[str, Any]:
    hits = list(meta.get("experience_retrieve_hits") or [])
    n_gp = 0
    sources = []
    for h in hits:
        feats = h.get("features") or {}
        if feats.get("surrogate_landscape_sim") is not None:
            n_gp += 1
        src = feats.get("landscape_source")
        if src:
            sources.append(src)
    return {
        "n_hits": len(hits),
        "n_gp_sim": n_gp,
        "landscape_sources": sorted(set(sources)),
        "hit_ids": [h.get("id") for h in hits],
    }


def make_base_args(ns: argparse.Namespace, *, task: Dict[str, Any], prompt_id: str) -> argparse.Namespace:
    args = argparse.Namespace(**vars(ns))
    args.task_type = task["task_type"]
    args.dataset = task["dataset"]
    args.algorithm = [task["algorithm"]]
    args.metric = task.get("metric") or ("bal_acc" if task["task_type"] == "cls" else "mse")
    args.prompt_variant = prompt_id
    args.hybrid_variant = "H0"
    args.optimizer = ["ptllm_qwen"]
    args.llm_backends = ["qwen-sft6"]
    args.llm_mock = True
    args.llm_n_initial_samples = 0
    args.llm_response_format = "rationale_json"
    args.llm_max_tokens = default_llm_max_tokens("rationale_json")
    args.llm_temperature = 1.0
    args.llm_timeout = 180
    args.llm_fallback_on_transport_error = False
    args.llm_response = None
    args.llm_chat_engine = None
    args.llm_model = None
    args.llm_history_window = 20
    args.evaluation = "holdout"
    args.valid_size = 0.35
    args.test_size = 0.2
    args.tpe_startup_trials = 3
    args.llm_tpe_llm_prob = 0.5
    args.llm_tpe_phase_trials = 0
    args.llm_tpe_tpe_startup_trials = 3
    args.llm_tpe_tpe_duplicate_retries = 5
    args.preset_warm_start_json = None
    args.qwen_adapter_path = None
    apply_prompt_settings(args, prompt_id)
    args.llm_n_initial_samples = 0
    return args


def reset_kd_prefix(opt: PT4OptLLMOptimizer, prefix_obs: List[Dict[str, Any]], algo: str) -> None:
    kd = opt.KD
    kd.configs.clear()
    kd.perfs.clear()
    if hasattr(kd, "configs_mab") and algo in kd.configs_mab:
        kd.configs_mab[algo].clear()
        kd.perfs_mab[algo].clear()
    opt.seen_signatures.clear()
    opt.seen_objectives.clear()
    opt._seen_config_count = 0
    opt.trial_sources.clear()
    opt.llm_history.clear()
    seed_optimizer_with_bo(opt, prefix_obs, source="bo_prefix")


def build_cash(args: argparse.Namespace, train_node, dataset_profile: Dict[str, Any], scratch: Path):
    variant_name = expand_optimizer_variants(["ptllm_qwen"], args)[0]
    dispatch, output_name, _backend = resolve_variant(variant_name)
    task_const = CLASSIFICATION if args.task_type == "cls" else REGRESSION
    metric = args.metric
    algorithms = list(args.algorithm)
    algo = algorithms[0]
    history_dir = Path(args.history_dir).resolve()
    scope_dir = history_scope_dir(history_dir, args.task_type, metric, args.dataset, output_name)
    CONFIGS.ORIGINAL_DATA_DIR = str(history_dir)
    CONFIGS.HISTORY_RUN_SCOPE = str(scope_dir)
    KnowledgeDatabase._instance = None
    optimizer_params = build_optimizer_params(args, variant_name, algorithms, dataset_profile)
    optimizer_params["mock_llm"] = True
    cash = CASH(
        include_algorithms=algorithms,
        task_type=task_const,
        metric=metric,
        data_node=train_node,
        evaluation=args.evaluation,
        resampling_params={"test_size": args.valid_size, "shuffle": False},
        optimizer=dispatch,
        time_limit=args.time_limit,
        amount_of_resource=max(args.prefixes) + 8,
        per_run_time_limit=args.per_run_time_limit,
        inner_iter_num_per_iter=1,
        output_dir=str(scratch / output_name),
        seed=args.seed,
        n_jobs=1,
        task_id=f"{args.dataset}_rl_prefix_prompt_{algo}",
        topk=max(args.prefixes),
        optimizer_params=optimizer_params,
    )
    opt: PT4OptLLMOptimizer = cash.optimizer  # type: ignore[assignment]
    opt.n_no_history_samples = 0
    opt.n_initial_samples = 0
    opt.use_mock_llm = True
    return opt, algo, variant_name


def active_scenarios(ns: argparse.Namespace) -> Dict[str, str]:
    rs_ver = str(getattr(ns, "rs_version", "v3") or "v3")
    wanted = getattr(ns, "scenarios", None)
    if not wanted:
        if getattr(ns, "collect_schedule", "rs_mix") == "rs_mix":
            wanted = list(ALL_SCENARIOS)
        else:
            wanted = ["rs_warm"]
    out: Dict[str, str] = {}
    for name in wanted:
        sc = canonical_scenario(name)
        if sc not in ALL_SCENARIOS and name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; choose from {sorted(set(SCENARIOS) | set(ALL_SCENARIOS))}")
        out[sc] = prompt_id_for_scenario(sc, rs_ver)
    return out


def scheduled_jobs(ns: argparse.Namespace, prefixes: Sequence[int]) -> List[Tuple[int, str, str]]:
    rs_ver = str(getattr(ns, "rs_version", "v3") or "v3")
    scenarios = active_scenarios(ns)
    if getattr(ns, "collect_schedule", "rs_mix") == "rs_mix":
        return mix_jobs(prefixes, rs_version=rs_ver, scenarios=list(scenarios))
    jobs: List[Tuple[int, str, str]] = []
    for sc, pid in scenarios.items():
        for k in prefixes:
            jobs.append((int(k), sc, pid))
    return jobs


def export_all(ns: argparse.Namespace) -> Path:
    tasks = load_light16_tasks(Path(ns.tasks_json))
    prefixes = list(ns.prefixes)
    if ns.dry_run:
        tasks = tasks[:1]
        if getattr(ns, "collect_schedule", "rs_mix") == "rs_mix":
            seen = set()
            dry_pref: List[int] = []
            for k in prefixes:
                sc = PREFIX_SCENARIO.get(int(k))
                if sc is None or sc in seen:
                    continue
                seen.add(sc)
                dry_pref.append(int(k))
            prefixes = dry_pref or prefixes[:1]
        else:
            prefixes = [min(prefixes)]
    jobs = scheduled_jobs(ns, prefixes)
    if not jobs:
        raise SystemExit("no collect jobs (check --prefixes vs mix schedule and --scenarios)")
    scenario_names = list(dict.fromkeys(sc for _k, sc, _pid in jobs))

    out_root = Path(ns.output_root).resolve()
    for scenario in scenario_names:
        (out_root / scenario).mkdir(parents=True, exist_ok=True)
        path = out_root / scenario / "prompts.jsonl"
        if path.exists() and not ns.dry_run:
            path.unlink()

    written = {s: 0 for s in scenario_names}
    skipped: List[Dict[str, Any]] = []
    missing_bo: List[str] = []
    samples: Dict[str, Dict[str, Any]] = {}
    n_empty_retrieve = 0
    n_gp = 0

    for task in tasks:
        tt, ds, algo = task["task_type"], task["dataset"], task["algorithm"]
        metric = task.get("metric") or ("bal_acc" if tt == "cls" else "mse")
        bo_path = bo_history_path(tt, ds, algo, metric=metric, root=Path(ns.bo_history_root))
        if not bo_path.is_file():
            missing_bo.append(str(bo_path))
            skipped.append({"reason": "missing_bo", "task": f"{tt}/{ds}/{algo}", "path": str(bo_path)})
            print(f"[skip] missing BO {bo_path}")
            continue
        all_obs = load_bo_observations(bo_path, algorithm=algo, max_n=ns.bo_max_n)
        if len(all_obs) < max(prefixes):
            skipped.append(
                {
                    "reason": "short_bo",
                    "task": f"{tt}/{ds}/{algo}",
                    "n": len(all_obs),
                    "need": max(prefixes),
                }
            )
            print(f"[skip] short BO {tt}/{ds}/{algo} n={len(all_obs)}")
            continue

        dummy = make_base_args(ns, task=task, prompt_id="P5plusA5Exploit")
        dummy.history_dir = ns.history_dir
        dummy.time_limit = ns.time_limit
        dummy.per_run_time_limit = ns.per_run_time_limit
        dummy.seed = ns.seed
        dummy.prefixes = prefixes
        dummy.bo_max_n = ns.bo_max_n
        _, train_node, _test, dataset_path, profile = load_real_data(dummy)
        profile["task_type"] = "classification" if tt == "cls" else "regression"
        print(f"[export] {tt}/{ds}/{algo} n_obs={len(all_obs)} n_samples={profile.get('n_samples')}")
        scratch = out_root / "_scratch" / f"{tt}_{ds}_{algo}"
        scratch.mkdir(parents=True, exist_ok=True)
        opt, algo_fixed, _variant = build_cash(dummy, train_node, profile, scratch)

        for k, scenario, prompt_id in jobs:
            sc_args = make_base_args(ns, task=task, prompt_id=prompt_id)
            apply_prompt_to_optimizer(opt, sc_args)
            out_path = out_root / scenario / "prompts.jsonl"
            reset_kd_prefix(opt, all_obs[:k], algo_fixed)
            opt._experience_gate = ExperienceGate(
                mode=opt.experience_self_gate, cap=opt.experience_self_start_trial
            )
            opt._explore_policy = ExplorePolicy(
                mode=opt.experience_explore_mode,
                seed=int(getattr(opt, "seed", 1) or 1),
                p_default=opt.experience_explore_prob,
            )
            system, user = opt._build_history_prompts(include_history=True, current_trial=int(k))
            meta = dict(getattr(opt, "_last_history_meta", None) or {})
            stats = hit_stats(meta)
            marks = prompt_markers(user)
            rec = {
                "schema": "p5plus_rl_mix_v1",
                "collect_schedule": getattr(ns, "collect_schedule", "rs_mix"),
                "rs_version": getattr(ns, "rs_version", "v3"),
                "scenario": scenario,
                "group_id": f"{tt}:{ds}:{algo}:{scenario}:k{k}",
                "prefix_k": k,
                "dataset": ds,
                "task_type": tt,
                "algorithm": algo,
                "metric": metric,
                "prompt_variant": prompt_id,
                "experience_bank_path": getattr(sc_args, "llm_experience_bank_path", None),
                "experience_retrieve_ranker": getattr(sc_args, "llm_experience_retrieve_ranker", None),
                "experience_retrieve_rerank_trial": getattr(sc_args, "llm_experience_retrieve_rerank_trial", None),
                "bo_history_path": str(bo_path),
                "dataset_path": str(dataset_path),
                "n_successes_before": k,
                "n_hits": stats["n_hits"],
                "n_gp_sim": stats["n_gp_sim"],
                "landscape_sources": stats["landscape_sources"],
                "hit_ids": stats["hit_ids"],
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "prompt_markers": marks,
                "history_meta": meta,
                "completion": None,
                "reward": None,
                "n_siblings_planned": 4,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if not markers_ok(scenario, marks, n_hits=int(stats["n_hits"])):
                skipped.append(
                    {
                        "reason": "marker_fail",
                        "group_id": rec["group_id"],
                        "markers": marks,
                        "n_hits": stats["n_hits"],
                    }
                )
                print(f"[skip] markers {rec['group_id']} {marks} n_hits={stats['n_hits']}")
                continue
            if canonical_scenario(scenario) == "rs_warm" and stats["n_hits"] == 0:
                n_empty_retrieve += 1
            if stats["n_gp_sim"]:
                n_gp += 1
            if ns.dry_run:
                samples[scenario] = rec
                print(
                    f"\n===== DRY_RUN {scenario} {rec['group_id']} chars={len(user)} "
                    f"hits={stats['n_hits']} gp={stats['n_gp_sim']} src={stats['landscape_sources']} ====="
                )
                print(user[:1800])
                if len(user) > 1800:
                    print("...[truncated]...")
                continue
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True, default=str) + "\n")
            written[scenario] += 1
            if scenario not in samples:
                samples[scenario] = rec

        print(f"[export] done {tt}/{ds}/{algo} written={dict(written)}")
        KnowledgeDatabase._instance = None
        try:
            import shutil

            shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass

    manifest = {
        "schema": "p5plus_rl_mix_v1",
        "dry_run": bool(ns.dry_run),
        "collect_schedule": getattr(ns, "collect_schedule", "rs_mix"),
        "rs_version": getattr(ns, "rs_version", "v3"),
        "jobs": [{"prefix_k": k, "scenario": sc, "prompt_variant": pid} for k, sc, pid in jobs],
        "tasks_json": str(ns.tasks_json),
        "prefixes": prefixes,
        "scenarios": scenario_names,
        "n_siblings_planned": 4,
        "written": written,
        "n_written": sum(written.values()),
        "n_skipped": len(skipped),
        "n_empty_retrieve": n_empty_retrieve,
        "n_gp_records": n_gp,
        "missing_bo": missing_bo,
        "skipped": skipped,
        "output_root": str(out_root),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    man_path = out_root / "manifest.json"
    if not ns.dry_run:
        man_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"[export] wrote {manifest['n_written']} records -> {out_root}")
        print(f"[export] empty_retrieve={n_empty_retrieve} gp_records={n_gp}")
        print(f"[export] manifest -> {man_path}")
    else:
        print(
            f"[export] DRY_RUN written={written} skipped={len(skipped)} "
            f"empty_retrieve={n_empty_retrieve} gp={n_gp}"
        )
    return man_path


def inspect_outputs(out_root: Path, scenarios: Optional[Dict[str, str]] = None) -> None:
    """Sanity-check markers, counts, and one sample prompt per scenario."""
    scenarios = scenarios or SCENARIOS
    print("\n===== inspect =====")
    total = 0
    for scenario in scenarios:
        path = out_root / scenario / "prompts.jsonl"
        if not path.is_file():
            print(f"[inspect] MISSING {path}")
            continue
        recs = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        total += len(recs)
        bad = []
        n_gp = n_empty = 0
        for rec in recs:
            marks = rec.get("prompt_markers") or prompt_markers(
                next((m["content"] for m in rec.get("messages", []) if m.get("role") == "user"), "")
            )
            n_hits = int(rec.get("n_hits") or 0)
            if not markers_ok(scenario, marks, n_hits=n_hits):
                bad.append(rec.get("group_id"))
            if rec.get("completion") is not None or rec.get("reward") is not None:
                bad.append(f"{rec.get('group_id')}:non_null_completion")
            n_gp += int(rec.get("n_gp_sim") or 0) > 0
            n_empty += int(n_hits == 0 and canonical_scenario(scenario) == "rs_warm")
        print(
            f"[inspect] {scenario}: n={len(recs)} gp={n_gp} empty={n_empty} "
            f"marker_fail={len(bad)} sample={recs[0]['group_id'] if recs else None}"
        )
        if recs:
            user = next((m["content"] for m in recs[0]["messages"] if m["role"] == "user"), "")
            print(f"----- {scenario} user prompt (first 900 chars) -----")
            print(user[:900])
            if len(user) > 900:
                print("...[truncated]...")
        if bad:
            print(f"[inspect] BAD {scenario}: {bad[:8]}")
    print(f"[inspect] total={total}")
    man = out_root / "manifest.json"
    if man.is_file():
        payload = json.loads(man.read_text(encoding="utf-8"))
        print(
            f"[inspect] manifest written={payload.get('written')} "
            f"skipped={payload.get('n_skipped')} empty_retrieve={payload.get('n_empty_retrieve')} "
            f"gp={payload.get('n_gp_records')}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export P5+ GRPO prompt-only prefix examples")
    p.add_argument("--tasks-json", default=str(DEFAULT_TASKS))
    p.add_argument("--output-root", default=None)
    p.add_argument("--bo-history-root", default=str(DEFAULT_BO_HISTORY_ROOT))
    p.add_argument("--history-dir", default=str(REPO_ROOT / "history_data"))
    p.add_argument("--prefixes", type=int, nargs="+", default=PREFIXES)
    p.add_argument(
        "--collect-schedule",
        choices=["rs_mix", "cartesian"],
        default="rs_mix",
        help="rs_mix: one scenario per prefix (0/5/10 rs_warm, 20/30/40 P5, 50/70/90 explore, 60/80/100 exploit).",
    )
    p.add_argument("--rs-version", choices=sorted(RS_WARM_PROMPT), default="v3")
    p.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        choices=sorted(set(SCENARIOS) | set(ALL_SCENARIOS)),
        help="Filter mix slots. Default: all mix scenarios (rs_warm, normal, explore, exploit).",
    )
    p.add_argument("--bo-max-n", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time-limit", type=int, default=600)
    p.add_argument("--per-run-time-limit", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inspect-only", action="store_true")
    ns = p.parse_args()
    if not ns.output_root:
        ns.output_root = str(default_output_root(ns.rs_version))
    return ns


def main() -> int:
    os.chdir(REPO_ROOT)
    ns = parse_args()
    env_dry = str(os.environ.get("DRY_RUN", "")).strip().lower()
    if env_dry in ("1", "true", "yes"):
        ns.dry_run = True
    if ns.inspect_only:
        inspect_outputs(Path(ns.output_root).resolve(), active_scenarios(ns))
        return 0
    export_all(ns)
    if not ns.dry_run:
        inspect_outputs(Path(ns.output_root).resolve(), active_scenarios(ns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
