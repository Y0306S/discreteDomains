#!/usr/bin/env python3
"""Verify P17-P25 each add exactly one prompt block on top of P5."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mindware.components.optimizers.llm_prompt_blocks import (  # noqa: E402
    build_playbook_run_context_block,
    build_tuning_playbook_system_extension,
    categorical_history_gate,
    format_tpe_hint_block,
    space_rel_diff,
)
from mindware.components.optimizers.pt4opt_llm_optimizer import build_history_prompt_block  # noqa: E402

BLOCK_MARKERS = {
    "budget_context": "Run budget:",
    "explore_refine_summary": "Tuning phase:",
    "best_so_far_trajectory": "Best-so-far trajectory",
    "hp_marginals": "Hyperparameter marginals",
    "source_tags": "source=",
    "coverage_map": "Coverage map:",
    "rationale_diversity_summary": "Run summary extensions:",
    "tuning_playbook_system": "Tuning principles",
    "playbook_run_context": "Playbook run context:",
    "tpe_hint": "TPE sampler suggestion",
}

VARIANT_BLOCKS = {
    "P17": "budget_context",
    "P18": "explore_refine_summary",
    "P19": "best_so_far_trajectory",
    "P20": "hp_marginals",
    "P21": "source_tags",
    "P22": "tuning_playbook_system",
    "P23": "coverage_map",
    "P24": "rationale_diversity_summary",
    "P25": "tpe_hint",
    "P26": "playbook_run_context",
}


def _mock_cs() -> ConfigurationSpace:
    cs = ConfigurationSpace()
    cs.add_hyperparameter(CategoricalHyperparameter("algorithm", ["random_forest"]))
    cs.add_hyperparameter(CategoricalHyperparameter("random_forest:bootstrap", ["True", "False"]))
    cs.add_hyperparameter(
        CategoricalHyperparameter("random_forest:criterion", ["gini", "entropy", "log_loss"])
    )
    cs.add_hyperparameter(UniformFloatHyperparameter("random_forest:max_features", 0.0, 1.0))
    cs.add_hyperparameter(UniformIntegerHyperparameter("random_forest:min_samples_leaf", 1, 20))
    cs.add_hyperparameter(UniformIntegerHyperparameter("random_forest:min_samples_split", 2, 20))
    return cs


class MockCfg:
    def __init__(self, d):
        self._d = d

    def get_dictionary(self):
        return self._d


def _mock_history(n: int = 12):
    configs = []
    perfs = []
    for i in range(n):
        configs.append(
            MockCfg(
                {
                    "algorithm": "random_forest",
                    "random_forest:bootstrap": "True" if i % 2 == 0 else "False",
                    "random_forest:criterion": ["gini", "entropy", "log_loss"][i % 3],
                    "random_forest:max_features": 0.1 + 0.07 * i,
                    "random_forest:min_samples_leaf": 1 + (i % 5),
                    "random_forest:min_samples_split": 2 + (i % 4),
                }
            )
        )
        perfs.append(0.4 + 0.05 * i + (0.1 if i % 4 == 0 else 0))
    return configs, perfs


def build_p5(**kwargs) -> str:
    configs, perfs = _mock_history()
    text, _ = build_history_prompt_block(
        configs,
        perfs,
        "random_forest",
        _mock_cs(),
        20,
        5,
        10,
        5,
        True,
        prompt_blocks=None,
        budget_total=80,
        **kwargs,
    )
    return text


def build_variant(block_key: str, **kwargs) -> str:
    blocks = {block_key: True}
    if block_key == "playbook_run_context":
        blocks["tuning_playbook_system"] = True
    configs, perfs = _mock_history()
    text, _ = build_history_prompt_block(
        configs,
        perfs,
        "random_forest",
        _mock_cs(),
        20,
        5,
        10,
        5,
        True,
        prompt_blocks=blocks,
        budget_total=80,
        **kwargs,
    )
    return text


def verify_history_blocks() -> list[str]:
    errors = []
    p5 = build_p5(
        llm_history=[{"trial": 10, "rationale": "test", "improved": False}],
        trial_sources={i: ("llm" if i % 2 else "tpe") for i in range(1, 13)},
    )

    for marker in BLOCK_MARKERS.values():
        if marker in p5 and marker not in ("source=",):
            errors.append(f"P5 baseline unexpectedly contains marker {marker!r}")

    for vid, block_key in VARIANT_BLOCKS.items():
        if block_key == "tuning_playbook_system":
            sys_text = build_tuning_playbook_system_extension("random_forest", {"n_samples": 400})
            if BLOCK_MARKERS[block_key] not in sys_text:
                errors.append(f"{vid}: missing system block marker {BLOCK_MARKERS[block_key]!r}")
            continue
        if block_key == "tpe_hint":
            hint = format_tpe_hint_block({"random_forest:max_features": 0.3})
            if BLOCK_MARKERS[block_key] not in hint:
                errors.append(f"{vid}: missing TPE hint marker")
            continue
        if block_key == "playbook_run_context":
            ctx = build_playbook_run_context_block(
                [{"trial": i, "score": 0.4 + 0.05 * i} for i in range(1, 13)],
                budget_total=80,
            )
            if BLOCK_MARKERS[block_key] not in ctx:
                errors.append(f"{vid}: missing playbook run context marker")
            continue

        extra_kwargs = {}
        if block_key == "source_tags":
            extra_kwargs["trial_sources"] = {i: ("llm" if i % 2 else "tpe") for i in range(1, 13)}
        if block_key == "rationale_diversity_summary":
            extra_kwargs["llm_history"] = [{"trial": 11, "rationale": "increase leaf", "improved": True}]

        variant = build_variant(block_key, **extra_kwargs)
        marker = BLOCK_MARKERS[block_key]
        if marker not in variant:
            errors.append(f"{vid}: missing expected marker {marker!r}")

        for other_key, other_marker in BLOCK_MARKERS.items():
            if other_key == block_key or other_key in ("tuning_playbook_system", "tpe_hint"):
                continue
            if block_key == "playbook_run_context" and other_key == "budget_context":
                continue
            if other_marker in variant:
                errors.append(f"{vid}: leaked marker {other_marker!r} from {other_key}")

        if block_key == "explore_refine_summary" and "top trials vs worst trials" in variant.lower():
            errors.append(f"{vid}: explore_refine should replace numeric HP trends")

    return errors


def verify_conversation_logs(log_root: Path) -> list[str]:
    errors = []
    if not log_root.is_dir():
        return [f"log root missing: {log_root}"]

    for vid, block_key in VARIANT_BLOCKS.items():
        if block_key in ("tuning_playbook_system", "tpe_hint"):
            pattern = f"*{vid}*" if vid != "P25" else "*P25*"
        else:
            pattern = f"*ptllm__{vid}*" if vid not in ("P21", "P25") else f"*ptllm_tpe__{vid}*"

        logs = sorted(log_root.glob(f"**/{pattern}/**/llm_conversations/trial_*.json"))
        if not logs:
            errors.append(f"{vid}: no conversation logs under {log_root}")
            continue

        found = False
        for log_path in logs:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if not data.get("llm_input", {}).get("include_history"):
                continue
            found = True
            inp = data["llm_input"]
            user = inp.get("user_prompt", "")
            system = inp.get("system_prompt", "")
            marker = BLOCK_MARKERS[block_key]

            if block_key == "tuning_playbook_system":
                if marker not in system:
                    errors.append(f"{vid}: {log_path.name} missing playbook in system prompt")
            elif block_key == "playbook_run_context":
                if "Tuning principles" not in system:
                    errors.append(f"{vid}: {log_path.name} missing inherited playbook in system prompt")
                if marker not in user:
                    errors.append(f"{vid}: {log_path.name} missing {marker!r} in user prompt")
            elif block_key == "tpe_hint":
                if marker not in user:
                    errors.append(f"{vid}: {log_path.name} missing TPE hint in user prompt")
            elif marker not in user:
                errors.append(f"{vid}: {log_path.name} missing {marker!r} in user prompt")
            break

        if not found:
            errors.append(f"{vid}: no trial with include_history=true in logs")

    return errors


def verify_blame_heuristics() -> list[str]:
    errors = []
    log_spec = {"lo": 0.01, "hi": 2.0, "log": True}
    lin_spec = {"lo": 50.0, "hi": 500.0, "log": False}
    # 0.01 vs 0.1 is a large log-span move (~0.5 of [log 0.01, log 2])
    log_rel = space_rel_diff(0.01, 0.1, log_spec)
    lin_as_log = space_rel_diff(0.01, 0.1, lin_spec)  # would be tiny on a 50-500 scale
    if log_rel < 0.3:
        errors.append(f"log-space 0.01 vs 0.1 should be a large gap, got {log_rel:.3f}")
    if space_rel_diff(100, 110, lin_spec) >= 0.15:
        errors.append("linear 100 vs 110 on [50,500] should be below 0.15")
    if space_rel_diff(50, 400, lin_spec) < 0.15:
        errors.append("linear 50 vs 400 on [50,500] should be above 0.15")

    # cat gate: underpowered
    weak = [
        {"trial": i, "score": 0.9 if i < 3 else 0.5, "config": {"hp": "A" if i < 3 else "B"}}
        for i in range(8)
    ]
    if categorical_history_gate(weak, "hp") is not None:
        errors.append("cat gate should skip n<5")

    # cat gate: both values in top 25%
    both_top = []
    for i in range(20):
        both_top.append(
            {
                "trial": i + 1,
                "score": 1.0 - 0.01 * i,
                "config": {"hp": "A" if i % 2 == 0 else "B"},
            }
        )
    if categorical_history_gate(both_top, "hp") is not None:
        errors.append("cat gate should skip when both values appear in top 25%")

    # cat-neither: both in top -> prefer-only (emit_avoid=False), not dropped
    mixed_top = [
        {"trial": 1, "score": 0.95, "config": {"hp": "A"}},
        {"trial": 2, "score": 0.94, "config": {"hp": "B"}},
        {"trial": 3, "score": 0.93, "config": {"hp": "A"}},
        {"trial": 4, "score": 0.92, "config": {"hp": "B"}},
        {"trial": 5, "score": 0.91, "config": {"hp": "A"}},
        {"trial": 6, "score": 0.85, "config": {"hp": "B"}},
        {"trial": 7, "score": 0.84, "config": {"hp": "B"}},
        {"trial": 8, "score": 0.83, "config": {"hp": "B"}},
        {"trial": 9, "score": 0.82, "config": {"hp": "B"}},
        {"trial": 10, "score": 0.81, "config": {"hp": "B"}},
    ]
    neither = categorical_history_gate(
        mixed_top, "hp", min_n=3, score_delta=0.02, avoid_in_top_policy="prefer_only"
    )
    if neither is None or neither.get("emit_avoid") is not False:
        errors.append(f"cat-neither should keep prefer-only when both in top, got {neither}")

    # cat gate: sure case — A dominates top, B is only in the bottom
    sure = []
    for i in range(12):
        sure.append({"trial": i + 1, "score": 0.9 - 0.01 * i, "config": {"hp": "A"}})
    for i in range(12):
        sure.append({"trial": 13 + i, "score": 0.4 - 0.01 * i, "config": {"hp": "B"}})
    gated = categorical_history_gate(sure, "hp")
    if gated is None or gated["best_val"] != "A" or gated["worst_val"] != "B":
        errors.append(f"cat gate should prefer A over B when sure, got {gated}")

    _ = lin_as_log
    return errors


def main():
    log_root = REPO / "benchmark_data" / "p5_ablation_dryrun"
    hist_errors = verify_history_blocks()
    heur_errors = verify_blame_heuristics()
    log_errors = verify_conversation_logs(log_root) if log_root.is_dir() else []

    if hist_errors:
        print("[FAIL] history block checks:")
        for e in hist_errors:
            print(" ", e)
    else:
        print("[OK] history block isolation (programmatic)")

    if heur_errors:
        print("[FAIL] blame heuristic checks:")
        for e in heur_errors:
            print(" ", e)
    else:
        print("[OK] log-scale numeric + conservative cat gate")

    if log_errors:
        print("[FAIL] conversation log checks:")
        for e in log_errors:
            print(" ", e)
    else:
        print("[OK] conversation logs (dry-run)")

    if hist_errors or heur_errors or log_errors:
        sys.exit(1)
    print("[Done] all P5 ablation block checks passed")


if __name__ == "__main__":
    main()
