#!/usr/bin/env python3
"""Count Label-B factor-line suppressions on cascade top-1 hits (TUNE)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    RetrievalHit,
    cold_features,
    combine_score,
    format_retrieved_block,
    label_b_stats,
    reset_label_b_stats,
    role_signs_from_experience,
    role_signs_from_trials,
    spearman_transfer,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, topk_ids

PREFIX_N = 8
CASCADE_POOL = 12


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    reset_label_b_stats()

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < PREFIX_N + 3:
            continue
        if not is_tune(_ds(exp), split):
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    n_queries = 0
    n_flipped_roles = 0
    n_high_rho_lr_flip = 0
    n_high_rho = 0
    for members in by_group.values():
        if len(members) < 3:
            continue
        for tgt in members:
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:PREFIX_N]
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            scored = {}
            src_by_id = {}
            for src in cands:
                sid = str(src.get("id"))
                src_by_id[sid] = src
                feats = cold_features(q, src)
                pre = spearman_transfer(
                    src, prefix, target_algo=_algo(tgt), target_higher_is_better=hib
                )
                scored[sid] = {
                    "importance": combine_score(feats, warm_w),
                    "prefix_rho": pre,
                    "feats": feats,
                }
            imp_ranked = topk_ids([(sid, scored[sid]["importance"]) for sid in scored], CASCADE_POOL)
            casc = []
            for sid in imp_ranked:
                pre = scored[sid]["prefix_rho"]
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["importance"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["importance"]) for sid in imp_ranked]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            top_id = casc[0][0]
            src = src_by_id[top_id]
            tgt_signs = role_signs_from_trials(prefix, higher_is_better=hib)
            src_signs = role_signs_from_experience(src)
            flipped = {r for r in src_signs if r in tgt_signs and src_signs[r] != tgt_signs[r]}
            n_queries += 1
            n_flipped_roles += len(flipped)
            held = spearman_transfer(
                src, trials[PREFIX_N:], target_algo=_algo(tgt), target_higher_is_better=hib
            )
            if held >= 0.3:
                n_high_rho += 1
                if "learning_rate" in flipped:
                    n_high_rho_lr_flip += 1
            names = {}
            for fac in ((tgt.get("problem") or {}).get("space") or {}).get("factors") or []:
                role = fac.get("role")
                if role and role not in names:
                    names[role] = str(fac.get("name") or "")
            hit = RetrievalHit(
                experience=src,
                score=casc[0][1],
                features=scored[top_id]["feats"],
                spearman=scored[top_id]["prefix_rho"],
            )
            format_retrieved_block([hit], names, target_role_signs=tgt_signs)

    stats = label_b_stats()
    payload = {
        "n_tune_queries": n_queries,
        "n_flipped_role_pairs": n_flipped_roles,
        "n_high_rho_cascade": n_high_rho,
        "n_high_rho_lr_flip": n_high_rho_lr_flip,
        "label_b_stats": stats,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "label_b_gate_counts.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Count Label-B factor-line suppressions on cascade top-1 hits (TUNE)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    RetrievalHit,
    cold_features,
    combine_score,
    format_retrieved_block,
    label_b_stats,
    reset_label_b_stats,
    role_signs_from_experience,
    role_signs_from_trials,
    spearman_transfer,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, topk_ids

PREFIX_N = 8
CASCADE_POOL = 12


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    reset_label_b_stats()

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < PREFIX_N + 3:
            continue
        if not is_tune(_ds(exp), split):
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    n_queries = 0
    n_flipped_roles = 0
    n_high_rho_lr_flip = 0
    n_high_rho = 0
    for members in by_group.values():
        if len(members) < 3:
            continue
        for tgt in members:
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:PREFIX_N]
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            scored = {}
            src_by_id = {}
            for src in cands:
                sid = str(src.get("id"))
                src_by_id[sid] = src
                feats = cold_features(q, src)
                pre = spearman_transfer(
                    src, prefix, target_algo=_algo(tgt), target_higher_is_better=hib
                )
                scored[sid] = {
                    "importance": combine_score(feats, warm_w),
                    "prefix_rho": pre,
                    "feats": feats,
                }
            imp_ranked = topk_ids([(sid, scored[sid]["importance"]) for sid in scored], CASCADE_POOL)
            casc = []
            for sid in imp_ranked:
                pre = scored[sid]["prefix_rho"]
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["importance"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["importance"]) for sid in imp_ranked]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            top_id = casc[0][0]
            src = src_by_id[top_id]
            tgt_signs = role_signs_from_trials(prefix, higher_is_better=hib)
            src_signs = role_signs_from_experience(src)
            flipped = {r for r in src_signs if r in tgt_signs and src_signs[r] != tgt_signs[r]}
            n_queries += 1
            n_flipped_roles += len(flipped)
            held = spearman_transfer(
                src, trials[PREFIX_N:], target_algo=_algo(tgt), target_higher_is_better=hib
            )
            if held >= 0.3:
                n_high_rho += 1
                if "learning_rate" in flipped:
                    n_high_rho_lr_flip += 1
            names = {}
            for fac in ((tgt.get("problem") or {}).get("space") or {}).get("factors") or []:
                role = fac.get("role")
                if role and role not in names:
                    names[role] = str(fac.get("name") or "")
            hit = RetrievalHit(
                experience=src,
                score=casc[0][1],
                features=scored[top_id]["feats"],
                spearman=scored[top_id]["prefix_rho"],
            )
            format_retrieved_block([hit], names, target_role_signs=tgt_signs)

    stats = label_b_stats()
    payload = {
        "n_tune_queries": n_queries,
        "n_flipped_role_pairs": n_flipped_roles,
        "n_high_rho_cascade": n_high_rho,
        "n_high_rho_lr_flip": n_high_rho_lr_flip,
        "label_b_stats": stats,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "label_b_gate_counts.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Count Label-B factor-line suppressions on cascade top-1 hits (TUNE)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    RetrievalHit,
    cold_features,
    combine_score,
    format_retrieved_block,
    label_b_stats,
    reset_label_b_stats,
    role_signs_from_experience,
    role_signs_from_trials,
    spearman_transfer,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, topk_ids

PREFIX_N = 8
CASCADE_POOL = 12


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    reset_label_b_stats()

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < PREFIX_N + 3:
            continue
        if not is_tune(_ds(exp), split):
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    n_queries = 0
    n_flipped_roles = 0
    n_high_rho_lr_flip = 0
    n_high_rho = 0
    for members in by_group.values():
        if len(members) < 3:
            continue
        for tgt in members:
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:PREFIX_N]
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            scored = {}
            src_by_id = {}
            for src in cands:
                sid = str(src.get("id"))
                src_by_id[sid] = src
                feats = cold_features(q, src)
                pre = spearman_transfer(
                    src, prefix, target_algo=_algo(tgt), target_higher_is_better=hib
                )
                scored[sid] = {
                    "importance": combine_score(feats, warm_w),
                    "prefix_rho": pre,
                    "feats": feats,
                }
            imp_ranked = topk_ids([(sid, scored[sid]["importance"]) for sid in scored], CASCADE_POOL)
            casc = []
            for sid in imp_ranked:
                pre = scored[sid]["prefix_rho"]
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["importance"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["importance"]) for sid in imp_ranked]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            top_id = casc[0][0]
            src = src_by_id[top_id]
            tgt_signs = role_signs_from_trials(prefix, higher_is_better=hib)
            src_signs = role_signs_from_experience(src)
            flipped = {r for r in src_signs if r in tgt_signs and src_signs[r] != tgt_signs[r]}
            n_queries += 1
            n_flipped_roles += len(flipped)
            held = spearman_transfer(
                src, trials[PREFIX_N:], target_algo=_algo(tgt), target_higher_is_better=hib
            )
            if held >= 0.3:
                n_high_rho += 1
                if "learning_rate" in flipped:
                    n_high_rho_lr_flip += 1
            names = {}
            for fac in ((tgt.get("problem") or {}).get("space") or {}).get("factors") or []:
                role = fac.get("role")
                if role and role not in names:
                    names[role] = str(fac.get("name") or "")
            hit = RetrievalHit(
                experience=src,
                score=casc[0][1],
                features=scored[top_id]["feats"],
                spearman=scored[top_id]["prefix_rho"],
            )
            format_retrieved_block([hit], names, target_role_signs=tgt_signs)

    stats = label_b_stats()
    payload = {
        "n_tune_queries": n_queries,
        "n_flipped_role_pairs": n_flipped_roles,
        "n_high_rho_cascade": n_high_rho,
        "n_high_rho_lr_flip": n_high_rho_lr_flip,
        "label_b_stats": stats,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "label_b_gate_counts.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Count Label-B factor-line suppressions on cascade top-1 hits (TUNE)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dataset_split import is_tune, load_split
from mindware.components.experience.retrieve import (
    COLD_WEIGHTS,
    DEFAULT_WEIGHTS,
    ExperienceBank,
    RetrievalHit,
    cold_features,
    combine_score,
    format_retrieved_block,
    label_b_stats,
    reset_label_b_stats,
    role_signs_from_experience,
    role_signs_from_trials,
    spearman_transfer,
)

from eval_lodo_similarity import _algo, _ds, _task, query_from_prefix, topk_ids

PREFIX_N = 8
CASCADE_POOL = 12


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, default=ROOT / "history_data" / "experience_bank_v2.jsonl")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "benchmark_data" / "real_hpo_v2_mltest20_algo40" / "analysis" / "lodo_similarity",
    )
    args = p.parse_args(argv)

    bank = ExperienceBank.load(args.bank)
    split = load_split()
    warm_w = dict(bank.weights) if bank.weights else dict(DEFAULT_WEIGHTS)
    reset_label_b_stats()

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for exp in bank.experiences:
        trials = (exp.get("retrieval") or {}).get("trials") or []
        if len(trials) < PREFIX_N + 3:
            continue
        if not is_tune(_ds(exp), split):
            continue
        by_group[(_task(exp), _algo(exp))].append(exp)

    n_queries = 0
    n_flipped_roles = 0
    n_high_rho_lr_flip = 0
    n_high_rho = 0
    for members in by_group.values():
        if len(members) < 3:
            continue
        for tgt in members:
            trials = (tgt.get("retrieval") or {}).get("trials") or []
            prefix = trials[:PREFIX_N]
            cands = [s for s in members if _ds(s) != _ds(tgt)]
            if len(cands) < 3:
                continue
            q = query_from_prefix(tgt, prefix)
            hib = bool((tgt.get("problem") or {}).get("higher_is_better", True))
            scored = {}
            src_by_id = {}
            for src in cands:
                sid = str(src.get("id"))
                src_by_id[sid] = src
                feats = cold_features(q, src)
                pre = spearman_transfer(
                    src, prefix, target_algo=_algo(tgt), target_higher_is_better=hib
                )
                scored[sid] = {
                    "importance": combine_score(feats, warm_w),
                    "prefix_rho": pre,
                    "feats": feats,
                }
            imp_ranked = topk_ids([(sid, scored[sid]["importance"]) for sid in scored], CASCADE_POOL)
            casc = []
            for sid in imp_ranked:
                pre = scored[sid]["prefix_rho"]
                if pre < 0.0:
                    continue
                casc.append((sid, 0.6 * scored[sid]["importance"] + 0.4 * max(0.0, pre)))
            if not casc:
                casc = [(sid, scored[sid]["importance"]) for sid in imp_ranked]
            casc.sort(key=lambda kv: kv[1], reverse=True)
            top_id = casc[0][0]
            src = src_by_id[top_id]
            tgt_signs = role_signs_from_trials(prefix, higher_is_better=hib)
            src_signs = role_signs_from_experience(src)
            flipped = {r for r in src_signs if r in tgt_signs and src_signs[r] != tgt_signs[r]}
            n_queries += 1
            n_flipped_roles += len(flipped)
            held = spearman_transfer(
                src, trials[PREFIX_N:], target_algo=_algo(tgt), target_higher_is_better=hib
            )
            if held >= 0.3:
                n_high_rho += 1
                if "learning_rate" in flipped:
                    n_high_rho_lr_flip += 1
            names = {}
            for fac in ((tgt.get("problem") or {}).get("space") or {}).get("factors") or []:
                role = fac.get("role")
                if role and role not in names:
                    names[role] = str(fac.get("name") or "")
            hit = RetrievalHit(
                experience=src,
                score=casc[0][1],
                features=scored[top_id]["feats"],
                spearman=scored[top_id]["prefix_rho"],
            )
            format_retrieved_block([hit], names, target_role_signs=tgt_signs)

    stats = label_b_stats()
    payload = {
        "n_tune_queries": n_queries,
        "n_flipped_role_pairs": n_flipped_roles,
        "n_high_rho_cascade": n_high_rho,
        "n_high_rho_lr_flip": n_high_rho_lr_flip,
        "label_b_stats": stats,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "label_b_gate_counts.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
