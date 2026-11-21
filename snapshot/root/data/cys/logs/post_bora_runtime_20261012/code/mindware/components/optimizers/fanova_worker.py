#!/usr/bin/env python3
"""Standalone fANOVA worker for the dedicated venv (numpy 1.x + pyrfr + fanova).

Reads one JSON object per stdin line, writes one JSON object per stdout line.
Does not import mindware so it can run under /root/venvs/fanova.
"""

from __future__ import annotations

import json
import sys
import traceback


def _patch_numpy() -> None:
    import numpy as np

    aliases = {
        "float": float,
        "int": int,
        "bool": bool,
        "object": object,
        "str": str,
        "long": int,
        "unicode": str,
        "complex": complex,
    }
    for name, alias in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, alias)


def _add_hp(cs, hp) -> None:
    try:
        cs.add_hyperparameter(hp)
    except Exception:
        cs.add(hp)


def _build_config_space_from_specs(specs, names, kinds, X):
    from ConfigSpace import (
        CategoricalHyperparameter,
        ConfigurationSpace,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    import numpy as np

    by_name = {str(s.get("name")): s for s in (specs or []) if s and s.get("name")}
    cs = ConfigurationSpace()
    n_cols = X.shape[1]
    for i in range(n_cols):
        name = str(names[i])
        spec = by_name.get(name) or {}
        kind = str(spec.get("kind") or kinds[i] or "numeric").lower()
        col = X[:, i]
        if kind in ("categorical", "cat", "ordinal"):
            choices = spec.get("choices")
            if not choices:
                n_choices = int(np.nanmax(col)) + 1 if col.size else 1
                n_choices = max(n_choices, int(len(set(col.tolist()))) if col.size else 1)
                n_choices = max(2, n_choices)
                choices = [str(j) for j in range(n_choices)]
            else:
                choices = [str(c) for c in choices]
                if len(choices) < 2:
                    choices = list(choices) + [choices[0] + "__pad"]
            _add_hp(cs, CategoricalHyperparameter(name, choices))
            continue
        log = bool(spec.get("log", False))
        lo_s, hi_s = spec.get("lower"), spec.get("upper")
        mn = float(np.nanmin(col)) if col.size else 0.0
        mx = float(np.nanmax(col)) if col.size else 1.0
        pad = max(1e-8, 1e-6 * (abs(mn) + abs(mx) + 1.0))
        if mx - mn < pad:
            pad = max(pad, 1.0)
        lo = float(lo_s) if lo_s is not None else mn - pad
        hi = float(hi_s) if hi_s is not None else mx + pad
        if hi <= lo:
            hi = lo + pad
        if kind in ("integer", "int"):
            ilo, ihi = int(np.floor(lo)), int(np.ceil(hi))
            if ihi <= ilo:
                ihi = ilo + 1
            try:
                _add_hp(cs, UniformIntegerHyperparameter(name, ilo, ihi, log=log))
            except Exception:
                _add_hp(cs, UniformIntegerHyperparameter(name, ilo, ihi))
            continue
        if log and lo <= 0.0:
            log = False
        try:
            _add_hp(cs, UniformFloatHyperparameter(name, lo, hi, log=log))
        except Exception:
            _add_hp(cs, UniformFloatHyperparameter(name, lo, hi))
    return cs


def _build_config_space(names, kinds, X):
    return _build_config_space_from_specs([], names, kinds, X)


def _importance(payload: dict) -> dict:
    _patch_numpy()
    import numpy as np
    from fanova import fANOVA

    names = list(payload.get("names") or [])
    kinds = list(payload.get("kinds") or [])
    X = np.asarray(payload.get("X") or [], dtype=np.float64)
    Y = np.asarray(payload.get("Y") or [], dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[1] != len(names):
        raise ValueError(f"bad shapes X={X.shape} Y={Y.shape} names={len(names)}")
    if not kinds or len(kinds) != len(names):
        kinds = ["numeric"] * len(names)
    n_trees = int(payload.get("n_trees") or 32)
    seed = int(payload.get("seed") or 0)
    specs = list(payload.get("space") or [])
    kind = str(payload.get("importance_kind") or "individual").strip().lower()
    want_total = kind in ("total", "interaction", "st", "sobol_total")
    key = "total importance" if want_total else "individual importance"
    cs = _build_config_space_from_specs(specs, names, kinds, X)
    fan = fANOVA(X, Y, config_space=cs, n_trees=n_trees, seed=seed)
    individual, total = {}, {}
    for i, name in enumerate(names):
        try:
            raw = fan.quantify_importance((i,))
            rec = raw.get((i,)) or next(iter(raw.values()))
            ind = float(rec.get("individual importance", 0.0))
            tot = float(rec.get("total importance", ind))
        except Exception:
            ind, tot = 0.0, 0.0
        if ind != ind or ind < 0.0:
            ind = 0.0
        if tot != tot or tot < 0.0:
            tot = 0.0
        individual[str(name)] = min(1.0, ind)
        total[str(name)] = min(1.0, tot)
    picked = total if want_total else individual
    pairs_out = []
    pair_topk = int(payload.get("pair_topk") or 0)
    if pair_topk >= 2 and len(names) >= 2:
        ranked = sorted(range(len(names)), key=lambda i: -float(total.get(str(names[i]), 0.0)))
        top = ranked[: max(2, min(pair_topk, len(names)))]
        for a_i in range(len(top)):
            for b_i in range(a_i + 1, len(top)):
                i, j = int(top[a_i]), int(top[b_i])
                try:
                    raw = fan.quantify_importance((i, j))
                    rec = raw.get((i, j)) or raw.get((j, i)) or next(iter(raw.values()))
                    ind = float(rec.get("individual importance", 0.0) or 0.0)
                    tot = float(rec.get("total importance", ind) or ind)
                except Exception:
                    ind, tot = 0.0, 0.0
                if ind != ind or ind < 0.0:
                    ind = 0.0
                if tot != tot or tot < 0.0:
                    tot = 0.0
                pairs_out.append(
                    {
                        "a": str(names[i]),
                        "b": str(names[j]),
                        "importance_1d": min(1.0, ind),
                        "importance_total": min(1.0, tot),
                    }
                )
    return {
        "ok": True,
        "importance": picked,
        "importance_individual": individual,
        "importance_total": total,
        "pairs": pairs_out,
        "engine": "fanova",
        "importance_kind": "total" if want_total else "individual",
    }


def main() -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": f"json: {exc}"}) + "\n")
            sys.stdout.flush()
            continue
        cmd = str(payload.get("cmd") or "importance").strip().lower()
        if cmd in ("ping", "hello"):
            sys.stdout.write(json.dumps({"ok": True, "engine": "fanova"}) + "\n")
            sys.stdout.flush()
            continue
        if cmd in ("quit", "exit"):
            sys.stdout.write(json.dumps({"ok": True, "bye": True}) + "\n")
            sys.stdout.flush()
            return 0
        try:
            result = _importance(payload)
        except Exception as exc:
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=6),
            }
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
