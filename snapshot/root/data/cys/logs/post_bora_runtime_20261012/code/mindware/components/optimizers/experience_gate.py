"""When to insert this-run prefer/avoid (replaces a hard trial-50 clock)."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SKIP_HP_NAMES = frozenset({"algorithm"})
FANOVA_N_TREES = 32
FANOVA_MIN_N = 24
FANOVA_THRESH = 0.10
FANOVA_KEEP_TOPK = 4
FANOVA_KEEP_CUM = 0.60
FANOVA_KEEP_MIN = 2
FANOVA_KEEP_FLOOR = 0.03
_FANOVA_WORKER: Optional["FanovaWorkerClient"] = None
_FANOVA_LOCK = threading.Lock()
_LAST_FANOVA_META: Dict[str, Any] = {}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def region_signature(block: str) -> str:
    lines = []
    for line in (block or "").splitlines():
        s = line.strip()
        if s.startswith("- ") and ("prefer" in s or "avoid" in s):
            lines.append(s)
    return "\n".join(lines[:12])


def block_has_regions(block: str) -> bool:
    text = block or ""
    if "no clear separation yet" in text and "- avoid: none" in text:
        return False
    return ("prefer ~" in text) or ("prefer " in text and "over" in text)


_PREFER_JSON = re.compile(
    r"prefer\s+(?:~|hint toward\s+~|strongly prefer\s+~)?\s*"
    r'\{["\']low["\']:\s*(?P<lo>[-0-9.eE]+),\s*["\']high["\']:\s*(?P<hi>[-0-9.eE]+)'
)
_PREFER_CAT = re.compile(r"prefer\s+(?P<val>.+?)\s+over\s+")


def parse_prefer_constraints(block: str) -> List[Dict[str, Any]]:
    """Parse prefer intervals / categorical values from an experience_self block."""
    out: List[Dict[str, Any]] = []
    for line in (block or "").splitlines():
        s = line.strip()
        if not s.startswith("- ") or "prefer" not in s:
            continue
        rest = s[2:]
        split_at = rest.find(": ")
        if split_at < 0:
            continue
        name = rest[:split_at].strip()
        body = rest[split_at + 2 :]
        if name in SKIP_HP_NAMES or name in ("factors", "avoid", "mode", "dynamics", "baseline_kind"):
            continue
        m = _PREFER_JSON.search(body)
        if m:
            lo, hi = float(m.group("lo")), float(m.group("hi"))
            if hi < lo:
                lo, hi = hi, lo
            out.append({"name": name, "kind": "numeric", "low": lo, "high": hi})
            continue
        m = _PREFER_CAT.search(body)
        if m:
            raw = m.group("val").strip()
            try:
                val = ast.literal_eval(raw)
            except Exception:
                val = raw.strip("'\"")
            out.append({"name": name, "kind": "categorical", "value": val})
    return out


def config_prefer_hit_ratio(cfg: Optional[Dict[str, Any]], constraints: Sequence[Dict[str, Any]]) -> float:
    """Fraction of prefer constraints that the config satisfies. 0 if none apply."""
    if not cfg or not constraints:
        return 0.0
    hits = 0
    checks = 0
    for spec in constraints:
        name = spec["name"]
        if name not in cfg:
            continue
        checks += 1
        val = cfg[name]
        if spec["kind"] == "numeric":
            if _is_number(val) and float(spec["low"]) - 1e-12 <= float(val) <= float(spec["high"]) + 1e-12:
                hits += 1
        else:
            if val == spec.get("value") or str(val) == str(spec.get("value")):
                hits += 1
    if checks <= 0:
        return 0.0
    return hits / float(checks)


def stagnation_count(successes: Sequence[Dict[str, Any]]) -> int:
    if not successes:
        return 0
    best = float(successes[0]["score"])
    last = 0
    for i, item in enumerate(successes):
        s = float(item["score"])
        if s > best + 1e-12:
            best = s
            last = i
    return len(successes) - 1 - last


def contrast_ok(successes: Sequence[Dict[str, Any]], min_rel: float = 0.05) -> bool:
    scores = [float(x["score"]) for x in successes]
    if len(scores) < 8:
        return False
    lo, hi = min(scores), max(scores)
    span = abs(hi - lo)
    if span <= 1e-12:
        return False
    ordered = sorted(scores)
    k = max(2, int(math.ceil(0.2 * len(ordered))))
    top_m = sum(ordered[-k:]) / k
    bot_m = sum(ordered[:k]) / k
    return (top_m - bot_m) >= min_rel * span


def default_fanova_python() -> str:
    env = str(os.environ.get("FANOVA_PYTHON") or "").strip()
    if env:
        return env
    candidates = [
        Path("/root/venvs/fanova/bin/python"),
        Path.home() / "venvs/fanova/bin/python",
        Path(__file__).resolve().parents[3] / ".venv-fanova" / "bin" / "python",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return ""


class FanovaWorkerClient:
    """Persistent subprocess talking to the dedicated fanova venv."""

    def __init__(self, python_bin: str, timeout: float = 45.0) -> None:
        self.python_bin = python_bin
        self.timeout = float(timeout)
        self.proc: Optional[subprocess.Popen] = None
        worker = Path(__file__).with_name("fanova_worker.py")
        self.worker = str(worker)
        self._start()

    def _start(self) -> None:
        self.close()
        self.proc = subprocess.Popen(
            [self.python_bin, self.worker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        ping = self._call({"cmd": "ping"}, timeout=8.0)
        if not ping.get("ok"):
            raise RuntimeError(f"fanova worker ping failed: {ping.get('error')}")

    def _call(self, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        blob = (json.dumps(payload) + "\n").encode("utf-8")
        self.proc.stdin.write(blob)
        self.proc.stdin.flush()
        wait = self.timeout if timeout is None else float(timeout)
        import select

        ready, _, _ = select.select([self.proc.stdout], [], [], wait)
        if not ready:
            raise TimeoutError(f"fanova worker timed out after {wait}s")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("fanova worker died")
        return json.loads(line.decode("utf-8"))

    def importance_detail(
        self,
        names: Sequence[str],
        kinds: Sequence[str],
        X: Sequence[Sequence[float]],
        Y: Sequence[float],
        *,
        n_trees: int = FANOVA_N_TREES,
        seed: int = 0,
        space: Optional[Sequence[Dict[str, Any]]] = None,
        importance_kind: str = "individual",
        pair_topk: int = 0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        with _FANOVA_LOCK:
            rec = self._call(
                {
                    "cmd": "importance",
                    "names": list(names),
                    "kinds": list(kinds),
                    "X": [list(row) for row in X],
                    "Y": [float(y) for y in Y],
                    "n_trees": int(n_trees),
                    "seed": int(seed),
                    "space": list(space or []),
                    "importance_kind": str(importance_kind or "individual"),
                    "pair_topk": int(pair_topk or 0),
                },
                timeout=timeout,
            )
        if not rec.get("ok"):
            raise RuntimeError(str(rec.get("error") or "fanova worker failed"))
        return rec

    def importance(
        self,
        names: Sequence[str],
        kinds: Sequence[str],
        X: Sequence[Sequence[float]],
        Y: Sequence[float],
        *,
        n_trees: int = FANOVA_N_TREES,
        seed: int = 0,
        space: Optional[Sequence[Dict[str, Any]]] = None,
        importance_kind: str = "individual",
    ) -> Dict[str, float]:
        rec = self.importance_detail(
            names,
            kinds,
            X,
            Y,
            n_trees=n_trees,
            seed=seed,
            space=space,
            importance_kind=importance_kind,
        )
        raw = rec.get("importance") or {}
        return {str(k): float(v) for k, v in raw.items()}

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(b'{"cmd":"quit"}\n')
                self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass
        self.proc = None


def get_fanova_worker() -> FanovaWorkerClient:
    global _FANOVA_WORKER
    if _FANOVA_WORKER is not None:
        return _FANOVA_WORKER
    python_bin = default_fanova_python()
    if not python_bin:
        raise RuntimeError("FANOVA_PYTHON not set and no fanova venv found")
    _FANOVA_WORKER = FanovaWorkerClient(python_bin)
    import atexit

    atexit.register(_FANOVA_WORKER.close)
    return _FANOVA_WORKER


def serialize_config_space(config_space: Any) -> List[Dict[str, Any]]:
    """JSON-safe HP specs from a ConfigSpace (true bounds / types / log)."""
    if config_space is None:
        return []
    try:
        hps = list(config_space.get_hyperparameters())
    except Exception:
        try:
            hps = list(config_space.values())
        except Exception:
            return []
    specs: List[Dict[str, Any]] = []
    for hp in hps:
        name = str(getattr(hp, "name", "") or "")
        if not name or name in SKIP_HP_NAMES:
            continue
        spec: Dict[str, Any] = {"name": name, "kind": "float", "log": False}
        choices = getattr(hp, "choices", None)
        seq = getattr(hp, "sequence", None)
        if choices is not None:
            spec["kind"] = "categorical"
            spec["choices"] = [str(c) for c in list(choices)]
        elif seq is not None:
            spec["kind"] = "categorical"
            spec["choices"] = [str(c) for c in list(seq)]
        else:
            lo = getattr(hp, "lower", None)
            hi = getattr(hp, "upper", None)
            if lo is not None:
                spec["lower"] = float(lo)
            if hi is not None:
                spec["upper"] = float(hi)
            spec["log"] = bool(getattr(hp, "log", False))
            cls = type(hp).__name__.lower()
            if "int" in cls:
                spec["kind"] = "integer"
        specs.append(spec)
    return specs


def _encode_hp_matrix_observed(
    successes: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[List[float]], List[float]]:
    """Encode this-run configs from the sampled cloud (no true ConfigSpace)."""
    names = sorted(
        {
            k
            for item in successes
            for k in (item.get("config") or {})
            if k not in SKIP_HP_NAMES
        }
    )
    Y = [float(item["score"]) for item in successes]
    kinds: List[str] = []
    columns: List[List[float]] = []
    kept_names: List[str] = []
    for name in names:
        vals = [(item.get("config") or {}).get(name) for item in successes]
        if any(v is None for v in vals):
            continue
        if all(_is_number(v) for v in vals):
            nums = [float(v) for v in vals]
            if max(nums) - min(nums) <= 1e-15:
                continue
            kept_names.append(name)
            kinds.append("numeric")
            columns.append(nums)
            continue
        keys = [json.dumps(v, sort_keys=True, default=str) for v in vals]
        uniq = sorted(set(keys))
        if len(uniq) < 2:
            continue
        index = {k: float(i) for i, k in enumerate(uniq)}
        kept_names.append(name)
        kinds.append("categorical")
        columns.append([index[k] for k in keys])
    if not columns:
        return [], [], [], Y
    X = [list(row) for row in zip(*columns)]
    return kept_names, kinds, X, Y


def _encode_hp_matrix_from_space(
    successes: Sequence[Dict[str, Any]],
    config_space: Any,
) -> Tuple[List[str], List[str], List[List[float]], List[float], List[Dict[str, Any]]]:
    specs = serialize_config_space(config_space)
    Y = [float(item["score"]) for item in successes]
    if not specs or not successes:
        names, kinds, X, Y2 = _encode_hp_matrix_observed(successes)
        return names, kinds, X, Y2, []
    kept_names: List[str] = []
    kinds: List[str] = []
    columns: List[List[float]] = []
    kept_specs: List[Dict[str, Any]] = []
    for spec in specs:
        name = str(spec["name"])
        kind = str(spec.get("kind") or "float")
        vals = [(item.get("config") or {}).get(name) for item in successes]
        default = None
        try:
            hp = config_space.get_hyperparameter(name)
            default = getattr(hp, "default_value", None)
        except Exception:
            default = None
        filled = [default if v is None else v for v in vals]
        if any(v is None for v in filled):
            continue
        if kind == "categorical":
            choices = [str(c) for c in (spec.get("choices") or [])]
            if len(choices) < 2:
                continue
            index = {c: float(i) for i, c in enumerate(choices)}
            col: List[float] = []
            ok = True
            for v in filled:
                key = str(v)
                if key not in index:
                    # unseen label: skip this HP rather than corrupt the CS measure
                    ok = False
                    break
                col.append(index[key])
            if not ok or len(set(col)) < 2:
                continue
            kept_names.append(name)
            kinds.append("categorical")
            columns.append(col)
            kept_specs.append(spec)
            continue
        nums: List[float] = []
        ok = True
        lo = spec.get("lower")
        hi = spec.get("upper")
        for v in filled:
            if not _is_number(v):
                ok = False
                break
            x = float(v)
            if lo is not None:
                x = max(float(lo), x)
            if hi is not None:
                x = min(float(hi), x)
            nums.append(x)
        if not ok or max(nums) - min(nums) <= 1e-15:
            continue
        kept_names.append(name)
        kinds.append("integer" if kind == "integer" else "numeric")
        columns.append(nums)
        kept_specs.append(spec)
    if not columns:
        names, k2, X, Y2 = _encode_hp_matrix_observed(successes)
        return names, k2, X, Y2, []
    X = [list(row) for row in zip(*columns)]
    return kept_names, kinds, X, Y, kept_specs


def encode_hp_matrix(
    successes: Sequence[Dict[str, Any]],
    config_space: Any = None,
) -> Tuple[List[str], List[str], List[List[float]], List[float]]:
    """Encode this-run configs into a numeric matrix for RF / fANOVA."""
    if config_space is not None:
        names, kinds, X, Y, _specs = _encode_hp_matrix_from_space(successes, config_space)
        return names, kinds, X, Y
    return _encode_hp_matrix_observed(successes)


def calibrated_fanova_keep(imps: Dict[str, float]) -> set:
    """Keep a handful of drivers; if fewer than 2 clear HPs, keep nothing (unfiltered block)."""
    ranked = sorted(
        ((k, float(v)) for k, v in imps.items() if float(v) >= FANOVA_KEEP_FLOOR),
        key=lambda kv: -kv[1],
    )
    if len(ranked) < FANOVA_KEEP_MIN:
        return set()
    keep: List[str] = []
    acc = 0.0
    for name, val in ranked:
        keep.append(name)
        acc += val
        if len(keep) >= FANOVA_KEEP_MIN and (acc >= FANOVA_KEEP_CUM or len(keep) >= FANOVA_KEEP_TOPK):
            break
    return set(keep)


def hp_importance(successes: Sequence[Dict[str, Any]], name: str, n_bins: int = 4) -> float:
    pairs = [
        (float(item["config"][name]), float(item["score"]))
        for item in successes
        if name in (item.get("config") or {}) and _is_number(item["config"][name])
    ]
    if len(pairs) < 6:
        return 0.0
    scores = [s for _, s in pairs]
    var_y = _var(scores)
    if var_y <= 1e-18:
        return 0.0
    vals = [v for v, _ in pairs]
    lo, hi = min(vals), max(vals)
    if hi <= lo + 1e-15:
        return 0.0
    buckets: List[List[float]] = [[] for _ in range(n_bins)]
    for v, s in pairs:
        idx = min(n_bins - 1, int((v - lo) / (hi - lo) * n_bins))
        buckets[idx].append(s)
    means = [sum(b) / len(b) for b in buckets if len(b) >= 2]
    if len(means) < 2:
        return 0.0
    return min(1.0, _var(means) / var_y)


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def hp_importances_proxy(successes: Sequence[Dict[str, Any]], n_bins: int = 4) -> Dict[str, float]:
    names = {k for item in successes for k in (item.get("config") or {}) if k not in SKIP_HP_NAMES}
    return {name: hp_importance(successes, name, n_bins=n_bins) for name in names}


def hp_importances_rf(
    successes: Sequence[Dict[str, Any]],
    *,
    n_estimators: int = 32,
    seed: int = 0,
    kind: str = "impurity",
) -> Dict[str, float]:
    names, _kinds, X, Y = encode_hp_matrix(successes)
    if len(names) < 1 or len(X) < 8:
        return {}
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return {}
    import numpy as np

    y = np.asarray(Y, dtype=float)
    arr = np.asarray(X, dtype=float)
    if float(np.var(y)) <= 1e-18:
        return {n: 0.0 for n in names}
    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(seed),
        min_samples_leaf=2,
        max_depth=6,
    )
    rf.fit(arr, y)
    if str(kind).lower() in ("pd", "pdp", "anova", "main"):
        try:
            pred_var = float(np.var(rf.predict(arr)))
            if pred_var <= 1e-18:
                pred_var = float(np.var(y)) or 1.0
            out: Dict[str, float] = {}
            from sklearn.inspection import partial_dependence

            for i, name in enumerate(names):
                pd_res = partial_dependence(rf, arr, [i], kind="average", grid_resolution=8)
                avg = np.asarray(pd_res["average"]).reshape(-1)
                out[name] = min(1.0, float(np.var(avg)) / pred_var)
            return out
        except Exception:
            kind = "impurity"
    imps = [float(v) for v in rf.feature_importances_]
    total = sum(imps) or 1.0
    return {name: min(1.0, imps[i] / total) for i, name in enumerate(names)}


def hp_importances_full(
    successes: Sequence[Dict[str, Any]],
    *,
    n_trees: int = FANOVA_N_TREES,
    seed: int = 0,
    fallback: str = "rf",
    config_space: Any = None,
    importance_kind: str = "individual",
) -> Tuple[Dict[str, float], str]:
    space: List[Dict[str, Any]] = []
    if config_space is not None:
        names, kinds, X, Y, space = _encode_hp_matrix_from_space(successes, config_space)
    else:
        names, kinds, X, Y = encode_hp_matrix(successes)
    if len(names) < 1 or len(X) < FANOVA_MIN_N:
        proxy = hp_importances_proxy(successes)
        return proxy, "proxy_too_few"
    try:
        worker = get_fanova_worker()
        imps = worker.importance(
            names,
            kinds,
            X,
            Y,
            n_trees=n_trees,
            seed=seed,
            space=space,
            importance_kind=importance_kind,
        )
        if imps:
            return imps, "fanova"
    except Exception as exc:
        _LAST_FANOVA_META["fanova_error"] = f"{type(exc).__name__}: {exc}"
    if fallback in ("rf", "sklearn"):
        rf = hp_importances_rf(successes, seed=seed, kind="pd")
        if rf:
            return rf, "rf_pd_fallback"
        rf = hp_importances_rf(successes, seed=seed, kind="impurity")
        if rf:
            return rf, "rf_fallback"
    return hp_importances_proxy(successes), "proxy_fallback"


def hp_importances_both(
    successes: Sequence[Dict[str, Any]],
    *,
    n_trees: int = FANOVA_N_TREES,
    seed: int = 0,
    config_space: Any = None,
    pair_topk: int = 0,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]], str]:
    """One worker call returning 1D, total, optional pairs."""
    space: List[Dict[str, Any]] = []
    if config_space is not None:
        names, kinds, X, Y, space = _encode_hp_matrix_from_space(successes, config_space)
        used = "fanova_full_cs"
    else:
        names, kinds, X, Y = encode_hp_matrix(successes)
        used = "fanova_full"
    if len(names) < 1 or len(X) < FANOVA_MIN_N:
        proxy = hp_importances_proxy(successes)
        return proxy, dict(proxy), [], "proxy_too_few"
    try:
        worker = get_fanova_worker()
        rec = worker.importance_detail(
            names,
            kinds,
            X,
            Y,
            n_trees=n_trees,
            seed=seed,
            space=space,
            importance_kind="individual",
            pair_topk=pair_topk,
        )
        one = {str(k): float(v) for k, v in (rec.get("importance_individual") or {}).items()}
        tot = {str(k): float(v) for k, v in (rec.get("importance_total") or {}).items()}
        pairs = list(rec.get("pairs") or [])
        if one:
            return one, tot or dict(one), pairs, used if space else "fanova_full"
    except Exception as exc:
        _LAST_FANOVA_META["fanova_error"] = f"{type(exc).__name__}: {exc}"
    proxy = hp_importances_proxy(successes)
    return proxy, dict(proxy), [], "proxy_fallback"


def hp_importances(
    successes: Sequence[Dict[str, Any]],
    engine: str = "proxy",
    config_space: Any = None,
) -> Tuple[Dict[str, float], str]:
    kind = str(engine or "proxy").strip().lower()
    if kind in ("full", "fanova_full"):
        return hp_importances_full(successes)
    if kind in ("full_cs", "fanova_full_cs"):
        return hp_importances_full(
            successes, config_space=config_space, importance_kind="individual"
        )
    if kind in ("full_ix", "fanova_full_ix", "fanova_full_cs_ix"):
        return hp_importances_full(
            successes, config_space=config_space, importance_kind="total"
        )
    if kind in ("rf", "sklearn", "fanova_rf"):
        imps = hp_importances_rf(successes, kind="impurity")
        return imps, "rf"
    if kind in ("rf_pd", "pdp"):
        imps = hp_importances_rf(successes, kind="pd")
        return imps, "rf_pd"
    return hp_importances_proxy(successes), "proxy"


def bernstein_hw(n: int, var: float, bound: float = 1.0, delta: float = 0.1) -> float:
    n = max(int(n), 1)
    logt = math.log(max(3.0 / max(delta, 1e-9), 3.0))
    return math.sqrt(max(0.0, 2.0 * var * logt / n)) + 3.0 * bound * logt / n


def hoeffding_hw(n: int, bound: float = 1.0, delta: float = 0.1) -> float:
    """Hoeffding half-width on a [0, bound] variable. Usable at T=80 bin sizes."""
    n = max(int(n), 1)
    return float(bound) * math.sqrt(math.log(2.0 / max(float(delta), 1e-9)) / (2.0 * n))


@dataclass
class ExperienceGate:
    """Stateful latch for inserting the this-run experience block."""

    mode: str = "clock"
    floor: int = 24
    cap: int = 50
    k_stable: int = 3
    fanova_thresh: float = FANOVA_THRESH
    latched: bool = False
    stable: int = 0
    prev_sig: str = ""
    turbo_L: float = 0.8
    succ: int = 0
    fail: int = 0
    last_best: Optional[float] = None
    last_meta: Dict[str, Any] = field(default_factory=dict)

    def decide(
        self,
        *,
        current_trial: int,
        successes: Sequence[Dict[str, Any]],
        block: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        mode = (self.mode or "clock").strip().lower()
        n = len(successes)
        trial = int(current_trial)
        has_regions = block_has_regions(block)
        sig = region_signature(block)
        stag = stagnation_count(successes)
        contrast = contrast_ok(successes)

        if n and self.last_best is not None:
            best = max(float(x["score"]) for x in successes)
            if best > self.last_best + 1e-3 * abs(self.last_best):
                self.succ += 1
                self.fail = 0
            else:
                self.fail += 1
                self.succ = 0
            if self.succ >= 3:
                self.turbo_L = min(1.6, 2.0 * self.turbo_L)
                self.succ = 0
            else:
                keys = set()
                for item in successes:
                    keys.update((item.get("config") or {}).keys())
                n_hp = len(keys)
                if self.fail >= max(4, n_hp or 4):
                    self.turbo_L = max(0.25, self.turbo_L / 2.0)
                    self.fail = 0
        if successes:
            self.last_best = max(float(x["score"]) for x in successes)

        if sig and sig == self.prev_sig:
            self.stable += 1
        elif sig:
            self.stable = 1
            self.prev_sig = sig
        else:
            self.stable = 0
            self.prev_sig = ""

        if n >= self.floor and has_regions and self.stable >= self.k_stable:
            self.latched = True
        if trial >= self.cap and has_regions:
            self.latched = True

        active = False
        if mode in ("clock", "t50", ""):
            active = trial >= self.cap
        elif mode in (
            "ready",
            "race",
            "fanova",
            "fanova_full",
            "fanova_full_cs",
            "fanova_full_ix",
            "fanova_rf",
            "turbo",
            "strength",
        ):
            active = bool(self.latched) and n >= self.floor
        elif mode in ("contrast",):
            active = n >= self.floor and contrast
        elif mode in ("stagnation", "stag"):
            active = n >= self.floor and stag >= max(5, int(0.15 * n))
        else:
            active = trial >= self.cap

        meta = {
            "experience_self_gate": mode,
            "experience_self_active": bool(active),
            "experience_gate_latched": bool(self.latched),
            "experience_gate_stable": int(self.stable),
            "experience_gate_n": int(n),
            "experience_gate_has_regions": bool(has_regions),
            "experience_gate_stagnation": int(stag),
            "experience_gate_contrast": bool(contrast),
            "experience_gate_turbo_L": float(self.turbo_L),
        }
        self.last_meta = meta
        return bool(active), meta


def apply_strength(block: str, n: int, floor: int, cap: int) -> str:
    if not block:
        return block
    span = max(cap - floor, 1)
    t = min(1.0, max(0.0, (n - floor) / float(span)))
    if t < 0.33:
        src, dst = "prefer", "hint toward"
    elif t < 0.66:
        src, dst = "prefer", "prefer"
    else:
        src, dst = "prefer", "strongly prefer"
    if src != dst:
        block = block.replace(f"{src} ~", f"{dst} ~").replace(f"{src} ", f"{dst} ")
    return block


def apply_turbo_width(block: str, length: float) -> str:
    """Shrink/expand numeric prefer/avoid envelopes toward their centers by L."""
    import json
    import re

    L = min(1.6, max(0.25, float(length)))

    def _scale(m):
        try:
            obj = json.loads(m.group(0))
            lo, hi = float(obj["low"]), float(obj["high"])
        except Exception:
            return m.group(0)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * L
        obj["low"] = mid - half
        obj["high"] = mid + half
        return json.dumps(obj)

    return re.sub(r'\{"low":\s*[-0-9.eE]+,\s*"high":\s*[-0-9.eE]+\}', _scale, block or "")


def filter_fanova(
    block: str,
    successes: Sequence[Dict[str, Any]],
    thresh: float,
    *,
    engine: str = "proxy",
    config_space: Any = None,
) -> str:
    if not block:
        return block
    prev_err = _LAST_FANOVA_META.get("fanova_error")
    imps, used = hp_importances(successes, engine=engine, config_space=config_space)
    kind = str(engine or "proxy").strip().lower()
    calibrated = kind in (
        "full_cs",
        "fanova_full_cs",
        "full_ix",
        "fanova_full_ix",
        "fanova_full_cs_ix",
    )
    keep = calibrated_fanova_keep(imps) if calibrated else {k for k, v in imps.items() if float(v) >= float(thresh)}
    err = _LAST_FANOVA_META.get("fanova_error") or prev_err
    _LAST_FANOVA_META.clear()
    _LAST_FANOVA_META.update(
        {
            "experience_fanova_engine": used,
            "experience_fanova_requested": str(engine),
            "experience_fanova_n_keep": int(len(keep)),
            "experience_fanova_n_hps": int(len(imps)),
            "experience_fanova_importance": {k: round(float(v), 4) for k, v in sorted(imps.items())},
            "experience_fanova_keep_policy": "calibrated" if calibrated else "thresh",
        }
    )
    if err:
        _LAST_FANOVA_META["fanova_error"] = err
    if not keep:
        return block
    out = []
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("- ") and ":" in s and ("prefer" in s or "avoid" in s or "over" in s):
            name = s[2:].split(":", 1)[0].strip()
            if name and name not in keep and name not in ("factors", "avoid", "mode", "dynamics", "baseline_kind"):
                continue
        out.append(line)
    return "\n".join(out)


XOR_EXPLORE_MODES = frozenset(
    {
        "fifty_fifty",
        "50",
        "x50",
        "stagnation",
        "stag",
        "xstag",
        "coverage",
        "cov",
        "xcov",
        "intensify",
        "int",
        "xint",
        "refine",
        "xrefine",
        "hap",
        "xhap",
        "dts",
        "xdts",
        "ucb",
        "xucb",
        "tpe",
        "xtpe",
        "tpe_soft",
        "xtpesoft",
        "tpe_llm",
        "xtpellm",
        "tpe_llm_soft",
        "xtpellmsoft",
    }
)


def _clip_p(p: float, lo: float = 0.2, hi: float = 0.8) -> float:
    return min(hi, max(lo, float(p)))


@dataclass
class ExplorePolicy:
    """C2: pick exploit (regions) vs explore (coverage_map) on a latched LLM turn."""

    mode: str = "fifty_fifty"
    seed: int = 1
    p_default: float = 0.5
    intensify_k: int = 3
    intensify_left: int = 0
    cached_trial: Optional[int] = None
    cached_branch: str = "exploit"
    last_p: float = 0.5
    last_meta: Dict[str, Any] = field(default_factory=dict)
    tpe_peek: Any = None
    tpe_min_trial: int = 0
    _selector: Any = None

    def _bandit(self):
        if self._selector is None:
            from mindware.components.optimizers.hybrid_selector import HybridSelector
            import numpy as np

            mode = self.mode
            sel_mode = "ap" if mode in ("hap", "xhap") else (
                "dts" if mode in ("dts", "xdts") else "ucb"
            )
            self._selector = HybridSelector(
                mode=sel_mode,
                llm_prob=0.5,
                ap_pmin=0.2,
                ap_beta=0.3,
                dts_gamma=0.95,
                ucb_c=1.4,
                pucb_q="breakthrough",
                rng=np.random.RandomState(int(self.seed) + 17),
            )
        return self._selector

    def choose(
        self,
        *,
        trial: int,
        successes: Sequence[Dict[str, Any]],
        config_space=None,
        fixed_algorithm: str = "",
        mode_tag: str = "cohort",
        exp_block: str = "",
    ) -> str:
        if self.cached_trial == int(trial):
            return self.cached_branch

        mode = (self.mode or "fifty_fifty").strip().lower()
        n = len(successes)
        stag = stagnation_count(successes)
        coverage = 50.0
        if config_space is not None:
            try:
                from mindware.components.optimizers.llm_prompt_blocks import mean_coverage_pct

                coverage = float(mean_coverage_pct(list(successes), config_space, fixed_algorithm))
            except Exception:
                coverage = 50.0

        if successes and n >= 2:
            last = float(successes[-1]["score"])
            prev_best = max(float(x["score"]) for x in successes[:-1])
            if last > prev_best + 1e-12:
                self.intensify_left = int(self.intensify_k)

        tpe_extra: Dict[str, Any] = {}
        p = float(self.p_default)
        branch: Optional[str] = None

        if mode in ("fifty_fifty", "50", "x50"):
            p = float(self.p_default)
        elif mode in ("stagnation", "stag", "xstag"):
            denom = max(5.0, 0.15 * max(n, 1))
            p = _clip_p(stag / denom)
        elif mode in ("coverage", "cov", "xcov"):
            p = _clip_p(1.0 - coverage / 100.0)
        elif mode in ("intensify", "int", "xint"):
            if self.intensify_left > 0:
                branch = "exploit"
                self.intensify_left -= 1
                p = 0.0
            else:
                p = 0.5
        elif mode in ("refine", "xrefine"):
            if n < 15 or stag >= 8 or coverage < 40.0:
                branch = "explore"
                p = 1.0
            else:
                branch = "exploit"
                p = 0.0
        elif mode in ("hap", "xhap", "dts", "xdts", "ucb", "xucb"):
            sel = self._bandit()
            M = max(n, 1)
            m = max(0, n - 1)
            arm = sel.choose(m, M, higher_is_better=True)
            branch = "exploit" if arm == "llm" else "explore"
            p = float(sel.ap_p.get("tpe", 0.5)) if mode in ("hap", "xhap") else 0.5
        elif mode in (
            "tpe",
            "xtpe",
            "tpe_soft",
            "xtpesoft",
            "tpe_llm",
            "xtpellm",
            "tpe_llm_soft",
            "xtpellmsoft",
        ):
            constraints = parse_prefer_constraints(exp_block)
            peek_cfg = None
            llm_only = mode in ("tpe_llm", "xtpellm", "tpe_llm_soft", "xtpellmsoft")
            min_trial = int(self.tpe_min_trial or 0)
            deferred = bool(min_trial > 0 and int(trial) < min_trial)
            if deferred:
                p = float(self.p_default)
                tpe_extra = {
                    "experience_explore_tpe_deferred": True,
                    "experience_explore_tpe_min_trial": min_trial,
                    "experience_explore_tpe_n_constraints": int(len(constraints)),
                    "experience_explore_tpe_scope": "llm" if llm_only else "all",
                }
            else:
                peek_fn = self.tpe_peek if callable(self.tpe_peek) else getattr(self, "tpe_peek", None)
                if callable(peek_fn):
                    try:
                        peek_cfg = peek_fn(llm_only=llm_only)
                    except TypeError:
                        peek_cfg = peek_fn()
                    except Exception:
                        peek_cfg = None
                ratio = config_prefer_hit_ratio(peek_cfg if isinstance(peek_cfg, dict) else None, constraints)
                tpe_extra = {
                    "experience_explore_tpe_ratio": float(ratio),
                    "experience_explore_tpe_n_constraints": int(len(constraints)),
                    "experience_explore_tpe_peeked": bool(peek_cfg),
                    "experience_explore_tpe_scope": "llm" if llm_only else "all",
                    "experience_explore_tpe_deferred": False,
                    "experience_explore_tpe_min_trial": min_trial,
                }
                if not constraints or peek_cfg is None:
                    p = float(self.p_default)
                elif mode in ("tpe_soft", "xtpesoft", "tpe_llm_soft", "xtpellmsoft"):
                    p = _clip_p(1.0 - ratio)
                elif ratio >= 0.5:
                    branch = "exploit"
                    p = 0.0
                else:
                    branch = "explore"
                    p = 1.0
        else:
            p = float(self.p_default)

        if branch is None:
            import hashlib

            key = f"{int(self.seed)}:{int(trial)}:{mode_tag}:{mode}".encode("utf-8")
            u = int(hashlib.md5(key).hexdigest()[:8], 16) / float(0xFFFFFFFF)
            branch = "explore" if u < p else "exploit"

        self.cached_trial = int(trial)
        self.cached_branch = branch
        self.last_p = float(p)
        self.last_meta = {
            "experience_explore_mode": mode,
            "experience_explore_branch": branch,
            "experience_explore_p": float(p),
            "experience_explore_stagnation": int(stag),
            "experience_explore_coverage": float(coverage),
            "experience_explore_intensify_left": int(self.intensify_left),
        }
        if tpe_extra:
            self.last_meta.update(tpe_extra)
        return branch

    def update(self, branch: str, perf: float, *, beat_incumbent: bool) -> None:
        if self._selector is None:
            return
        arm = "llm" if branch == "exploit" else "tpe"
        self._selector.update(
            arm,
            float(perf),
            higher_is_better=True,
            beat_incumbent=bool(beat_incumbent),
            beat_arm_best=bool(beat_incumbent),
        )
