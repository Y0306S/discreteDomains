"""Two-arm LLM vs TPE selector: Bernoulli, UCB, and CoFEH-style PUCB.

Warm-start trials must not call ``update``. Bandit step ``m`` is the number of
post-warm-start selections already made (0-based at the first adaptive pull).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

ARMS = ("llm", "tpe")
BO_ARMS = ("tpe", "cmaes", "smac", "hebo", "rgpe", "fsbo")


def _safe_div(num: float, den: float, default: float = 0.5) -> float:
    if abs(den) < 1e-12:
        return default
    return num / den


class HybridSelector:
    """Choose ``llm`` or ``tpe`` each trial after warm-start."""

    def __init__(
        self,
        *,
        mode: str = "bernoulli",
        llm_prob: float = 0.5,
        ucb_c: float = 1.4,
        pucb_c2: float = 1.4,
        pucb_p1: float = 0.9,
        pucb_p_end: Optional[float] = None,
        pucb_floor: Optional[float] = None,
        pucb_floor_after_frac: Optional[float] = None,
        pucb_switch_after_frac: Optional[float] = None,
        pucb_q: str = "window_minmax",
        pucb_window: int = 10,
        dts_gamma: float = 0.95,
        ap_beta: float = 0.3,
        ap_pmin: float = 0.2,
        bo_arm: str = "tpe",
        rng=None,
    ):
        self.mode = str(mode or "bernoulli").strip().lower()
        self.llm_prob = min(1.0, max(0.0, float(llm_prob)))
        bo = str(bo_arm or "tpe").strip().lower()
        self.bo_arm = bo if bo in BO_ARMS else "tpe"
        self.arms: Tuple[str, str] = ("llm", self.bo_arm)
        self.ucb_c = float(ucb_c)
        self.pucb_c2 = float(pucb_c2)
        self.pucb_p1 = float(pucb_p1)
        self.pucb_p2 = 1.0 - self.pucb_p1
        # None = old CoFEH 0.9→0.1. 0.5 = LLM 0.9→0.5 and TPE 0.1→0.5.
        self.pucb_p_end = None if pucb_p_end is None else float(pucb_p_end)
        # Late-only floor: keep the 0.9→0.1 decay, then clamp ω_llm once frac >= after.
        self.pucb_floor = None if pucb_floor is None else float(pucb_floor)
        self.pucb_floor_after_frac = (
            None if pucb_floor_after_frac is None else float(pucb_floor_after_frac)
        )
        # After this frac of the adaptive budget, PUCB hands off to Bernoulli(llm_prob).
        self.pucb_switch_after_frac = (
            None if pucb_switch_after_frac is None else float(pucb_switch_after_frac)
        )
        self.pucb_q = str(pucb_q or "window_minmax").strip().lower()
        self.pucb_window = max(1, int(pucb_window))
        self.dts_gamma = min(1.0, max(0.5, float(dts_gamma)))
        self.ap_beta = min(1.0, max(0.0, float(ap_beta)))
        self.ap_pmin = min(0.49, max(0.0, float(ap_pmin)))
        if rng is None:
            import numpy as np

            rng = np.random.RandomState(0)
        self.rng = rng

        self.n: Dict[str, int] = {a: 0 for a in self.arms}
        self.attempt_n: Dict[str, int] = {a: 0 for a in self.arms}
        self.successes: Dict[str, int] = {a: 0 for a in self.arms}
        self.failures: Dict[str, float] = {a: 0.0 for a in self.arms}
        self.arm_best: Dict[str, Optional[float]] = {a: None for a in self.arms}
        self.window: Deque[Tuple[str, float]] = deque(maxlen=self.pucb_window)
        self.t = 0  # post-warm-start selections (increments on choose)
        self.last_snapshot: Dict[str, Any] = {}
        self.ap_p: Dict[str, float] = {a: 0.5 for a in self.arms}
        self.ap_q: Dict[str, float] = {a: 0.5 for a in self.arms}
        self.disc_sum: Dict[str, float] = {a: 0.0 for a in self.arms}
        self.disc_n: Dict[str, float] = {a: 0.0 for a in self.arms}
        self.ph_mean = 0.0
        self.ph_m = 0.0
        self.ph_max = 0.0
        self.ph_n = 0

    def _frac(self, m: int, M: int) -> float:
        return 0.0 if M <= 0 else min(1.0, max(0.0, float(m) / float(M)))

    def _floor_active(self, frac: float) -> bool:
        if self.pucb_floor is None or self.pucb_floor_after_frac is None:
            return False
        return frac + 1e-12 >= self.pucb_floor_after_frac

    def omega(self, arm: str, m: int, M: int) -> float:
        frac = self._frac(m, M)
        llm_end = self.pucb_p2 if self.pucb_p_end is None else self.pucb_p_end
        tpe_end = self.pucb_p1 if self.pucb_p_end is None else self.pucb_p_end
        if arm == "llm":
            # default 0.9 → 0.1; with p_end=0.5: 0.9 → 0.5
            w = self.pucb_p1 - (self.pucb_p1 - llm_end) * frac
            if self._floor_active(frac):
                w = max(w, self.pucb_floor)
            return w
        # default 0.1 → 0.9; with p_end=0.5: 0.1 → 0.5
        w = self.pucb_p2 + (tpe_end - self.pucb_p2) * frac
        if self._floor_active(frac):
            w = min(w, 1.0 - self.pucb_floor)
        return w

    def q_values(self, higher_is_better: bool = True) -> Dict[str, float]:
        kind = self.pucb_q
        if kind in ("window_minmax", "window", "minmax"):
            return self._q_window_minmax(higher_is_better=higher_is_better)
        if kind in ("breakthrough", "incumbent"):
            return {a: (self.successes[a] + 1.0) / (self.n[a] + 2.0) for a in self.arms}
        if kind in ("arm_best", "arm"):
            return {a: (self.successes[a] + 1.0) / (self.n[a] + 2.0) for a in self.arms}
        return self._q_window_minmax(higher_is_better=higher_is_better)

    def _q_window_minmax(self, *, higher_is_better: bool) -> Dict[str, float]:
        best: Dict[str, Optional[float]] = {a: None for a in self.arms}
        for arm, score in self.window:
            if not math.isfinite(score):
                continue
            cur = best[arm]
            if cur is None:
                best[arm] = score
            elif higher_is_better and score > cur:
                best[arm] = score
            elif (not higher_is_better) and score < cur:
                best[arm] = score
        present = [best[a] for a in self.arms if best[a] is not None]
        if len(present) < 2:
            return {a: 0.5 if best[a] is None else 0.5 for a in self.arms}
        lo, hi = min(present), max(present)
        out: Dict[str, float] = {}
        for a in self.arms:
            if best[a] is None:
                out[a] = 0.5
            else:
                val = best[a] if higher_is_better else -best[a]
                lo_o = lo if higher_is_better else -hi
                hi_o = hi if higher_is_better else -lo
                out[a] = _safe_div(val - lo_o, hi_o - lo_o, 0.5)
                out[a] = min(1.0, max(0.0, out[a]))
        return out

    def choose(self, m: int, M: int, *, higher_is_better: bool = True) -> str:
        """Return arm name. ``m`` is current post-warm-start index; ``M`` is total budget."""
        self.t += 1
        q = self.q_values(higher_is_better=higher_is_better)
        omega = {a: self.omega(a, m, M) for a in self.arms}
        scores: Dict[str, float] = {}
        draw: Optional[float] = None

        frac = self._frac(m, M)
        switch = (
            self.pucb_switch_after_frac is not None
            and frac + 1e-12 >= self.pucb_switch_after_frac
        )

        if self.mode in ("bernoulli", "mix", "fixed") or (
            switch and self.mode in ("pucb", "rucb")
        ):
            draw = float(self.rng.rand())
            arm = "llm" if draw < self.llm_prob else self.bo_arm
            scores = {a: (1.0 if a == "llm" else 0.0) for a in self.arms}
        elif self.mode == "ucb":
            unused = [a for a in self.arms if self.n[a] == 0]
            if unused:
                arm = unused[0]
            else:
                t = max(self.t, 1)
                for a in self.arms:
                    bonus = self.ucb_c * math.sqrt(math.log(t) / max(self.n[a], 1))
                    scores[a] = q[a] + bonus
                arm = max(self.arms, key=lambda a: scores[a])
        elif self.mode in ("pucb", "rucb"):
            attempts = self.attempt_counts()
            unused = [a for a in self.arms if attempts[a] == 0]
            if unused:
                arm = unused[0]
            else:
                n_sum = sum(attempts.values())
                for a in self.arms:
                    explore = math.sqrt(n_sum / (1.0 + attempts[a]))
                    scores[a] = q[a] + self.pucb_c2 * omega[a] * explore
                arm = max(self.arms, key=lambda a: scores[a])
        elif self.mode in ("dts", "thompson", "hdts"):
            arm = self._choose_dts()
        elif self.mode in ("ap", "pursuit", "hap"):
            draw = float(self.rng.rand())
            arm = "llm" if draw < self.ap_p["llm"] else self.bo_arm
        elif self.mode in ("ducb", "d-ucb"):
            arm = self._choose_ducb()
        elif self.mode in ("dmab",):
            arm = self._choose_ucb_like(q)
        elif self.mode in ("transient", "htran"):
            # LLINBO-Transient: LLM prob ~ 1/t, floored so the arm never dies.
            t = max(self.t, 1)
            p_llm = min(0.9, max(0.1, 8.0 / float(t)))
            draw = float(self.rng.rand())
            arm = "llm" if draw < p_llm else self.bo_arm
            scores = {"llm": p_llm, self.bo_arm: 1.0 - p_llm}
        else:
            draw = float(self.rng.rand())
            arm = "llm" if draw < self.llm_prob else self.bo_arm

        bo = self.bo_arm
        self.last_snapshot = {
            "hybrid_selector": self.mode,
            "pucb_q": self.pucb_q,
            "chosen_arm": arm,
            "bo_arm": bo,
            "Q_llm": q["llm"],
            "Q_tpe": q.get("tpe", q.get(bo, 0.5)),
            "Q_bo": q.get(bo, 0.5),
            "N_llm": self.n["llm"],
            "N_tpe": self.n.get("tpe", self.n.get(bo, 0)),
            "N_bo": self.n.get(bo, 0),
            "N_attempt_llm": self.attempt_counts()["llm"],
            "N_attempt_tpe": self.attempt_counts().get("tpe", self.attempt_counts().get(bo, 0)),
            "omega_llm": omega["llm"],
            "omega_tpe": omega.get("tpe", omega.get(bo, 0.0)),
            "omega_bo": omega.get(bo, 0.0),
            "selection_draw": draw,
            "ucb_c": self.ucb_c,
            "pucb_c2": self.pucb_c2,
            "pucb_p1": self.pucb_p1,
            "pucb_p_end": self.pucb_p_end,
            "pucb_floor": self.pucb_floor,
            "pucb_floor_after_frac": self.pucb_floor_after_frac,
            "pucb_switch_after_frac": self.pucb_switch_after_frac,
            "ap_p_llm": self.ap_p["llm"],
            "dts_gamma": self.dts_gamma,
            "m": m,
            "M": M,
        }
        return arm

    def _choose_dts(self) -> str:
        samples = {}
        for a in self.arms:
            alpha = 1.0 + float(self.successes[a])
            beta = 1.0 + float(self.failures[a])
            samples[a] = float(self.rng.beta(alpha, beta))
        return max(self.arms, key=lambda a: samples[a])

    def _choose_ducb(self) -> str:
        unused = [a for a in self.arms if self.disc_n[a] <= 1e-9]
        if unused:
            return unused[0]
        n_sum = sum(self.disc_n.values())
        best = None
        best_s = -1e18
        for a in self.arms:
            mean = self.disc_sum[a] / max(self.disc_n[a], 1e-9)
            bonus = self.ucb_c * math.sqrt(math.log(max(n_sum, 1.0)) / max(self.disc_n[a], 1e-9))
            s = mean + bonus
            if s > best_s:
                best_s = s
                best = a
        return best or self.bo_arm

    def _choose_ucb_like(self, q: Dict[str, float]) -> str:
        unused = [a for a in self.arms if self.n[a] == 0]
        if unused:
            return unused[0]
        t = max(self.t, 1)
        best = None
        best_s = -1e18
        for a in self.arms:
            bonus = self.ucb_c * math.sqrt(math.log(t) / max(self.n[a], 1))
            s = q[a] + bonus
            if s > best_s:
                best_s = s
                best = a
        return best or self.bo_arm

    def _reset_bandit(self) -> None:
        self.n = {a: 0 for a in self.arms}
        self.attempt_n = {a: 0 for a in self.arms}
        self.successes = {a: 0 for a in self.arms}
        self.failures = {a: 0.0 for a in self.arms}
        self.window.clear()
        self.disc_sum = {a: 0.0 for a in self.arms}
        self.disc_n = {a: 0.0 for a in self.arms}
        self.ph_mean = 0.0
        self.ph_m = 0.0
        self.ph_max = 0.0
        self.ph_n = 0

    def attempt_counts(self) -> Dict[str, int]:
        if not hasattr(self, "attempt_n"):
            self.attempt_n = dict(self.n)
        return self.attempt_n

    def record_attempt(self, arm: str) -> None:
        """Count an attempted arm without assigning an evaluated reward to it."""
        if arm in self.arms:
            self.attempt_counts()[arm] += 1

    def update(
        self,
        arm: str,
        perf: float,
        *,
        higher_is_better: bool = True,
        beat_incumbent: bool = False,
        beat_arm_best: bool = False,
    ) -> None:
        if arm not in self.arms:
            return
        score = float(perf)
        self.record_attempt(arm)
        self.n[arm] += 1
        self.window.append((arm, score))
        prev = self.arm_best[arm]
        valid = math.isfinite(score)
        improved_self = valid and (
            prev is None or not math.isfinite(prev) or
            (score > prev if higher_is_better else score < prev)
        )
        if improved_self:
            self.arm_best[arm] = score
        if self.mode in ("dts", "thompson", "hdts"):
            g = self.dts_gamma
            for a in self.arms:
                self.successes[a] = float(self.successes[a]) * g
                self.failures[a] = float(self.failures[a]) * g
        if self.pucb_q in ("arm_best", "arm"):
            win = valid and bool(beat_arm_best or improved_self)
        else:
            win = valid and bool(beat_incumbent)
        if win:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1.0

        g = self.dts_gamma
        r = 1.0 if valid and beat_incumbent else 0.0
        for a in self.arms:
            self.disc_sum[a] *= g
            self.disc_n[a] *= g
        self.disc_sum[arm] += r
        self.disc_n[arm] += 1.0

        q = self.q_values(higher_is_better=higher_is_better)
        alpha = 0.3
        for a in self.arms:
            self.ap_q[a] = (1.0 - alpha) * self.ap_q[a] + alpha * q[a]
        winner = max(self.arms, key=lambda a: self.ap_q[a])
        pmax = 1.0 - self.ap_pmin
        for a in self.arms:
            target = pmax if a == winner else self.ap_pmin
            self.ap_p[a] = self.ap_p[a] + self.ap_beta * (target - self.ap_p[a])
        s = sum(self.ap_p.values()) or 1.0
        self.ap_p = {a: self.ap_p[a] / s for a in self.arms}

        if self.mode == "dmab":
            self._page_hinkley(r)

    def _page_hinkley(self, reward: float, delta: float = 0.15, gamma: float = 8.0) -> None:
        self.ph_n += 1
        self.ph_mean = self.ph_mean + (reward - self.ph_mean) / float(self.ph_n)
        e = reward - self.ph_mean + delta
        self.ph_m += e
        self.ph_max = max(self.ph_max, abs(self.ph_m))
        if self.ph_max - abs(self.ph_m) > gamma and self.ph_n >= 8:
            self._reset_bandit()

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.last_snapshot)


def selector_from_params(params: Dict[str, Any], rng=None) -> HybridSelector:
    mode = str(params.get("hybrid_selector") or "bernoulli").strip().lower()
    return HybridSelector(
        mode=mode,
        llm_prob=float(params.get("llm_prob", 0.5)),
        ucb_c=float(params.get("ucb_c", 1.4)),
        pucb_c2=float(params.get("pucb_c2", 1.4)),
        pucb_p1=float(params.get("pucb_p1", 0.9)),
        pucb_p_end=(None if params.get("pucb_p_end") is None else float(params.get("pucb_p_end"))),
        pucb_floor=(None if params.get("pucb_floor") is None else float(params.get("pucb_floor"))),
        pucb_floor_after_frac=(
            None
            if params.get("pucb_floor_after_frac") is None
            else float(params.get("pucb_floor_after_frac"))
        ),
        pucb_switch_after_frac=(
            None
            if params.get("pucb_switch_after_frac") is None
            else float(params.get("pucb_switch_after_frac"))
        ),
        pucb_q=str(params.get("pucb_q") or "window_minmax"),
        pucb_window=int(params.get("pucb_window") or 10),
        dts_gamma=float(params.get("dts_gamma", 0.95)),
        ap_beta=float(params.get("ap_beta", 0.3)),
        ap_pmin=float(params.get("ap_pmin", 0.2)),
        bo_arm=str(params.get("bo_arm") or params.get("hybrid_bo_arm") or "tpe"),
        rng=rng,
    )
