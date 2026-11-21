"""TaskSet adaptation of the original ML S1B/S2-G data contract."""

import importlib.util
import math
from pathlib import Path

ML_SELECTOR = Path(__file__).resolve().parents[1] / "export_p5_s2_dpo_pareto.py"
_spec = importlib.util.spec_from_file_location("taskset_ml_s2g_selector", ML_SELECTOR)
ML = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ML)


def improvement_events(rows):
    best, successful = math.inf, []
    for position, row in enumerate(rows):
        if not row["successful"]:
            continue
        if row["loss"] < best:
            if len(successful) >= 2:
                losses = [r["loss"] for r in successful]
                lo, hi = min(losses), max(losses)
                delta = (lo - row["loss"]) / (hi - lo) if hi != lo else 0.0
                yield dict(row, position=position, prefix_successes=len(successful), delta_rel_score=delta)
            best = row["loss"]
        successful.append(row)


def structured_space(cs):
    from ConfigSpace.hyperparameters import CategoricalHyperparameter, Constant

    result = {}
    for hp in cs.get_hyperparameters():
        if isinstance(hp, (CategoricalHyperparameter, Constant)):
            result[hp.name] = {"kind": "categorical" if isinstance(hp, CategoricalHyperparameter) else "constant"}
        else:
            result[hp.name] = dict(kind="float", low=float(hp.lower), high=float(hp.upper), log=bool(hp.log))
    return result


def preference_pair(rows, event, space):
    # Reuse ML's distance and Pareto implementation, including its stable tie order.
    seen = {ML.cfg_key(r["config"]) for r in rows[:event["position"] + 1] if r["successful"]}
    pool = []
    for row in rows:
        if (not row["successful"] or ML.cfg_key(row["config"]) in seen
                or row["loss"] - event["loss"] <= 1e-12):
            continue
        pool.append(dict(row=row, sim=ML.similarity(row["config"], event["config"], space),
                         gap=row["loss"] - event["loss"]))
    if not pool:
        return None
    lo, hi = min(p["gap"] for p in pool), max(p["gap"] for p in pool)
    for p in pool:
        p["worse_norm"] = 1.0 if hi <= lo + 1e-15 else (p["gap"] - lo) / (hi - lo)
        p["weighted"] = 0.5 * p["sim"] + 0.5 * p["worse_norm"]
    front = ML.pareto_front(pool)
    pick = max(front, key=lambda p: (p["weighted"], p["sim"], p["worse_norm"]))
    return dict(chosen=event["config"], rejected=pick["row"]["config"],
                chosen_loss=event["loss"], rejected_loss=pick["row"]["loss"],
                rejected_trial=pick["row"]["trial"], sim=pick["sim"], worse_norm=pick["worse_norm"],
                weighted=pick["weighted"], pool_size=len(pool), pareto_size=len(front))
