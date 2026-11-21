#!/usr/bin/env python3
"""Debug RGS pooled normalization for one task (pool drop-worst 10%, min→0 max→1)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec_path = REPO / "examples/benchmark/plot_mltest20_algo40_avg_rank.py"
spec = importlib.util.spec_from_file_location("plot_algo40", spec_path)
plot = importlib.util.module_from_spec(spec)
sys.modules["plot_algo40"] = plot
spec.loader.exec_module(plot)

TASK = ("rgs", "house_sales", "lightgbm")
KEY = "rgs/house_sales/lightgbm"
sources = plot.build_sources()
variants = [v for v in sources if not plot.is_smac_arm(v)]


def main() -> None:
    tt, ds, _algo = TASK
    scores: dict[str, float] = {}
    pool: list[float] = []
    for label, (root, disk_var) in sources.items():
        if label not in variants:
            continue
        traj = plot.find_traj(root, tt, ds, disk_var, 80)
        if traj is None:
            continue
        perfs = plot.trial_perfs(traj, 80)
        pool.extend(perfs)
        best = plot.final_best(traj)
        if best is not None:
            scores[label] = best
        print(
            f"{label[:40]:40s} max_perf={max(perfs) if perfs else float('nan'):.4g} "
            f"final_best={best:.4g}"
        )

    norms = plot.normalize_rgs_from_pooled_drop_worst(scores, pool)
    xs = sorted(pool)
    n_drop = int(math.floor(0.1 * len(xs)))
    kept = xs[n_drop:]
    lo, hi = kept[0], kept[-1]
    print(f"\nPool n={len(pool)} drop_worst={n_drop} kept={len(kept)} lo={lo:.6g} hi={hi:.6g}")
    print("norms (higher better):")
    for k, v in sorted(norms.items(), key=lambda kv: -kv[1]):
        print(f"  {k[:42]:42s} score={scores[k]:.6g} norm={v:.4f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Debug RGS pooled normalization for one task (pool drop-worst 10%, min→0 max→1)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec_path = REPO / "examples/benchmark/plot_mltest20_algo40_avg_rank.py"
spec = importlib.util.spec_from_file_location("plot_algo40", spec_path)
plot = importlib.util.module_from_spec(spec)
sys.modules["plot_algo40"] = plot
spec.loader.exec_module(plot)

TASK = ("rgs", "house_sales", "lightgbm")
KEY = "rgs/house_sales/lightgbm"
sources = plot.build_sources()
variants = [v for v in sources if not plot.is_smac_arm(v)]


def main() -> None:
    tt, ds, _algo = TASK
    scores: dict[str, float] = {}
    pool: list[float] = []
    for label, (root, disk_var) in sources.items():
        if label not in variants:
            continue
        traj = plot.find_traj(root, tt, ds, disk_var, 80)
        if traj is None:
            continue
        perfs = plot.trial_perfs(traj, 80)
        pool.extend(perfs)
        best = plot.final_best(traj)
        if best is not None:
            scores[label] = best
        print(
            f"{label[:40]:40s} max_perf={max(perfs) if perfs else float('nan'):.4g} "
            f"final_best={best:.4g}"
        )

    norms = plot.normalize_rgs_from_pooled_drop_worst(scores, pool)
    xs = sorted(pool)
    n_drop = int(math.floor(0.1 * len(xs)))
    kept = xs[n_drop:]
    lo, hi = kept[0], kept[-1]
    print(f"\nPool n={len(pool)} drop_worst={n_drop} kept={len(kept)} lo={lo:.6g} hi={hi:.6g}")
    print("norms (higher better):")
    for k, v in sorted(norms.items(), key=lambda kv: -kv[1]):
        print(f"  {k[:42]:42s} score={scores[k]:.6g} norm={v:.4f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Debug RGS pooled normalization for one task (pool drop-worst 10%, min→0 max→1)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec_path = REPO / "examples/benchmark/plot_mltest20_algo40_avg_rank.py"
spec = importlib.util.spec_from_file_location("plot_algo40", spec_path)
plot = importlib.util.module_from_spec(spec)
sys.modules["plot_algo40"] = plot
spec.loader.exec_module(plot)

TASK = ("rgs", "house_sales", "lightgbm")
KEY = "rgs/house_sales/lightgbm"
sources = plot.build_sources()
variants = [v for v in sources if not plot.is_smac_arm(v)]


def main() -> None:
    tt, ds, _algo = TASK
    scores: dict[str, float] = {}
    pool: list[float] = []
    for label, (root, disk_var) in sources.items():
        if label not in variants:
            continue
        traj = plot.find_traj(root, tt, ds, disk_var, 80)
        if traj is None:
            continue
        perfs = plot.trial_perfs(traj, 80)
        pool.extend(perfs)
        best = plot.final_best(traj)
        if best is not None:
            scores[label] = best
        print(
            f"{label[:40]:40s} max_perf={max(perfs) if perfs else float('nan'):.4g} "
            f"final_best={best:.4g}"
        )

    norms = plot.normalize_rgs_from_pooled_drop_worst(scores, pool)
    xs = sorted(pool)
    n_drop = int(math.floor(0.1 * len(xs)))
    kept = xs[n_drop:]
    lo, hi = kept[0], kept[-1]
    print(f"\nPool n={len(pool)} drop_worst={n_drop} kept={len(kept)} lo={lo:.6g} hi={hi:.6g}")
    print("norms (higher better):")
    for k, v in sorted(norms.items(), key=lambda kv: -kv[1]):
        print(f"  {k[:42]:42s} score={scores[k]:.6g} norm={v:.4f}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Debug RGS pooled normalization for one task (pool drop-worst 10%, min→0 max→1)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec_path = REPO / "examples/benchmark/plot_mltest20_algo40_avg_rank.py"
spec = importlib.util.spec_from_file_location("plot_algo40", spec_path)
plot = importlib.util.module_from_spec(spec)
sys.modules["plot_algo40"] = plot
spec.loader.exec_module(plot)

TASK = ("rgs", "house_sales", "lightgbm")
KEY = "rgs/house_sales/lightgbm"
sources = plot.build_sources()
variants = [v for v in sources if not plot.is_smac_arm(v)]


def main() -> None:
    tt, ds, _algo = TASK
    scores: dict[str, float] = {}
    pool: list[float] = []
    for label, (root, disk_var) in sources.items():
        if label not in variants:
            continue
        traj = plot.find_traj(root, tt, ds, disk_var, 80)
        if traj is None:
            continue
        perfs = plot.trial_perfs(traj, 80)
        pool.extend(perfs)
        best = plot.final_best(traj)
        if best is not None:
            scores[label] = best
        print(
            f"{label[:40]:40s} max_perf={max(perfs) if perfs else float('nan'):.4g} "
            f"final_best={best:.4g}"
        )

    norms = plot.normalize_rgs_from_pooled_drop_worst(scores, pool)
    xs = sorted(pool)
    n_drop = int(math.floor(0.1 * len(xs)))
    kept = xs[n_drop:]
    lo, hi = kept[0], kept[-1]
    print(f"\nPool n={len(pool)} drop_worst={n_drop} kept={len(kept)} lo={lo:.6g} hi={hi:.6g}")
    print("norms (higher better):")
    for k, v in sorted(norms.items(), key=lambda kv: -kv[1]):
        print(f"  {k[:42]:42s} score={scores[k]:.6g} norm={v:.4f}")


if __name__ == "__main__":
    main()
