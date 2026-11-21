#!/usr/bin/env python3
"""Mint synthetic TaskSet trajectories (best config + sampled worse configs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo.taskset import convert_taskset
from mindware.domains.paths import TASKSET_ROOT, domain_history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", type=Path, default=TASKSET_ROOT / "best_optimizer_per_task.json")
    p.add_argument("--spaces", type=Path, default=TASKSET_ROOT / "scripts/optimizer_search_spaces.json")
    p.add_argument("--out", type=Path, default=domain_history("dl"))
    p.add_argument("--n-fake", type=int, default=16)
    p.add_argument("--max-tasks", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.best.is_file():
        raise SystemExit(f"Missing TaskSet dump: {args.best}")
    written = convert_taskset(
        args.best,
        args.spaces,
        args.out,
        n_fake=args.n_fake,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )
    print(f"Wrote {len(written)} synthetic TaskSet histories under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Mint synthetic TaskSet trajectories (best config + sampled worse configs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo.taskset import convert_taskset
from mindware.domains.paths import TASKSET_ROOT, domain_history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", type=Path, default=TASKSET_ROOT / "best_optimizer_per_task.json")
    p.add_argument("--spaces", type=Path, default=TASKSET_ROOT / "scripts/optimizer_search_spaces.json")
    p.add_argument("--out", type=Path, default=domain_history("dl"))
    p.add_argument("--n-fake", type=int, default=16)
    p.add_argument("--max-tasks", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.best.is_file():
        raise SystemExit(f"Missing TaskSet dump: {args.best}")
    written = convert_taskset(
        args.best,
        args.spaces,
        args.out,
        n_fake=args.n_fake,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )
    print(f"Wrote {len(written)} synthetic TaskSet histories under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Mint synthetic TaskSet trajectories (best config + sampled worse configs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo.taskset import convert_taskset
from mindware.domains.paths import TASKSET_ROOT, domain_history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", type=Path, default=TASKSET_ROOT / "best_optimizer_per_task.json")
    p.add_argument("--spaces", type=Path, default=TASKSET_ROOT / "scripts/optimizer_search_spaces.json")
    p.add_argument("--out", type=Path, default=domain_history("dl"))
    p.add_argument("--n-fake", type=int, default=16)
    p.add_argument("--max-tasks", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.best.is_file():
        raise SystemExit(f"Missing TaskSet dump: {args.best}")
    written = convert_taskset(
        args.best,
        args.spaces,
        args.out,
        n_fake=args.n_fake,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )
    print(f"Wrote {len(written)} synthetic TaskSet histories under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Mint synthetic TaskSet trajectories (best config + sampled worse configs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mindware.domains.dlhpo.taskset import convert_taskset
from mindware.domains.paths import TASKSET_ROOT, domain_history


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--best", type=Path, default=TASKSET_ROOT / "best_optimizer_per_task.json")
    p.add_argument("--spaces", type=Path, default=TASKSET_ROOT / "scripts/optimizer_search_spaces.json")
    p.add_argument("--out", type=Path, default=domain_history("dl"))
    p.add_argument("--n-fake", type=int, default=16)
    p.add_argument("--max-tasks", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not args.best.is_file():
        raise SystemExit(f"Missing TaskSet dump: {args.best}")
    written = convert_taskset(
        args.best,
        args.spaces,
        args.out,
        n_fake=args.n_fake,
        max_tasks=args.max_tasks,
        seed=args.seed,
    )
    print(f"Wrote {len(written)} synthetic TaskSet histories under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
