#!/usr/bin/env python3
"""TaskSet evaluation matching Metz et al. 2020 §3.3 (paper-exact scoring).

Training (via task_set.train_inner):
  - 10,000 optimizer steps
  - eval every 200 steps, 50 mini-batches per eval (train_inner defaults)
  - 5 random parameter initializations (replicas; separate processes)

Scoring per task:
  - Use VALID_OUTER validation loss (meta-validation split)
  - Normalize: 1 = val loss at init, 0 = best val across grid optimizers on task
  - Clip normalized values > 1 to 1
  - Scalar f = mean normalized val loss over all logged steps
  - Average f over 5 replicas; pick best candidate prediction by lowest f

Requires a baseline cache (--baseline-cache) with per-task best_val from the
optimizer search grid, or build it with --build-baselines-only first.

Debugging:
  --debug                  save per-replica train_inner logs + manifest.json
  --skip-baseline-build    evaluate model configs only (requires baseline cache)
  --smoke-test             reduced steps/replicas for quick pipeline checks
  --force-optimizer-register  bypass duplicate optimizer name collisions
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_INNER_STEP_RE = re.compile(r"Evaluating (\d+)")
# Cap OpenMP/TF threads per train_inner subprocess (parallel eval runs many).
TRAIN_INNER_SUBPROCESS_THREADS = 4


def apply_train_inner_thread_env(env: dict[str, str], threads: int | None = None) -> int:
    """Limit OMP/MKL/TF thread pools for parallel CPU eval."""
    count = threads if threads is not None else TRAIN_INNER_SUBPROCESS_THREADS
    env["OMP_NUM_THREADS"] = str(count)
    env["OPENBLAS_NUM_THREADS"] = str(count)
    env["MKL_NUM_THREADS"] = str(count)
    env["TF_NUM_INTRAOP_THREADS"] = str(count)
    env["TF_NUM_INTEROP_THREADS"] = str(max(2, count // 2))
    return count


def resolve_task_set_root() -> Path:
    for candidate in (
        Path("/root/data/cys/google-research"),
        Path("/root/LLMOptimizer/LLMOptimizer/google-research"),
        Path("/root/data/cys/blockdata_archive/LLMOptimizer/google-research"),
    ):
        if (candidate / "task_set/train_inner.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate google-research task_set checkout.")


TASK_SET_ROOT = resolve_task_set_root()
TASK_CONFIGS_DIR = TASK_SET_ROOT / "task_set/tasks/configs"
OPTIMIZER_CONFIGS_PATH = TASK_SET_ROOT / "task_set/optimizers/configs"
CONDA_ACTIVATE = "/root/miniconda3/bin/activate"
CONDA_ENV = "taskset_eval"
# Direct interpreter (preferred): conda activate still works via
# /root/miniconda3/envs/taskset_eval -> /root/data/cys/conda_envs/taskset_eval
TASKSET_EVAL_PYTHON = next(
    (
        Path(p)
        for p in (
            os.environ.get("TASKSET_EVAL_PYTHON", ""),
            "/root/data/cys/conda_envs/taskset_eval/bin/python",
            "/root/miniconda3/envs/taskset_eval/bin/python",
            "/root/data/cys/blockdata_archive/miniconda3/envs/taskset_eval/bin/python",
        )
        if p and Path(p).exists()
    ),
    Path("/root/data/cys/conda_envs/taskset_eval/bin/python"),
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_config_cli import extract_json_object
from taskset_paper_scoring import (
    CPU_BASELINE_TASK_WORKERS,
    CPU_TASK_WORKERS,
    PAPER_BASELINE_TASK_WORKERS,
    PAPER_EVAL_EVERY_N,
    PAPER_NUM_REPLICAS,
    PAPER_REPLICA_AGGREGATION,
    PAPER_TASK_WORKERS,
    PAPER_TRAINING_STEPS,
    WIDE_GRID_FAMILIES,
    aggregate_replica_scores,
    get_task_best_val,
    list_grid_optimizer_names,
    load_baseline_cache,
    load_json,
    min_valid_outer,
    paper_eval_protocol,
    save_baseline_cache,
    score_replica_curve,
    save_json,
)


class EvalProgress:
    """Write machine-readable progress to work-dir/progress.json for tail/monitoring."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, Any] = {"updated_at": self._now()}
        self._write()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write(self) -> None:
        self._state["updated_at"] = self._now()
        save_json(self.path, self._state)

    def start_phase(self, phase: str, total: int, **extra: Any) -> None:
        self._state = {
            "phase": phase,
            "total": total,
            "done": 0,
            "updated_at": self._now(),
            **extra,
        }
        self._write()

    def set_active_run(
        self,
        *,
        task_index: int | None = None,
        task_name: str | None = None,
        optimizer_name: str | None = None,
        replica: int | None = None,
        step: int | None = None,
        training_steps: int | None = None,
        label: str | None = None,
    ) -> None:
        self._state["active_run"] = {
            k: v
            for k, v in {
                "task_index": task_index,
                "task_name": task_name,
                "optimizer_name": optimizer_name,
                "replica": replica,
                "step": step,
                "training_steps": training_steps,
                "label": label,
                "updated_at": self._now(),
            }.items()
            if v is not None
        }
        self._write()

    def task_finished(self, task_index: int, status: str, **extra: Any) -> None:
        self._state["done"] = int(self._state.get("done", 0)) + 1
        completed = self._state.setdefault("completed_tasks", [])
        completed.append({"index": task_index, "status": status, **extra})
        self._write()

    def restore_model_eval(self, results: list[dict], total: int, **extra: Any) -> None:
        """Restore progress counter from a checkpoint (resume)."""
        self._state = {
            "phase": "model_eval",
            "total": total,
            "done": len(results),
            "completed_tasks": [
                {"index": r.get("index"), "status": r.get("status", "unknown")}
                for r in results
            ],
            "updated_at": self._now(),
            **extra,
        }
        self._write()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_checkpoint_path(work_dir: Path) -> Path:
    return work_dir / "eval_checkpoint.json"


def load_eval_checkpoint(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    payload = load_json(path)
    if isinstance(payload, dict):
        return list(payload.get("results") or [])
    if isinstance(payload, list):
        return payload
    return []


def save_eval_checkpoint(path: Path, results: list[dict], *, model_label: str | None = None) -> None:
    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "results": sorted(results, key=lambda r: r.get("index", -1)),
    }
    if model_label:
        payload["model"] = model_label
    save_json(path, payload)


def is_valid_replica_curves(curves: Any) -> bool:
    return isinstance(curves, dict) and bool(curves)


def try_load_cached_replica_result(
    replica_dir: Path,
    replica: int,
    *,
    debug: bool = False,
) -> dict[str, Any] | None:
    result_file = replica_dir / "result"
    if not result_file.is_file():
        return None
    try:
        curves = load_json(result_file)
    except (json.JSONDecodeError, OSError):
        return None
    if not is_valid_replica_curves(curves):
        return None
    if debug:
        print(f"  [debug] resume cached replica={replica} -> {result_file}")
    return {
        "replica": replica,
        "status": "ok",
        "curves": curves,
        "result_dir": str(replica_dir),
        "elapsed_s": 0.0,
        "resumed": True,
    }


def resolve_task_config_path(task_family: str) -> Path:
    primary = TASK_CONFIGS_DIR / f"{task_family}.json"
    if primary.exists():
        return primary
    if task_family.endswith("_family"):
        alt = TASK_CONFIGS_DIR / f"{task_family[: -len('_family')]}.json"
        if alt.exists():
            return alt
    raise FileNotFoundError(f"No task config file for family {task_family!r}")


def normalize_task_for_config_match(task_family: str, target_task: dict | list) -> dict | list:
    """Normalize JSONL task specs to the format stored in task config files."""
    if task_family == "losg_tasks_family" and isinstance(target_task, dict):
        if "losg_problem" in target_task:
            return [
                target_task["losg_problem"],
                target_task["params"],
                target_task["init_seed"],
            ]
    return target_task


def find_task_seed(task_family: str, target_task: dict) -> str | None:
    needle = normalize_task_for_config_match(task_family, target_task)
    with resolve_task_config_path(task_family).open() as f:
        task_configs = json.load(f)
    for key, cfg_list in task_configs.items():
        if cfg_list[0] == needle:
            return key
    # dl_test_15_modified may reference quadratic tasks not in the 1000 seed file.
    if task_family == "quadratic_family" and isinstance(needle, dict):
        dltest_key = f"quadratic_family_dltest_{needle.get('seed', 'unknown')}"
        if dltest_key in task_configs:
            return dltest_key
    return None


def search_space_families(gold_record: dict) -> list[str]:
    user_msg = json.loads(gold_record["messages"][1]["content"])
    families = list(user_msg.get("optimizer_search_space", {}).keys())
    return families or list(WIDE_GRID_FAMILIES)


def normalize_pred_config(prediction: dict) -> dict | None:
    pred_config = prediction.get("config")
    if pred_config:
        if "warm_start_config" in pred_config:
            pred_config = pred_config["warm_start_config"]
        if "config" in pred_config and isinstance(pred_config["config"], dict):
            pred_config = pred_config["config"]
        if pred_config.get("optimizer_family") and pred_config.get("hyperparameters"):
            return pred_config

    raw = prediction.get("raw_response")
    if raw:
        try:
            parsed = extract_json_object(str(raw))
            if parsed.get("optimizer_family") and parsed.get("hyperparameters"):
                return parsed
        except Exception:
            pass
    return None


def create_optimizer_config(
    optimizer_family: str, hyperparameters: dict, seed: int, out_dir: Path
) -> tuple[str, str]:
    # eval{seed} avoids collisions across tasks (seed = task_index * 100 + candidate).
    config_name = f"custom_{optimizer_family}_eval{seed}"
    config_path = out_dir / f"{config_name}.json"
    payload = [hyperparameters, optimizer_family]
    with config_path.open("w") as f:
        json.dump({config_name: payload}, f)
    return config_name, str(config_path)


def find_optimizer_name_collisions(config_name: str, exclude: Path | None = None) -> list[str]:
    hits: list[str] = []
    exclude_resolved = exclude.resolve() if exclude else None
    for path in sorted(OPTIMIZER_CONFIGS_PATH.glob("*.json")):
        if exclude_resolved and path.resolve() == exclude_resolved:
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        if config_name in data:
            hits.append(str(path))
    return hits


def register_custom_optimizer(
    opt_path: str,
    opt_seed: int,
    *,
    debug: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(opt_path)
    config_name = next(iter(load_json(source)))
    target_opt_file = OPTIMIZER_CONFIGS_PATH / f"custom_opt_{opt_seed}.json"
    collisions = [
        p for p in find_optimizer_name_collisions(config_name, exclude=target_opt_file)
        if Path(p).resolve() != source.resolve()
    ]
    info: dict[str, Any] = {
        "config_name": config_name,
        "source": str(source),
        "target": str(target_opt_file),
        "collisions": collisions,
    }
    if collisions and not force:
        msg = (
            f"Optimizer name {config_name!r} already exists in: {collisions}. "
            "Remove stale files under task_set/optimizers/configs/ or pass "
            "--force-optimizer-register to overwrite custom_opt_{seed}.json only."
        )
        raise ValueError(msg)
    if collisions and force and debug:
        print(f"WARNING: registering despite name collisions in other files: {collisions}")
    shutil.copy(opt_path, target_opt_file)
    info["registered"] = True
    return info


def replica_debug_log_path(
    debug_dir: Path | None,
    task_name: str,
    optimizer_name: str,
    replica: int,
) -> Path | None:
    if debug_dir is None:
        return None
    safe = f"{task_name}__{optimizer_name}__r{replica}.log"
    return debug_dir / "train_inner" / safe


def list_replica_dir(replica_dir: Path) -> list[str]:
    if not replica_dir.exists():
        return []
    return sorted(str(p.relative_to(replica_dir)) for p in replica_dir.rglob("*") if p.is_file())


def write_debug_log(log_path: Path | None, header: str, body: str) -> str | None:
    if log_path is None:
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(header + "\n" + body)
    return str(log_path)


def _stderr_reader(
    proc: subprocess.Popen,
    stderr_lines: list[str],
    on_step: Callable[[int], None] | None,
) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        stderr_lines.append(line)
        if on_step:
            match = TRAIN_INNER_STEP_RE.search(line)
            if match:
                on_step(int(match.group(1)))


def run_train_inner(
    task_name: str,
    optimizer_name: str,
    output_dir: Path,
    training_steps: int,
    eval_every_n: int,
    replica: int,
    timeout: int,
    *,
    debug: bool = False,
    debug_dir: Path | None = None,
    show_progress: bool = True,
    progress: EvalProgress | None = None,
    task_index: int | None = None,
) -> dict[str, Any]:
    # train_inner runs with cwd=TASK_SET_ROOT; output_directory must be absolute.
    output_dir = Path(output_dir).resolve()
    replica_dir = output_dir / task_name / optimizer_name / str(training_steps) / str(replica)
    replica_dir.mkdir(parents=True, exist_ok=True)

    cached = try_load_cached_replica_result(replica_dir, replica, debug=debug)
    if cached is not None:
        if show_progress and tqdm is not None:
            print(f"[resume] {task_name}/{optimizer_name}/r{replica} (cached result)", file=sys.stderr)
        return cached

    py = TASKSET_EVAL_PYTHON
    cmd = (
        f"cd {TASK_SET_ROOT} && "
        f"{py} -m task_set.train_inner "
        f"--optimizer_name={optimizer_name} "
        f"--task_name={task_name} "
        f"--output_directory={output_dir} "
        f"--training_steps={training_steps} "
        f"--eval_every_n={eval_every_n} "
        f"--replica={replica}"
    )
    log_path = replica_debug_log_path(debug_dir, task_name, optimizer_name, replica)
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()

    env = os.environ.copy()
    env["TFDS_DATA_DIR"] = "/root/data/cys/tensorflow_datasets"
    env["PYTHONUNBUFFERED"] = "1"
    apply_train_inner_thread_env(env)

    run_label = f"{task_name}/{optimizer_name}/r{replica}"
    if debug:
        print(f"  [debug] train_inner replica={replica} timeout={timeout}s")
        print(f"  [debug] cmd: {cmd}")
        print(f"  [debug] replica_dir: {replica_dir}")

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(
            total=training_steps,
            desc=run_label,
            unit="step",
            file=sys.stderr,
            leave=False,
            dynamic_ncols=True,
        )

    def on_step(step: int) -> None:
        if pbar is not None:
            pbar.n = min(step, training_steps)
            pbar.refresh()
        if progress is not None:
            progress.set_active_run(
                task_index=task_index,
                task_name=task_name,
                optimizer_name=optimizer_name,
                replica=replica,
                step=step,
                training_steps=training_steps,
                label=run_label,
            )

    stderr_lines: list[str] = []
    stdout_data = ""
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        reader = threading.Thread(
            target=_stderr_reader,
            args=(proc, stderr_lines, on_step),
            daemon=True,
        )
        stdout_chunks: list[str] = []

        def _stdout_reader() -> None:
            if proc.stdout is None:
                return
            stdout_chunks.append(proc.stdout.read() or "")

        out_reader = threading.Thread(target=_stdout_reader, daemon=True)
        reader.start()
        out_reader.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
        finally:
            reader.join(timeout=5)
            out_reader.join(timeout=5)
        stdout_data = "".join(stdout_chunks)
        result_returncode = proc.returncode
        stderr_data = "".join(stderr_lines)

        elapsed_s = round(time.monotonic() - t0, 2)
        if pbar is not None:
            pbar.n = training_steps
            pbar.close()

        result_file = replica_dir / "result"
        dir_files = list_replica_dir(replica_dir)
        log_header = (
            f"started_at={started_at}\n"
            f"elapsed_s={elapsed_s}\n"
            f"returncode={result_returncode}\n"
            f"timeout_s={timeout}\n"
            f"cmd={cmd}\n"
            f"replica_dir={replica_dir}\n"
            f"result_exists={result_file.exists()}\n"
            f"files_in_replica_dir={dir_files}\n"
            "--- stdout ---"
        )
        log_file = write_debug_log(
            log_path,
            log_header,
            (stdout_data or "") + "\n--- stderr ---\n" + stderr_data,
        )

        if not result_file.exists():
            failure: dict[str, Any] = {
                "replica": replica,
                "status": "failed",
                "error": "No result file",
                "returncode": result_returncode,
                "elapsed_s": elapsed_s,
                "timeout_s": timeout,
                "replica_dir": str(replica_dir),
                "files_in_replica_dir": dir_files,
                "debug_log": log_file,
                "stderr_tail": stderr_data[-2000:],
                "stdout_tail": (stdout_data or "")[-2000:],
            }
            if result_returncode != 0:
                failure["error"] = f"train_inner exit code {result_returncode}"
            return failure
        curves = load_json(result_file)
        ok: dict[str, Any] = {
            "replica": replica,
            "status": "ok",
            "curves": curves,
            "result_dir": str(replica_dir),
            "elapsed_s": elapsed_s,
            "debug_log": log_file,
        }
        if debug:
            print(f"  [debug] replica={replica} ok in {elapsed_s}s -> {result_file}")
        return ok
    except subprocess.TimeoutExpired:
        elapsed_s = round(time.monotonic() - t0, 2)
        if pbar is not None:
            pbar.close()
        dir_files = list_replica_dir(replica_dir)
        stderr_data = "".join(stderr_lines)
        log_file = write_debug_log(
            log_path,
            (
                f"started_at={started_at}\n"
                f"elapsed_s={elapsed_s}\n"
                f"status=timeout\n"
                f"timeout_s={timeout}\n"
                f"cmd={cmd}\n"
                f"files_in_replica_dir={dir_files}\n"
                "--- output ---\n"
                f"{stdout_data}\n--- stderr ---\n{stderr_data}"
            ),
            "",
        )
        return {
            "replica": replica,
            "status": "timeout",
            "elapsed_s": elapsed_s,
            "timeout_s": timeout,
            "replica_dir": str(replica_dir),
            "files_in_replica_dir": dir_files,
            "debug_log": log_file,
        }
    except Exception as exc:
        elapsed_s = round(time.monotonic() - t0, 2)
        if pbar is not None:
            pbar.close()
        return {
            "replica": replica,
            "status": "error",
            "error": str(exc),
            "elapsed_s": elapsed_s,
            "debug_log": str(log_path) if log_path else None,
        }


def _run_train_inner_replica(
    task_name: str,
    optimizer_name: str,
    output_dir: Path,
    training_steps: int,
    eval_every_n: int,
    replica: int,
    replica_timeout: int,
    train_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Pool worker: keyword-only train_inner args must not be passed positionally."""
    return run_train_inner(
        task_name,
        optimizer_name,
        output_dir,
        training_steps,
        eval_every_n,
        replica,
        replica_timeout,
        **train_kwargs,
    )


def run_replicas_parallel(
    task_name: str,
    optimizer_name: str,
    output_dir: Path,
    training_steps: int,
    eval_every_n: int,
    num_replicas: int,
    warm_timeout: int,
    replica_timeout: int,
    *,
    debug: bool = False,
    debug_dir: Path | None = None,
    progress: EvalProgress | None = None,
    task_index: int | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    train_kwargs = {
        "debug": debug,
        "debug_dir": debug_dir,
        "progress": progress,
        "task_index": task_index,
        "show_progress": show_progress,
    }
    warm = run_train_inner(
        task_name,
        optimizer_name,
        output_dir,
        training_steps,
        eval_every_n,
        0,
        warm_timeout,
        **train_kwargs,
    )
    remaining = list(range(1, num_replicas))
    if remaining:
        with multiprocessing.Pool(processes=len(remaining)) as pool:
            parallel = pool.starmap(
                _run_train_inner_replica,
                [
                    (
                        task_name,
                        optimizer_name,
                        output_dir,
                        training_steps,
                        eval_every_n,
                        replica,
                        replica_timeout,
                        {**train_kwargs, "show_progress": False},
                    )
                    for replica in remaining
                ],
            )
    else:
        parallel = []
    return [warm] + parallel


def build_task_baseline(
    task_name: str,
    families: list[str],
    baseline_output_dir: Path,
    training_steps: int,
    eval_every_n: int,
    num_replicas: int,
    max_per_family: int | None,
    warm_timeout: int,
    replica_timeout: int,
    *,
    debug: bool = False,
    debug_dir: Path | None = None,
    progress: EvalProgress | None = None,
    task_index: int | None = None,
) -> dict[str, Any]:
    optimizer_names = list_grid_optimizer_names(OPTIMIZER_CONFIGS_PATH, families, max_per_family)
    if not optimizer_names:
        raise ValueError(f"No grid optimizers found for families={families}")

    print(
        f"  Building baseline for {task_name}: "
        f"{len(optimizer_names)} grid optimizers x {num_replicas} replicas"
    )
    global_best = float("inf")
    per_optimizer_min: dict[str, float] = {}

    grid_iter: Any = optimizer_names
    if tqdm is not None:
        grid_iter = tqdm(
            optimizer_names,
            desc=f"baseline {task_name}",
            unit="opt",
            file=sys.stderr,
            leave=False,
        )

    for i, optimizer_name in enumerate(grid_iter):
        if progress is not None:
            progress.set_active_run(
                task_index=task_index,
                task_name=task_name,
                optimizer_name=optimizer_name,
                label=f"baseline {task_name} {i + 1}/{len(optimizer_names)}",
            )
        replica_results = run_replicas_parallel(
            task_name,
            optimizer_name,
            baseline_output_dir,
            training_steps,
            eval_every_n,
            num_replicas,
            warm_timeout,
            replica_timeout,
            debug=debug,
            debug_dir=debug_dir,
            progress=progress,
            task_index=task_index,
            show_progress=False,
        )
        opt_min = float("inf")
        for rep in replica_results:
            if rep.get("status") != "ok":
                continue
            opt_min = min(opt_min, min_valid_outer(rep["curves"]))
        if opt_min < float("inf"):
            per_optimizer_min[optimizer_name] = opt_min
            global_best = min(global_best, opt_min)
        if (i + 1) % 25 == 0 or i + 1 == len(optimizer_names):
            print(f"    grid progress {i + 1}/{len(optimizer_names)}, best_val={global_best}")

    if global_best == float("inf"):
        raise RuntimeError(f"Failed to compute baseline for {task_name}")

    return {
        "best_val": global_best,
        "search_space_families": families,
        "num_grid_optimizers": len(optimizer_names),
        "num_replicas": num_replicas,
        "per_optimizer_min_valid_outer": per_optimizer_min,
    }


def evaluate_prediction(
    task_index: int,
    prediction: dict,
    gold_record: dict,
    custom_config_dir: Path,
    train_output_dir: Path,
    baseline_cache: dict[str, dict[str, Any]],
    training_steps: int,
    eval_every_n: int,
    num_replicas: int,
    opt_seed: int,
    warm_timeout: int,
    replica_timeout: int,
    *,
    debug: bool = False,
    debug_dir: Path | None = None,
    force_optimizer_register: bool = False,
    progress: EvalProgress | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    pred_config = normalize_pred_config(prediction)
    if not pred_config:
        return {"index": task_index, "status": "no_prediction", "sample_id": prediction.get("sample_id")}

    optimizer_family = pred_config.get("optimizer_family")
    hyperparameters = pred_config.get("hyperparameters", {})
    if not optimizer_family or not hyperparameters:
        return {
            "index": task_index,
            "status": "invalid_config",
            "sample_id": prediction.get("sample_id"),
        }

    user_msg = json.loads(gold_record["messages"][1]["content"])
    task_family = user_msg["task_family"]
    target_task = user_msg["task"]
    task_name = find_task_seed(task_family, target_task)
    if not task_name:
        return {
            "index": task_index,
            "status": "task_not_found",
            "task_family": task_family,
            "sample_id": prediction.get("sample_id"),
        }

    best_val = get_task_best_val(task_name, baseline_cache)
    if best_val is None:
        return {
            "index": task_index,
            "status": "missing_baseline",
            "task_name": task_name,
            "sample_id": prediction.get("sample_id"),
        }

    opt_name, opt_path = create_optimizer_config(
        optimizer_family, hyperparameters, opt_seed, custom_config_dir
    )
    try:
        reg_info = register_custom_optimizer(
            opt_path,
            opt_seed,
            debug=debug,
            force=force_optimizer_register,
        )
    except ValueError as exc:
        return {
            "index": task_index,
            "status": "optimizer_register_failed",
            "task_name": task_name,
            "optimizer_family": optimizer_family,
            "opt_name": opt_name,
            "error": str(exc),
            "sample_id": prediction.get("sample_id"),
        }

    print(f"[Task {task_index} sample {prediction.get('sample_id', 0)}] {task_name} / {opt_name}")
    if debug:
        print(f"  [debug] optimizer registered: {reg_info}")

    replica_results = run_replicas_parallel(
        task_name,
        opt_name,
        train_output_dir,
        training_steps,
        eval_every_n,
        num_replicas,
        warm_timeout,
        replica_timeout,
        debug=debug,
        debug_dir=debug_dir,
        progress=progress,
        task_index=task_index,
        show_progress=show_progress,
    )

    scored_replicas: list[dict[str, float]] = []
    replica_details: list[dict[str, Any]] = []
    for rep in replica_results:
        if rep.get("status") != "ok":
            replica_details.append(rep)
            continue
        scores = score_replica_curve(rep["curves"], best_val)
        scored_replicas.append(scores)
        replica_details.append({"replica": rep["replica"], "status": "ok", **scores})

    if not scored_replicas:
        return {
            "index": task_index,
            "status": "all_failed",
            "task_name": task_name,
            "optimizer_family": optimizer_family,
            "opt_name": opt_name,
            "replica_details": replica_details,
        }

    agg = aggregate_replica_scores(scored_replicas)
    return {
        "index": task_index,
        "sample_id": prediction.get("sample_id"),
        "task_name": task_name,
        "optimizer_family": optimizer_family,
        "opt_name": opt_name,
        "status": "ok",
        "metric": "mean_normalized_valid_outer",
        "best_val_reference": best_val,
        "mean_normalized_valid_outer": agg["mean_normalized_valid_outer"],
        "mean_final_test_loss": agg["final_test_loss"],
        "mean_init_valid_outer": agg["init_valid_outer"],
        "replicas_ok": len(scored_replicas),
        "replicas_total": num_replicas,
        "replica_details": replica_details,
    }


def group_predictions(predictions: list[dict], model_label: str | None) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for pred in predictions:
        if model_label and str(pred.get("model")) != model_label:
            continue
        idx = int(pred["index"])
        grouped.setdefault(idx, []).append(pred)
    for idx in grouped:
        grouped[idx].sort(key=lambda row: int(row.get("sample_id", 0)))
    return grouped


def dedupe_predictions(preds: list[dict]) -> list[dict]:
    """Keep first prediction per unique (optimizer_family, hyperparameters) config."""
    unique: list[dict] = []
    seen: set[str] = set()
    for pred in preds:
        cfg = normalize_pred_config(pred)
        if cfg is None:
            unique.append(pred)
            continue
        key = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
        if key not in seen:
            seen.add(key)
            unique.append(pred)
    return unique


def summarize_prediction_dedupe(grouped: dict[int, list[dict]]) -> dict[str, int]:
    """Count raw vs deduped prediction rows (e.g. 3 generation runs merged per task)."""
    raw_total = 0
    unique_total = 0
    tasks_multi_raw = 0
    tasks_multi_unique = 0
    duplicate_rows = 0
    for preds in grouped.values():
        raw_n = len(preds)
        unique_n = len(dedupe_predictions(preds))
        raw_total += raw_n
        unique_total += unique_n
        duplicate_rows += raw_n - unique_n
        if raw_n > 1:
            tasks_multi_raw += 1
        if unique_n > 1:
            tasks_multi_unique += 1
    return {
        "tasks_with_predictions": len(grouped),
        "prediction_rows_raw": raw_total,
        "prediction_rows_unique": unique_total,
        "duplicate_rows_skipped": duplicate_rows,
        "tasks_with_multiple_raw_predictions": tasks_multi_raw,
        "tasks_with_multiple_unique_candidates": tasks_multi_unique,
        "candidate_eval_runs": unique_total,
    }


def dry_run_parse(
    predictions: list[dict],
    gold_records: list[dict],
    model_label: str | None,
    baseline_cache: dict[str, dict[str, Any]],
) -> dict:
    grouped = group_predictions(predictions, model_label)
    stats = {
        "tasks_total": len(grouped),
        "parse_ok": 0,
        "no_prediction": 0,
        "invalid_config": 0,
        "task_not_found": 0,
        "task_resolved": 0,
        "missing_baseline": 0,
        "eval_ready": 0,
        "by_optimizer_family": {},
    }
    issues: list[dict] = []
    tasks: list[dict] = []

    for task_index in sorted(grouped.keys()):
        if task_index >= len(gold_records):
            continue
        gold_record = gold_records[task_index]
        pred = dedupe_predictions(grouped[task_index])[0]
        pred_config = normalize_pred_config(pred)
        user_msg = json.loads(gold_record["messages"][1]["content"])
        dataset = user_msg.get("task", {}).get("dataset", [None])[0]
        task_info: dict[str, Any] = {
            "index": task_index,
            "task_family": user_msg.get("task_family"),
            "dataset": dataset,
        }
        if not pred_config:
            stats["no_prediction"] += 1
            task_info["status"] = "no_prediction"
            issues.append(task_info)
            tasks.append(task_info)
            continue
        if not pred_config.get("optimizer_family") or not pred_config.get("hyperparameters"):
            stats["invalid_config"] += 1
            task_info["status"] = "invalid_config"
            issues.append(task_info)
            tasks.append(task_info)
            continue
        stats["parse_ok"] += 1
        task_info["optimizer_family"] = pred_config["optimizer_family"]
        task_name = find_task_seed(user_msg["task_family"], user_msg["task"])
        if not task_name:
            stats["task_not_found"] += 1
            task_info["status"] = "task_not_found"
            issues.append(task_info)
            tasks.append(task_info)
            continue
        stats["task_resolved"] += 1
        task_info["task_name"] = task_name
        if get_task_best_val(task_name, baseline_cache) is None:
            stats["missing_baseline"] += 1
            task_info["status"] = "missing_baseline"
            issues.append(task_info)
            tasks.append(task_info)
            continue
        stats["eval_ready"] += 1
        fam = pred_config["optimizer_family"]
        stats["by_optimizer_family"][fam] = stats["by_optimizer_family"].get(fam, 0) + 1
        task_info["status"] = "eval_ready"
        tasks.append(task_info)

    return {"summary": stats, "tasks": tasks, "issues": issues}


def collect_tasks_for_baselines(
    gold_records: list[dict],
    grouped: dict[int, list[dict]] | None,
) -> list[tuple[int, str, list[str]]]:
    tasks: list[tuple[int, str, list[str]]] = []
    indices = sorted(grouped.keys()) if grouped else range(len(gold_records))
    for task_index in indices:
        if task_index >= len(gold_records):
            continue
        gold_record = gold_records[task_index]
        user_msg = json.loads(gold_record["messages"][1]["content"])
        task_name = find_task_seed(user_msg["task_family"], user_msg["task"])
        if not task_name:
            continue
        tasks.append((task_index, task_name, search_space_families(gold_record)))
    return tasks


def evaluate_task_index(
    task_index: int,
    gold_record: dict,
    candidates: list[dict],
    custom_config_dir: Path,
    train_output_dir: Path,
    baseline_cache: dict[str, dict[str, Any]],
    training_steps: int,
    eval_every_n: int,
    num_replicas: int,
    warm_timeout: int,
    replica_timeout: int,
    *,
    debug: bool = False,
    debug_dir: Path | None = None,
    force_optimizer_register: bool = False,
    progress: EvalProgress | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    best_result: dict | None = None
    best_cost: float | None = None
    candidate_results: list[dict] = []

    for pred_i, prediction in enumerate(candidates):
        result = evaluate_prediction(
            task_index,
            prediction,
            gold_record,
            custom_config_dir,
            train_output_dir,
            baseline_cache,
            training_steps,
            eval_every_n,
            num_replicas,
            opt_seed=task_index * 100 + pred_i,
            warm_timeout=warm_timeout,
            replica_timeout=replica_timeout,
            debug=debug,
            debug_dir=debug_dir,
            force_optimizer_register=force_optimizer_register,
            progress=progress,
            show_progress=show_progress,
        )
        candidate_results.append(result)
        cost = result.get("mean_normalized_valid_outer")
        if cost is not None and (best_cost is None or cost < best_cost):
            best_cost = cost
            best_result = result

    if best_result is None:
        best_result = candidate_results[0] if candidate_results else {
            "index": task_index,
            "status": "no_prediction",
        }
    best_result = dict(best_result)
    best_result["num_candidates"] = len(candidates)
    best_result["candidate_results"] = [
        {
            "sample_id": r.get("sample_id"),
            "status": r.get("status"),
            "mean_normalized_valid_outer": r.get("mean_normalized_valid_outer"),
            "mean_final_test_loss": r.get("mean_final_test_loss"),
        }
        for r in candidate_results
    ]
    return best_result


def _baseline_worker(payload: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
    task_index = payload["task_index"]
    task_name = payload["task_name"]
    families = payload["families"]
    baseline = build_task_baseline(
        task_name,
        families,
        Path(payload["baseline_dir"]),
        payload["training_steps"],
        payload["eval_every_n"],
        payload["num_replicas"],
        payload["max_per_family"],
        payload["warm_timeout"],
        payload["replica_timeout"],
        debug=payload["debug"],
        debug_dir=Path(payload["debug_dir"]) if payload.get("debug_dir") else None,
        progress=None,
        task_index=task_index,
    )
    return task_index, task_name, baseline


def _eval_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    task_index = payload["task_index"]
    result = evaluate_task_index(
        task_index,
        payload["gold_record"],
        payload["candidates"],
        Path(payload["custom_config_dir"]),
        Path(payload["train_output_dir"]),
        payload["baseline_cache"],
        payload["training_steps"],
        payload["eval_every_n"],
        payload["num_replicas"],
        payload["warm_timeout"],
        payload["replica_timeout"],
        debug=payload["debug"],
        debug_dir=Path(payload["debug_dir"]) if payload.get("debug_dir") else None,
        force_optimizer_register=payload["force_optimizer_register"],
        progress=None,
        show_progress=False,
    )
    return result


def build_baselines_for_tasks(
    pending: list[tuple[int, str, list[str]]],
    baseline_cache: dict[str, dict[str, Any]],
    baseline_dir: Path,
    training_steps: int,
    eval_every_n: int,
    num_replicas: int,
    max_per_family: int | None,
    warm_timeout: int,
    replica_timeout: int,
    baseline_cache_path: Path | None,
    *,
    task_workers: int,
    debug: bool = False,
    debug_dir: Path | None = None,
    progress: EvalProgress | None = None,
    smoke_test: bool = False,
) -> None:
    if not pending:
        return
    progress.start_phase("baseline_build", total=len(pending), task_workers=task_workers)
    worker_payload = {
        "baseline_dir": str(baseline_dir.resolve()),
        "training_steps": training_steps,
        "eval_every_n": eval_every_n,
        "num_replicas": num_replicas,
        "max_per_family": max_per_family,
        "warm_timeout": warm_timeout,
        "replica_timeout": replica_timeout,
        "debug": debug,
        "debug_dir": str(debug_dir) if debug_dir else None,
    }

    if task_workers <= 1:
        task_iter: Any = pending
        if tqdm is not None:
            task_iter = tqdm(pending, desc="baseline tasks", unit="task", file=sys.stderr)
        for task_index, task_name, families in task_iter:
            print(f"[Task {task_index}] {task_name}")
            baseline_cache[task_name] = build_task_baseline(
                task_name,
                families,
                baseline_dir,
                training_steps,
                eval_every_n,
                num_replicas,
                max_per_family,
                warm_timeout,
                replica_timeout,
                debug=debug,
                debug_dir=debug_dir,
                progress=progress,
                task_index=task_index,
            )
            if baseline_cache_path:
                save_baseline_cache(
                    baseline_cache_path,
                    baseline_cache,
                    meta={**paper_eval_protocol(), "smoke_test": smoke_test},
                )
            if progress:
                progress.task_finished(task_index, "baseline_ok", task_name=task_name)
        return

    with ProcessPoolExecutor(max_workers=task_workers) as pool:
        futures = {
            pool.submit(
                _baseline_worker,
                {
                    **worker_payload,
                    "task_index": task_index,
                    "task_name": task_name,
                    "families": families,
                },
            ): (task_index, task_name)
            for task_index, task_name, families in pending
        }
        task_bar = tqdm(total=len(futures), desc="baseline tasks", unit="task", file=sys.stderr) if tqdm else None
        for fut in as_completed(futures):
            task_index, task_name, baseline = fut.result()
            baseline_cache[task_name] = baseline
            if baseline_cache_path:
                save_baseline_cache(
                    baseline_cache_path,
                    baseline_cache,
                    meta={**paper_eval_protocol(), "smoke_test": smoke_test},
                )
            if progress:
                progress.task_finished(task_index, "baseline_ok", task_name=task_name)
            if task_bar:
                task_bar.update(1)
        if task_bar:
            task_bar.close()


def resolve_eval_protocol(args: argparse.Namespace) -> tuple[int, int, int]:
    """Return (training_steps, eval_every_n, num_replicas) for this run."""
    if args.smoke_test:
        steps = args.smoke_training_steps
        eval_every = args.smoke_eval_every_n
        replicas = args.smoke_num_replicas
        print(
            "=== SMOKE TEST protocol (non-paper; debugging only) ==="
            f"\n  training_steps={steps}, eval_every_n={eval_every}, num_replicas={replicas}"
        )
        return steps, eval_every, replicas

    steps = PAPER_TRAINING_STEPS
    eval_every = PAPER_EVAL_EVERY_N
    replicas = PAPER_NUM_REPLICAS
    ignored = []
    if args.training_steps is not None and args.training_steps != PAPER_TRAINING_STEPS:
        ignored.append(f"--training-steps={args.training_steps}")
    if args.eval_every_n is not None and args.eval_every_n != PAPER_EVAL_EVERY_N:
        ignored.append(f"--eval-every-n={args.eval_every_n}")
    if args.num_replicas is not None:
        replicas = args.num_replicas
        if replicas != PAPER_NUM_REPLICAS:
            print(
                f"NOTE: Using --num-replicas={replicas} "
                f"(paper default is {PAPER_NUM_REPLICAS}; mean aggregation unchanged)."
            )
    if ignored:
        print(
            "WARNING: Ignoring non-paper eval overrides: "
            + ", ".join(ignored)
            + f". Using {steps} steps, eval every {eval_every}."
        )

    return steps, eval_every, replicas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-exact TaskSet DL evaluation.")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--model-label", default=None)
    parser.add_argument(
        "--training-steps",
        type=int,
        default=None,
        help=f"Ignored unless --smoke-test; production uses {PAPER_TRAINING_STEPS}.",
    )
    parser.add_argument(
        "--eval-every-n",
        type=int,
        default=None,
        help=f"Ignored unless --smoke-test; production uses {PAPER_EVAL_EVERY_N}.",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help=(
            f"Random train_inner initializations per candidate config "
            f"(default {PAPER_NUM_REPLICAS}; use 3 for reduced CPU eval)."
        ),
    )
    parser.add_argument(
        "--cpu-profile",
        action="store_true",
        help=(
            f"Tune parallelism for ~96-core CPU hosts "
            f"(task workers={CPU_TASK_WORKERS}, baseline workers={CPU_BASELINE_TASK_WORKERS})."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Debug-only reduced protocol; production eval always uses paper §3.3 constants.",
    )
    parser.add_argument("--smoke-training-steps", type=int, default=400)
    parser.add_argument("--smoke-eval-every-n", type=int, default=200)
    parser.add_argument("--smoke-num-replicas", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/task_set_dl_eval_opt"))
    parser.add_argument("--baseline-cache", type=Path, default=None)
    parser.add_argument(
        "--baseline-max-per-family",
        type=int,
        default=None,
        help="Optional cap when building baselines (paper uses all 1000 per family).",
    )
    parser.add_argument(
        "--build-baselines-only",
        action="store_true",
        help="Only compute per-task grid best_val baselines; no model eval.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task-indices", type=int, nargs="*")
    parser.add_argument("--warm-timeout", type=int, default=3600)
    parser.add_argument("--replica-timeout", type=int, default=2400)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging; save full train_inner stdout/stderr per replica under work-dir/debug/.",
    )
    parser.add_argument(
        "--skip-baseline-build",
        action="store_true",
        help="Do not auto-build missing baselines; tasks without baseline cache entries fail fast.",
    )
    parser.add_argument(
        "--baseline-task-workers",
        type=int,
        default=None,
        help=f"Parallel tasks during baseline grid build (default {PAPER_BASELINE_TASK_WORKERS}).",
    )
    parser.add_argument(
        "--task-workers",
        type=int,
        default=None,
        help=f"Parallel tasks during model eval (default {PAPER_TASK_WORKERS}).",
    )
    parser.add_argument(
        "--force-optimizer-register",
        action="store_true",
        help="Overwrite custom_opt_{seed}.json even if the optimizer name exists in other config files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reuse cached train_inner result files and eval_checkpoint.json (default: on).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore cached train_inner results and task checkpoints.",
    )
    return parser.parse_args()


def write_debug_manifest(work_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_path = work_dir / "debug" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest_path, payload)
    return manifest_path


def main() -> None:
    args = parse_args()
    training_steps, eval_every_n, num_replicas = resolve_eval_protocol(args)
    if not args.test_data:
        raise SystemExit("--test-data is required")

    gold_records = load_jsonl(args.test_data)
    baseline_cache = load_baseline_cache(args.baseline_cache)
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = work_dir / "debug" if args.debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug artifacts: {debug_dir}")
    baseline_dir = work_dir / "baseline_runs"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    progress = EvalProgress(work_dir / "progress.json")
    run_started_at = datetime.now(timezone.utc).isoformat()
    default_task_workers = CPU_TASK_WORKERS if args.cpu_profile else PAPER_TASK_WORKERS
    default_baseline_workers = (
        CPU_BASELINE_TASK_WORKERS if args.cpu_profile else PAPER_BASELINE_TASK_WORKERS
    )
    task_workers = args.task_workers if args.task_workers is not None else default_task_workers
    baseline_task_workers = (
        args.baseline_task_workers
        if args.baseline_task_workers is not None
        else default_baseline_workers
    )

    predictions: list[dict] = []
    grouped: dict[int, list[dict]] | None = None
    if args.predictions:
        predictions = load_jsonl(args.predictions)
        grouped = group_predictions(predictions, args.model_label)
        if args.task_indices:
            grouped = {idx: grouped[idx] for idx in args.task_indices if idx in grouped}

    if args.dry_run:
        if not args.output_json:
            raise SystemExit("--output-json is required for --dry-run")
        preds_for_dry = predictions
        if args.task_indices:
            allowed = set(args.task_indices)
            preds_for_dry = [p for p in predictions if p.get("index") in allowed]
        report = dry_run_parse(preds_for_dry, gold_records, args.model_label, baseline_cache)
        if args.task_indices:
            report["task_indices"] = args.task_indices
        save_json(args.output_json, report)
        print("=== DRY RUN ===")
        print(json.dumps(report["summary"], indent=2))
        return

    if args.build_baselines_only:
        tasks = collect_tasks_for_baselines(gold_records, grouped)
        pending = [
            (task_index, task_name, families)
            for task_index, task_name, families in tasks
            if task_name not in baseline_cache
        ]
        print(
            f"=== Building baselines for {len(pending)} tasks "
            f"({len(tasks) - len(pending)} cached, {baseline_task_workers} workers) ==="
        )
        build_baselines_for_tasks(
            pending,
            baseline_cache,
            baseline_dir,
            training_steps,
            eval_every_n,
            num_replicas,
            args.baseline_max_per_family,
            args.warm_timeout,
            args.replica_timeout,
            args.baseline_cache,
            task_workers=baseline_task_workers,
            debug=args.debug,
            debug_dir=debug_dir,
            progress=progress,
            smoke_test=args.smoke_test,
        )
        out_path = args.output_json or args.baseline_cache or (work_dir / "task_baselines.json")
        save_baseline_cache(
            out_path,
            baseline_cache,
            meta={**paper_eval_protocol(), "smoke_test": args.smoke_test},
        )
        print(f"Baselines saved to {out_path}")
        print(f"Progress log: {progress.path}")
        return

    if not args.predictions or not args.output_json:
        raise SystemExit("--predictions and --output-json are required for evaluation")

    if grouped is None:
        grouped = group_predictions(predictions, args.model_label)

    # Build any missing per-task baselines before scoring predictions.
    baseline_pending: list[tuple[int, str, list[str]]] = []
    for task_index, task_name, families in collect_tasks_for_baselines(gold_records, grouped):
        if get_task_best_val(task_name, baseline_cache) is not None:
            continue
        if args.skip_baseline_build:
            print(
                f"[Task {task_index}] skip baseline build for {task_name} "
                "(--skip-baseline-build; task will fail without cached best_val)"
            )
            continue
        baseline_pending.append((task_index, task_name, families))
    if baseline_pending:
        print(f"Building baselines for {len(baseline_pending)} tasks before model eval...")
        build_baselines_for_tasks(
            baseline_pending,
            baseline_cache,
            baseline_dir,
            training_steps,
            eval_every_n,
            num_replicas,
            args.baseline_max_per_family,
            args.warm_timeout,
            args.replica_timeout,
            args.baseline_cache,
            task_workers=baseline_task_workers,
            debug=args.debug,
            debug_dir=debug_dir,
            progress=progress,
            smoke_test=args.smoke_test,
        )

    custom_config_dir = work_dir / "custom_optimizers"
    custom_config_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir = work_dir / "train_outputs"
    train_output_dir.mkdir(parents=True, exist_ok=True)

    task_indices = sorted(grouped.keys())
    dedupe_summary = summarize_prediction_dedupe(grouped)
    print("=== TaskSet eval (paper §3.3) ===")
    print(f"task_set root: {TASK_SET_ROOT}")
    print(f"Model label: {args.model_label or 'ALL'}")
    print(f"Tasks: {len(task_indices)} | Replicas: {num_replicas} | Task workers: {task_workers}")
    print(
        "Predictions: "
        f"{dedupe_summary['prediction_rows_raw']} raw rows -> "
        f"{dedupe_summary['prediction_rows_unique']} unique configs "
        f"({dedupe_summary['duplicate_rows_skipped']} duplicate rows skipped); "
        f"{dedupe_summary['tasks_with_multiple_unique_candidates']} tasks have >1 unique candidate"
    )
    print(
        f"Training: {training_steps} steps, eval every {eval_every_n}, "
        "50 minibatches/eval (train_inner default)"
    )
    print(
        "Metric: mean normalized VALID_OUTER loss (lower is better); "
        f"replica aggregation={PAPER_REPLICA_AGGREGATION}"
    )
    print(f"Work dir: {work_dir}")
    print(f"Progress: {progress.path}")
    if args.resume:
        print("Resume: enabled (cached train_inner results + eval_checkpoint.json)")
    print()

    checkpoint_path = eval_checkpoint_path(work_dir)
    all_results: list[dict] = []
    done_indices: set[int] = set()
    if args.resume:
        all_results = load_eval_checkpoint(checkpoint_path)
        done_indices = {
            int(r["index"]) for r in all_results if r.get("index") is not None
        }
        if all_results:
            print(
                f"Resume: loaded {len(all_results)} finished task(s) from {checkpoint_path.name}"
            )
        cached_replicas = len(list(train_output_dir.glob("*/*/*/*/result")))
        if cached_replicas:
            print(f"Resume: {cached_replicas} cached train_inner result file(s) on disk")

    pending_indices = [idx for idx in task_indices if idx not in done_indices]
    if not pending_indices and all_results:
        print("Resume: all tasks already in checkpoint — writing final output only")
    elif len(pending_indices) < len(task_indices):
        print(f"Resume: {len(pending_indices)} task(s) remaining")

    progress.start_phase("model_eval", total=len(task_indices), task_workers=task_workers)
    if all_results:
        progress.restore_model_eval(all_results, len(task_indices), task_workers=task_workers)

    def checkpoint_results(results: list[dict]) -> None:
        if args.resume:
            save_eval_checkpoint(checkpoint_path, results, model_label=args.model_label)

    eval_common = {
        "custom_config_dir": str(custom_config_dir.resolve()),
        "train_output_dir": str(train_output_dir.resolve()),
        "baseline_cache": baseline_cache,
        "training_steps": training_steps,
        "eval_every_n": eval_every_n,
        "num_replicas": num_replicas,
        "warm_timeout": args.warm_timeout,
        "replica_timeout": args.replica_timeout,
        "debug": args.debug,
        "debug_dir": str(debug_dir) if debug_dir else None,
        "force_optimizer_register": args.force_optimizer_register,
    }

    if task_workers <= 1:
        task_iter: Any = pending_indices
        if tqdm is not None:
            task_iter = tqdm(pending_indices, desc="eval tasks", unit="task", file=sys.stderr)
        for task_index in task_iter:
            if task_index >= len(gold_records):
                continue
            gold_record = gold_records[task_index]
            candidates = dedupe_predictions(grouped[task_index])
            best_result = evaluate_task_index(
                task_index,
                gold_record,
                candidates,
                custom_config_dir,
                train_output_dir,
                baseline_cache,
                training_steps,
                eval_every_n,
                num_replicas,
                args.warm_timeout,
                args.replica_timeout,
                debug=args.debug,
                debug_dir=debug_dir,
                force_optimizer_register=args.force_optimizer_register,
                progress=progress,
                show_progress=True,
            )
            all_results.append(best_result)
            progress.task_finished(task_index, best_result.get("status", "unknown"))
            checkpoint_results(all_results)
            print(
                f"[Task {task_index}] best mean_normalized_valid_outer: "
                f"{best_result.get('mean_normalized_valid_outer')}"
            )
    else:
        payloads = []
        for task_index in pending_indices:
            if task_index >= len(gold_records):
                continue
            payloads.append(
                {
                    **eval_common,
                    "task_index": task_index,
                    "gold_record": gold_records[task_index],
                    "candidates": dedupe_predictions(grouped[task_index]),
                }
            )
        with ProcessPoolExecutor(max_workers=task_workers) as pool:
            futures = {pool.submit(_eval_task_worker, p): p["task_index"] for p in payloads}
            task_bar = tqdm(total=len(futures), desc="eval tasks", unit="task", file=sys.stderr) if tqdm else None
            for fut in as_completed(futures):
                task_index = futures[fut]
                best_result = fut.result()
                all_results.append(best_result)
                progress.task_finished(task_index, best_result.get("status", "unknown"))
                checkpoint_results(all_results)
                if task_bar:
                    task_bar.update(1)
                print(
                    f"[Task {task_index}] best mean_normalized_valid_outer: "
                    f"{best_result.get('mean_normalized_valid_outer')}"
                )
            if task_bar:
                task_bar.close()
        all_results.sort(key=lambda r: r.get("index", -1))

    ok_results = [r for r in all_results if r.get("status") == "ok"]
    summary = {
        "model": args.model_label,
        "tasks_total": len(grouped),
        "tasks_ok": len(ok_results),
        "metric": "mean_normalized_valid_outer",
        **paper_eval_protocol(),
        "num_replicas": num_replicas,
        "training_steps": training_steps,
        "eval_every_n": eval_every_n,
        "prediction_dedupe": dedupe_summary,
        "smoke_test": args.smoke_test,
        "cpu_profile": args.cpu_profile,
        "task_workers": task_workers,
        "progress_json": str(progress.path),
    }
    if ok_results:
        costs = [r["mean_normalized_valid_outer"] for r in ok_results]
        tests = [r["mean_final_test_loss"] for r in ok_results if r.get("mean_final_test_loss") is not None]
        summary.update(
            {
                "mean_of_task_costs": sum(costs) / len(costs),
                "best_task_cost": min(costs),
                "worst_task_cost": max(costs),
            }
        )
        if tests:
            summary["mean_final_test_loss"] = sum(tests) / len(tests)

    payload = {"summary": summary, "results": all_results}
    save_json(args.output_json, payload)
    print(f"\nResults saved to: {args.output_json}")
    print(f"Progress log: {progress.path}")

    if args.debug:
        failed = [
            r for r in all_results
            if r.get("status") not in ("ok",) or any(
                c.get("status") not in ("ok",) for c in r.get("candidate_results", [])
            )
        ]
        manifest = write_debug_manifest(
            work_dir,
            {
                "started_at": run_started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "work_dir": str(work_dir),
                "debug_dir": str(debug_dir) if debug_dir else None,
                "output_json": str(args.output_json),
                "protocol": {
                    "training_steps": training_steps,
                    "eval_every_n": eval_every_n,
                    "num_replicas": num_replicas,
                    "smoke_test": args.smoke_test,
                    "skip_baseline_build": args.skip_baseline_build,
                },
                "summary": summary,
                "failed_tasks": [
                    {
                        "index": r.get("index"),
                        "status": r.get("status"),
                        "task_name": r.get("task_name"),
                        "replica_details": r.get("replica_details"),
                    }
                    for r in all_results
                    if r.get("status") != "ok"
                ],
            },
        )
        print(f"Debug manifest: {manifest}")
        if failed:
            print(
                f"WARNING: {len(failed)} task(s) did not complete successfully. "
                f"Inspect logs under {debug_dir / 'train_inner' if debug_dir else work_dir}"
            )

    if ok_results:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
