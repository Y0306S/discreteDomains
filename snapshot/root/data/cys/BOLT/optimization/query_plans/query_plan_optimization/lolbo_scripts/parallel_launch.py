"""
Launch parallel query-plan BO runs across GPUs (BayesQO-style scheduler).

Each worker gets its own GPU and Postgres port so oracle calls do not contend.

Example:
  python parallel_launch.py --num_per_gpu 2 \\
    --task_file ../../tasks/ceb_tasks.txt \\
    --init_w_bao True --allow_cross_joins False \\
    --track_with_wandb False --max_n_oracle_calls 80

Do not pass `run_lolbo done` here; this launcher calls run_lolbo() itself.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
import traceback

import fire
import torch

sys.path.append("../")

from lolbo_scripts.info_transformer_vae_optimization import InfoTransformerVAEOptimization

# Fire leftover command tokens from BOLT single-run CLI; parallel_launch calls run_lolbo() itself.
_FIRE_COMMAND_TOKENS = {"run_lolbo", "done", "-"}

# Linear slots: GPU g runner r -> port 5432 + g*num_per_gpu + r
# 8 GPUs × 1 worker: 5432–5439
# 4 GPUs × 2 workers: 5432–5439
def worker_db_port(gpu_id: int, runner: int, num_per_gpu: int) -> int:
    return 5432 + gpu_id * num_per_gpu + runner


def isolated_run_task(
    task_id: str,
    gpu_id: int,
    runner: int,
    num_per_gpu: int,
    *args: list,
    **kwargs: dict,
) -> bool:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("DB_HOST", "127.0.0.1")
    os.environ.setdefault("DB_USER", "imdb")
    os.environ.setdefault("DB_PASSWORD", "imdb")
    os.environ.setdefault("DB_NAME", "imdb")
    os.environ["DB_PORT"] = str(worker_db_port(gpu_id, runner, num_per_gpu))

    opt = None
    try:
        with torch.cuda.device(gpu_id):
            print(
                f"[worker gpu={gpu_id} slot={runner} port={os.environ['DB_PORT']}] starting {task_id}",
                flush=True,
            )
            opt = InfoTransformerVAEOptimization(workload_name=task_id, *args, **kwargs)
            opt.run_lolbo()
        return True
    except Exception as e:
        if opt is not None:
            opt.done()
        print(f"Error in task {task_id} on GPU {gpu_id} runner {runner}: {e}")
        traceback.print_exc()
        raise e


def worker(
    task_queue: mp.JoinableQueue,
    gpu_id: int,
    runner: int,
    num_per_gpu: int,
    max_retries: int = 3,
    retry_delay: int = 60,
    *args: list,
    **kwargs: dict,
) -> None:
    os.environ.setdefault("DB_HOST", "127.0.0.1")
    os.environ.setdefault("DB_USER", "imdb")
    os.environ.setdefault("DB_PASSWORD", "imdb")
    os.environ.setdefault("DB_NAME", "imdb")
    os.environ["DB_PORT"] = str(worker_db_port(gpu_id, runner, num_per_gpu))
    while True:
        task_id = task_queue.get()
        print("#" * 80, flush=True)
        if task_id is None:
            print(f"GPU {gpu_id} runner {runner} exiting", flush=True)
            task_queue.task_done()
            break

        retries = 0
        while retries < max_retries:
            print(
                f"GPU {gpu_id} runner {runner} running task {task_id} (attempt {retries + 1})",
                flush=True,
            )
            p = mp.Process(
                target=isolated_run_task,
                args=(task_id, gpu_id, runner, num_per_gpu, *args),
                kwargs=kwargs,
            )
            p.start()
            p.join()

            if p.exitcode == 0:
                done_dir = "optimization_all_collected_data"
                os.makedirs(done_dir, exist_ok=True)
                with open(os.path.join(done_dir, f"{task_id}.done"), "w") as fh:
                    fh.write("ok\n")
                print(
                    f"GPU {gpu_id} runner {runner} successfully completed task {task_id}",
                    flush=True,
                )
                break

            retries += 1
            if retries < max_retries:
                print(
                    f"GPU {gpu_id} runner {runner} task {task_id} failed; retry in {retry_delay}s",
                    flush=True,
                )
                time.sleep(retry_delay)
            else:
                print(
                    f"GPU {gpu_id} runner {runner} task {task_id} failed after {max_retries} attempts",
                    flush=True,
                )

        task_queue.task_done()


def _completed_csv_path(task_id: str, collected_dir: str) -> str:
    return os.path.join(collected_dir, f"optimimze-db_nowandb_{task_id}_all-data-collected.csv")


def _csv_is_complete(path: str, min_rows: int) -> bool:
    if not os.path.isfile(path) or min_rows <= 0:
        return False
    with open(path, "r") as f:
        n = sum(1 for _ in f) - 1
    return n >= min_rows


def _task_is_complete(task_id: str, collected_dir: str, min_rows: int) -> bool:
    if os.path.isfile(os.path.join(collected_dir, f"{task_id}.done")):
        return True
    return _csv_is_complete(_completed_csv_path(task_id, collected_dir), min_rows)


def launch_tasks(
    num_per_gpu: int,
    task_ids: list[str] = None,
    task_file: str = None,
    skip_completed: bool = False,
    collected_dir: str = "optimization_all_collected_data",
    min_completed_rows: int = 140,
    *args: list,
    **kwargs: dict,
) -> None:
    if (task_ids is None) == (task_file is None):
        raise ValueError("Exactly one of task_ids or task_file must be provided")

    if task_file is not None:
        with open(task_file, "r") as f:
            task_ids = [line.strip() for line in f if line.strip()]
    elif isinstance(task_ids, str):
        task_ids = [task_ids]

    args = tuple(a for a in args if str(a) not in _FIRE_COMMAND_TOKENS)
    kwargs.pop("skip_completed", None)
    kwargs.pop("collected_dir", None)
    kwargs.pop("min_completed_rows", None)

    if skip_completed:
        kept = []
        skipped = 0
        for tid in task_ids:
            if _task_is_complete(tid, collected_dir, min_completed_rows):
                skipped += 1
            else:
                kept.append(tid)
        print(
            f"skip_completed: keeping {len(kept)}/{len(task_ids)} tasks "
            f"(skipped {skipped} with .done or >={min_completed_rows} CSV rows)",
            flush=True,
        )
        task_ids = kept

    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    print(f"Launching {num_gpus} GPUs x {num_per_gpu} workers = {num_gpus * num_per_gpu} slots", flush=True)
    for gpu_id in range(num_gpus):
        for runner in range(num_per_gpu):
            print(
                f"  slot gpu={gpu_id} runner={runner} -> DB_PORT={worker_db_port(gpu_id, runner, num_per_gpu)}",
                flush=True,
            )

    task_queue = mp.JoinableQueue()
    processes = []

    def signal_handler(sig, frame):
        print("Interrupt received, terminating processes...", flush=True)
        while not task_queue.empty():
            task_queue.get()
            task_queue.task_done()
        for _ in range(num_gpus * num_per_gpu):
            task_queue.put(None)
        for proc in processes:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for task_id in task_ids:
        task_queue.put(task_id)

    for gpu_id in range(num_gpus):
        for runner in range(num_per_gpu):
            proc = mp.Process(
                target=worker,
                args=(task_queue, gpu_id, runner, num_per_gpu, *args),
                kwargs=kwargs,
            )
            proc.start()
            processes.append(proc)
            time.sleep(1)

    for _ in range(num_gpus * num_per_gpu):
        task_queue.put(None)

    task_queue.join()

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    fire.Fire(launch_tasks)
