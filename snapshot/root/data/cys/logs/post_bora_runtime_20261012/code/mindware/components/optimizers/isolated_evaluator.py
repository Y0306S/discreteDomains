"""Bound CPU evaluations without forking an optimizer's CUDA runtime."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import traceback


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def _birth(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except OSError:
        return None


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _watch(journal_path, owner, birth, group):
    group_birth = _birth(group)
    while group_birth is not None and _birth(group) == group_birth:
        if _birth(owner) != birth:
            try:
                record = json.loads(Path(journal_path).read_text())
                if record.get("status") == "running" and (record["owner_pid"], record["owner_birth"]) == (owner, birth):
                    record.update(status="interrupted", ended_at=time.time(),
                                  accounting="watchdog", elapsed_seconds=record["previous_seconds"] + time.monotonic() - record["started_monotonic"])
                    _atomic_json(journal_path, record)
            finally:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return
        time.sleep(0.2)


def _evaluate(connection, objective, args, kwargs, journal_path=None, owner=None, birth=None):
    os.setsid()
    if journal_path:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watch", journal_path,
                          str(owner), birth, str(os.getpid())], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        result = dict(result=objective(*args, **kwargs), timeout=False, traceback=None)
    except Exception:
        result = dict(result=None, timeout=False, traceback=traceback.format_exc())
    try:
        connection.send(result)
    finally:
        connection.close()


def _stop(process):
    if process.pid is None:
        return
    # The evaluator owns a session, including any estimator subprocesses.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        if process.is_alive():
            process.kill()
    process.join(timeout=5)


def run_isolated(objective, args, kwargs, timeout):
    if timeout is None or timeout <= 0:
        raise ValueError("Isolated evaluation requires a positive time limit")
    directory = os.environ.get("ML_SESSION_DIR")
    evaluation_id = os.environ.get("ML_EVALUATION_ID")
    journal_path = result_path = None
    prior = {}
    previous = 0.0
    if directory and evaluation_id is not None:
        from mindware.components.optimizers.ml_checkpoint import atomic_pickle
        from mindware.components.optimizers.run_artifact_store import read_bytes as artifact_bytes
        folder = Path(directory) / "evaluations" / evaluation_id
        folder.mkdir(parents=True, exist_ok=True)
        journal_path, result_path = folder / "execution.json", folder / "result.pkl"
        try:
            return pickle.loads(artifact_bytes(directory, str(result_path.relative_to(directory))))
        except FileNotFoundError:
            pass
        if journal_path.exists():
            prior = json.loads(journal_path.read_text())
            if prior.get("status") == "running":
                same_boot = prior.get("boot_id") == _boot_id()
                if same_boot and _birth(prior["owner_pid"]) == prior["owner_birth"]:
                    raise RuntimeError("Evaluation already has a live owner")
                time.sleep(0.3)
                prior = json.loads(journal_path.read_text())
                if prior.get("status") == "running":
                    prior.update(status="interrupted", accounting="last_heartbeat",
                                 ended_at=prior.get("updated_at", prior["started_at"]))
            child_pid, child_birth = prior.get("evaluator_pid"), prior.get("evaluator_birth")
            if prior.get("boot_id") == _boot_id() and child_birth and _birth(child_pid) == child_birth:
                try:
                    os.killpg(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                for _ in range(50):
                    if _birth(child_pid) != child_birth:
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("Previous evaluator has not exited")
            previous = float(prior.get("elapsed_seconds", 0))
        if previous >= timeout:
            result = dict(result=None, timeout=True, traceback=None, elapsed_time=previous)
            atomic_pickle(result_path, result)
            return result
    started = time.monotonic()
    owner, birth = os.getpid(), _birth(os.getpid())
    executions = list(prior.get("executions", []))
    if prior:
        executions.append({key: value for key, value in prior.items() if key != "executions"})
    record = {"status": "running", "owner_pid": owner, "owner_birth": birth, "boot_id": _boot_id(),
              "started_at": time.time(), "started_monotonic": started,
              "previous_seconds": previous, "elapsed_seconds": previous,
              "execution_number": int(prior.get("execution_number", 0)) + 1,
              "interrupted_executions": int(prior.get("interrupted_executions", 0)) + bool(prior),
              "downtime_seconds": float(prior.get("downtime_seconds", 0)) +
              (max(0.0, time.time() - prior.get("ended_at", prior.get("updated_at", time.time()))) if prior else 0.0),
              "allowance_seconds": timeout, "heartbeat_seconds": 1, "executions": executions}
    if journal_path:
        _atomic_json(journal_path, record)
    context = mp.get_context("forkserver")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_evaluate, args=(sender, objective, args, kwargs,
                              str(journal_path) if journal_path else None, owner, birth))
    result = None
    try:
        process.start()
        record.update(evaluator_pid=process.pid, evaluator_birth=_birth(process.pid))
        if journal_path:
            _atomic_json(journal_path, record)
        sender.close()
        while result is None:
            remaining = max(0.0, timeout - previous - (time.monotonic() - started))
            if receiver.poll(min(1.0, remaining)):
                try:
                    result = receiver.recv()
                except EOFError:
                    result = dict(result=None, timeout=False, traceback="Evaluator exited without a result")
            elif remaining <= 0:
                result = dict(result=None, timeout=True, traceback=None)
            if journal_path:
                record.update(elapsed_seconds=previous + time.monotonic() - started, updated_at=time.time())
                _atomic_json(journal_path, record)
    finally:
        sender.close()
        receiver.close()
        _stop(process)
        elapsed = previous + time.monotonic() - started
        if journal_path:
            record.update(status="completed" if result is not None else "interrupted",
                          elapsed_seconds=elapsed, ended_at=time.time())
            if result is not None:
                result["elapsed_time"] = elapsed
                atomic_pickle(result_path, result)
            _atomic_json(journal_path, record)
    result["elapsed_time"] = elapsed
    output_dir = getattr(objective, "output_dir", None)
    if output_dir:
        event = {
            "evaluation_id": evaluation_id,
            "timestamp": time.time(),
            "timeout_seconds": timeout,
            "elapsed_seconds": result["elapsed_time"],
            "timeout": result["timeout"],
            "traceback": result["traceback"],
            "start_method": "forkserver",
        }
        with open(os.path.join(output_dir, "evaluation_events.jsonl"), "a") as stream:
            stream.write(json.dumps(event) + "\n")
    return result


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "--watch":
    _watch(sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
