"""Inventory or retire only the superseded September 2026 ML campaign workers."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import time


ROOTS = {"benchmark_data/ml_test_20_fixed_lolbo100": "lolbo",
         "benchmark_data/ml_test_20_fixed_rs100": "random_search",
         "benchmark_data/ml_test_20_fixed_tpe100": "tpe"}
LAUNCHERS = {"run_ml_test_20_fixed_baselines.sh", "run_ml_test_20_fixed_baselines_lolbo.sh"}


def snapshot():
    processes = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
            command = [p.decode() for p in (path / "cmdline").read_bytes().split(b"\0") if p]
            processes[int(path.name)] = {"pid": int(path.name), "ppid": int(fields[1]),
                                         "start_ticks": fields[19], "state": fields[0], "command": command}
        except (OSError, ValueError):
            continue
    return processes


def selected(process, methods, include_recovery=False):
    command = process["command"]
    if len(command) < 2 or any(arg in command for arg in ("-c", "-lc")):
        return False
    if Path(command[0]).name == "bash" and Path(command[1]).name in LAUNCHERS:
        try:
            values = (Path("/proc") / str(process["pid"]) / "environ").read_bytes().split(b"\0")
            env = dict(value.split(b"=", 1) for value in values if b"=" in value)
            root = env.get(b"OUTPUT_ROOT", b"").decode()
            return root in ROOTS and ROOTS[root] in methods and env.get(b"OPTIMIZERS", b"").decode() == ROOTS[root]
        except OSError:
            return False
    if "python" in Path(command[0]).name and Path(command[1]).name == "run_toy_categorical_optimizers.py":
        try:
            root = command[command.index("--output-dir") + 1]
            method = command[command.index("--optimizer") + 1]
            legacy = root in ROOTS and method == ROOTS[root]
            recovery = include_recovery and (
                "/benchmark_data/ml_test_20_fixed_recovery_" in root
                and root.endswith("/outputs/" + method)
            )
            return method in methods and (legacy or recovery)
        except (ValueError, IndexError):
            return False
    if include_recovery and "python" in Path(command[0]).name and Path(command[1]).name == "ml_fixed_recovery.py":
        try:
            return command[2] == "worker" and command[command.index("--method") + 1] in methods
        except (ValueError, IndexError):
            return False
    return False


def same_process(process):
    try:
        fields = (Path("/proc") / str(process["pid"]) / "stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19] == process["start_ticks"] and fields[0] != "Z"
    except OSError:
        return False


def send(process, sig):
    if same_process(process):
        try:
            os.kill(process["pid"], sig)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("lolbo", "random_search", "tpe"),
                        default=("lolbo", "random_search", "tpe"))
    parser.add_argument("--include-recovery", action="store_true")
    args = parser.parse_args()
    processes = snapshot()
    roots = {pid for pid, process in processes.items() if selected(process, args.methods, args.include_recovery)}
    if args.apply:
        for pid in roots:
            send(processes[pid], signal.SIGSTOP)
        processes.update(snapshot())
    victims = set(roots)
    while True:
        children = {pid for pid, process in processes.items() if process["ppid"] in victims}
        if children <= victims:
            break
        victims.update(children)
    records = [processes[pid] for pid in sorted(victims)]
    audit = {"host": socket.gethostname(), "timestamp": time.time(),
             "applied": args.apply, "processes": records}
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2))
    print(json.dumps({"host": audit["host"], "apply": args.apply,
                      "matched": len(roots), "including_descendants": len(victims),
                      "pids": sorted(victims)}, indent=2), flush=True)
    if args.apply:
        for process in records:
            send(process, signal.SIGTERM)
            send(process, signal.SIGCONT)
        time.sleep(3)
        for process in records:
            send(process, signal.SIGKILL)
        time.sleep(1)
        audit["surviving_pids"] = [p["pid"] for p in records if same_process(p)]
        args.audit.write_text(json.dumps(audit, indent=2))
        print("Survivors:", audit["surviving_pids"])
        if audit["surviving_pids"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
