#!/usr/bin/env python3
"""Launch auditable FSBO/RGPE fits in a fresh directory, with per-group status."""

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.transfer_hpo.protocol import atomic_json, sha256_file, validate_dump

BASE = Path("/root/data/cys/baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("fsbo", "rgpe"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dump-dir", default=str(ROOT / "benchmark_data/ml_train_80_transfer_v2"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--groups", nargs="+", help="Optional dump filename stems to fit")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--validation-tasks")
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if args.method == "fsbo" and (not args.steps or args.steps < 1 or not args.validation_tasks):
        parser.error("FSBO requires positive --steps and --validation-tasks")
    if not args.execute:
        if directory.exists() and any(directory.iterdir()):
            parser.error("Use a fresh run directory; existing runs are preserved")
        coverage = json.loads((Path(args.dump_dir) / "coverage.json").read_text())
        if args.groups:
            coverage["groups"] = [group for group in coverage["groups"] if Path(group["dump"]).stem in args.groups]
            if {Path(group["dump"]).stem for group in coverage["groups"]} != set(args.groups):
                parser.error("Unknown requested groups")
        for group in coverage["groups"]:
            validate_dump(json.loads(Path(group["dump"]).read_text()))
        directory.mkdir(parents=True, exist_ok=True)
        snapshots = directory / "sources"
        snapshots.mkdir()
        for group in coverage["groups"]:
            source = Path(group["dump"])
            target = snapshots / source.name
            shutil.copy2(source, target)
            group["dump"] = str(target)
        atomic_json(snapshots / "coverage.json", coverage)
        if args.validation_tasks:
            shutil.copy2(args.validation_tasks, directory / "validation_tasks.json")
        command = [sys.executable, str(Path(__file__).resolve()), args.method, "--run-dir", str(directory),
                   "--dump-dir", str(snapshots), "--seed", str(args.seed), "--execute"]
        if args.method == "fsbo":
            command += ["--steps", str(args.steps), "--validation-tasks", str(directory / "validation_tasks.json"),
                        "--validate-every", str(args.validate_every)]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if args.detach:
            with (directory / "launcher.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(directory / "launch.json", {"pid": child.pid, "command": command,
                        "launched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            print(json.dumps({"pid": child.pid, "run_dir": str(directory)}))
            return
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return

    coverage = json.loads((Path(args.dump_dir) / "coverage.json").read_text())
    environment, repository = ("bl_fsbo_exact", "FSBO") if args.method == "fsbo" else ("bl_rgpe", "transfer-hpo-framework")
    worker = ROOT / "examples/benchmark" / (args.method + "_worker.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(BASE / "repos" / repository)])
    status = {"method": args.method, "status": "running", "pid": os.getpid(),
              "worker_sha256": sha256_file(worker), "seed": args.seed, "groups": {}}
    atomic_json(directory / "status.json", status)
    failed = False
    for group in coverage["groups"]:
        tag = Path(group["dump"]).stem
        output = directory / args.method
        output.mkdir(exist_ok=True)
        command = [str(BASE / "envs" / environment / "bin/python"), str(worker)]
        if args.method == "rgpe":
            command += ["fit-sources", "--dump", group["dump"], "--cache", str(output / (tag + "_seed%d.pkl" % args.seed)),
                        "--seed", str(args.seed)]
        else:
            command += ["meta-train", "--dump", group["dump"], "--ckpt", str(output / tag),
                        "--validation-tasks", args.validation_tasks, "--steps", str(args.steps),
                        "--seed", str(args.seed), "--validate-every", str(args.validate_every)]
        entry = {"status": "running", "n_source_tasks": group["n_source_tasks"], "command": command}
        status["groups"][tag] = entry
        atomic_json(directory / "status.json", status)
        print("START", tag, flush=True)
        with (directory / (tag + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode)
        failed |= result.returncode != 0
        atomic_json(directory / "status.json", status)
        print(entry["status"].upper(), tag, flush=True)
    status["status"] = "failed" if failed else "complete"
    atomic_json(directory / "status.json", status)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
