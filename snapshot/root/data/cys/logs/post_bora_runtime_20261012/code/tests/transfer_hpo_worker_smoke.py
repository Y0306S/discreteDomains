"""Bounded integration checks using only real allowed historical observations."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark_data/ml_train_80_transfer_v2"
BASE = Path("/root/data/cys/baseline")


def run(method, *args):
    environment = {"fsbo": "bl_fsbo_exact", "rgpe": "bl_rgpe", "optformer": "bl_optformer"}[method]
    repository = {"fsbo": "FSBO", "rgpe": "transfer-hpo-framework", "optformer": "optformer-neurips22"}[method]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(BASE / "repos" / repository), str(BASE / "repos/t5x-2023-07")])
    command = [str(BASE / "envs" / environment / "bin/python"),
               str(ROOT / "examples/benchmark" / (method + "_worker.py")), *map(str, args)]
    with tempfile.NamedTemporaryFile(prefix="transfer_" + method + "_", suffix=".log", delete=False) as log:
        print("WORKER LOG", log.name, flush=True)
        result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        print(Path(log.name).read_text()[-6000:], flush=True)
        result.check_returncode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("fsbo", "rgpe", "optformer"))
    parser.add_argument("--directory")
    args = parser.parse_args()
    directory = Path(args.directory or tempfile.mkdtemp(prefix="ml_transfer_" + args.method + "_"))
    directory.mkdir(parents=True, exist_ok=True)
    print("SMOKE ARTIFACTS", directory, flush=True)
    dump = json.loads((DATA / "rgs_adaboost.json").read_text())
    selected = list(dump["tasks"])[:2]
    dump["tasks"] = {key: dump["tasks"][key] for key in selected}
    for record in dump["tasks"].values():
        for key in ("X", "y_maximize", "y_minimize", "configs", "times"):
            record[key] = record[key][:6]
        record["n"] = 6
    dump_path = directory / "dump.json"
    dump_path.write_text(json.dumps(dump))
    validation = directory / "validation.json"
    validation.write_text(json.dumps(selected[:1]))
    record = dump["tasks"][selected[1]]
    request = {"schema_fingerprint": dump["schema_fingerprint"], "seed": 1}
    request_path, out = directory / "request.json", directory / "suggestion.json"
    if args.method == "fsbo":
        model = directory / "checkpoint"
        flags = ["--dump", dump_path, "--validation-tasks", validation, "--ckpt", model]
        run("fsbo", "meta-train", *flags, "--steps", 1)
        run("fsbo", "meta-train", *flags, "--steps", 2, "--resume")
        request.update(X_obs=record["X"][:3], y_obs=record["y_maximize"][:3],
                       X_pen=record["X"][3:], finetune_epochs=2)
        request_path.write_text(json.dumps(request))
        run("fsbo", "suggest", "--ckpt", model, "--request", request_path, "--out", out)
    elif args.method == "rgpe":
        cache = directory / "sources.pkl"
        run("rgpe", "fit-sources", "--dump", dump_path, "--cache", cache)
        request.update(target_configs=record["configs"][:3], y_minimize=record["y_minimize"][:3],
                       pending_configs=record["configs"][3:], n_trials=10, num_posterior_samples=1000)
        request_path.write_text(json.dumps(request))
        run("rgpe", "suggest", "--dump", dump_path, "--cache", cache, "--request", request_path, "--out", out)
    else:
        records = [json.loads(line) for line in (DATA / "optformer/sequences.jsonl").read_text().splitlines()]
        rows = [row for row in records if row["task_id"] in selected]
        for row in rows:
            row.update(dump["tasks"][row["task_id"]])
        sequences = directory / "sequences.jsonl"
        sequences.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        model = directory / "model"
        run("optformer", "train", "--sequences", sequences, "--validation-tasks", validation,
            "--model-dir", model, "--steps", 1, "--batch-size", 1, "--model-size", "tiny", "--save-every", 1,
            "--eval-steps", 1)
        run("optformer", "train", "--sequences", sequences, "--validation-tasks", validation,
            "--model-dir", model, "--steps", 2, "--batch-size", 1, "--model-size", "tiny", "--save-every", 1,
            "--eval-steps", 1, "--resume")
        request.update(rows[1])
        request.update(configs=record["configs"][:3], y_maximize=record["y_maximize"][:3], num_samples=2)
        request_path.write_text(json.dumps(request))
        run("optformer", "suggest", "--model-dir", model, "--request", request_path, "--out", out)
    suggestion = json.loads(out.read_text())
    if args.method != "optformer":
        assert 0 <= suggestion["index"] < 3, suggestion
    print("PASSED", args.method, suggestion, flush=True)


if __name__ == "__main__":
    main()
