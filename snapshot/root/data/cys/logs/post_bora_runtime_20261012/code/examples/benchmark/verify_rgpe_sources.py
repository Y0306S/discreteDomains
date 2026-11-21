#!/usr/bin/env python3
"""Validate fitted source GP posteriors, then optionally publish local links."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from smac.configspace import convert_configurations_to_array

from rgpe_worker import configs_from_rows, load_dump
from mindware.components.optimizers.transfer_hpo.protocol import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--publish-dir", type=Path)
    args = parser.parse_args()
    coverage = json.loads((args.dump_dir / "coverage.json").read_text())
    artifacts, total = [], 0
    for group in coverage["groups"]:
        dump_path = args.dump_dir / Path(group["dump"]).name
        dump, tasks, cs = load_dump(dump_path)
        name = "%s_seed%d.pkl" % (dump_path.stem, args.seed)
        candidates = [path / name for path in args.cache_dir if (path / name).is_file()]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one fitted cache: %s" % name)
        cache = candidates[0]
        meta = json.loads(cache.with_suffix(".json").read_text())
        if (meta["status"] != "complete" or meta["dump_sha256"] != sha256_file(dump_path)
                or meta["cache_sha256"] != sha256_file(cache) or meta["source_tasks"] != tasks
                or meta["schema_fingerprint"] != dump["schema_fingerprint"]):
            raise ValueError("Invalid fitted cache: %s" % cache)
        with cache.open("rb") as stream:
            payload = pickle.load(stream)
        if len(payload["base_models"]) != len(tasks):
            raise ValueError("Fitted GP count differs from source task count")
        for task, model in zip(tasks, payload["base_models"]):
            configs = configs_from_rows(cs, dump["tasks"][task]["configs"][:3], dump["algorithm"])
            mean, variance = model.predict(convert_configurations_to_array(configs))
            if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any():
                raise ValueError("Invalid source posterior: %s" % task)
            total += 1
        artifacts.extend([cache, cache.with_suffix(".json")])
        print("VERIFIED", dump_path.stem, len(tasks), flush=True)
    if args.publish_dir:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        for source in artifacts:
            destination = args.publish_dir / source.name
            if destination.exists() or destination.is_symlink():
                raise ValueError("Refusing to replace existing artifact: %s" % destination)
        for source in artifacts:
            (args.publish_dir / source.name).symlink_to(source.resolve())
    print(json.dumps({"verified_source_gps": total, "verified_groups": len(coverage["groups"]),
                      "published": str(args.publish_dir) if args.publish_dir else None}))


if __name__ == "__main__":
    main()
