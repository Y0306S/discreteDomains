#!/usr/bin/env python3
"""Report source integrity and actual trained artifacts, without launching jobs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.transfer_hpo.protocol import sha256_file, sha256_directory, validate_dump


def checkpoint_status(path, dump, method):
    if not path.exists():
        return "missing"
    try:
        meta = json.loads(path.read_text())
        if meta.get("status") != "complete":
            return meta.get("status", "invalid")
        if meta.get("schema_fingerprint") != dump["schema_fingerprint"]:
            return "schema_mismatch"
        if method == "fsbo":
            artifact, field = path.parent / "weights", "weights_sha256"
        else:
            artifact, field = path.with_suffix(".pkl"), "cache_sha256"
            if meta.get("source_tasks") != sorted(dump["tasks"]):
                return "source_mismatch"
        if meta[field] != sha256_file(artifact):
            return "digest_mismatch"
        return "complete"
    except (OSError, ValueError, KeyError):
        return "invalid"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, default=ROOT / "benchmark_data/ml_train_80_transfer_v2")
    parser.add_argument("--fsbo-ckpt-dir", type=Path, default=ROOT / "benchmark_data/ml_train_80_fsbo_ckpts")
    parser.add_argument("--rgpe-cache-dir", type=Path, action="append")
    parser.add_argument("--optformer-model-dir", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    caches = args.rgpe_cache_dir or [ROOT / "benchmark_data/ml_train_80_rgpe_sources"]
    coverage = json.loads((args.dump_dir / "coverage.json").read_text())
    complete = {"fsbo": 0, "rgpe": 0}
    for group in coverage["groups"]:
        source = args.dump_dir / Path(group["dump"]).name
        dump = json.loads(source.read_text())
        validate_dump(dump)
        tag = source.stem
        fsbo_path = args.fsbo_ckpt_dir / tag / "meta.json"
        fsbo = checkpoint_status(fsbo_path, dump, "fsbo")
        cache_paths = [cache / ("%s_seed%d.json" % (tag, args.seed)) for cache in caches]
        cache_path = next((path for path in cache_paths if path.exists()), cache_paths[0])
        rgpe = checkpoint_status(cache_path, dump, "rgpe")
        for method, status, path in (("fsbo", fsbo, fsbo_path), ("rgpe", rgpe, cache_path)):
            if status == "complete" and json.loads(path.read_text()).get("dump_sha256") != sha256_file(source):
                if method == "fsbo":
                    fsbo = "source_mismatch"
                else:
                    rgpe = "source_mismatch"
        complete["fsbo"] += fsbo == "complete"
        complete["rgpe"] += rgpe == "complete"
        print("%-33s sources=%3d fsbo=%-16s rgpe=%s" % (tag, len(dump["tasks"]), fsbo, rgpe))
    optformer = "missing"
    if args.optformer_model_dir:
        try:
            directory = args.optformer_model_dir
            meta = json.loads((directory / "meta.json").read_text())
            optformer = meta.get("status", "invalid")
            if meta.get("model_size") != "base":
                optformer = "test_model_only"
            elif optformer == "complete":
                if (sha256_directory(directory / meta["checkpoint"]) != meta["checkpoint_sha256"]
                        or sha256_file(directory / "model.gin") != meta["model_config_sha256"]
                        or sha256_file(meta["sequences"]) != meta["sequences_sha256"]):
                    optformer = "digest_mismatch"
        except (OSError, ValueError, KeyError):
            optformer = "invalid"
    print("OptFormer:", optformer)
    print("Verified artifacts:", complete, "of", len(coverage["groups"]), "compatible spaces")
    return 0 if all(n == len(coverage["groups"]) for n in complete.values()) and optformer == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
