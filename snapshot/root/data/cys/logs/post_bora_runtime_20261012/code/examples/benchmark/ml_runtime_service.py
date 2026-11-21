"""Serve a private ToSFiT actor or a persistent OptFormer model on loopback."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mindware.components.optimizers.runtime_rpc import fingerprint, server


def tosfit_identity():
    from mindware.components.optimizers.online_llm_optimizer import QWEN_BASE
    from mindware.components.optimizers.tosfit_runtime import RECIPE_PATH
    from mindware.components.optimizers.online_upstream import UPSTREAM
    files = [Path(__file__), RECIPE_PATH,
             ROOT / "mindware/components/optimizers/tosfit_runtime.py",
             ROOT / "mindware/components/optimizers/tosfit_remote.py",
             ROOT / "mindware/components/optimizers/runtime_rpc.py"]
    files += [UPSTREAM / "tosfit/src" / name for name in
              ("language_model.py", "gaussian_process.py", "lite.py", "utils/misc.py")]
    files += [Path(QWEN_BASE) / name for name in ("config.json", "model.safetensors.index.json")]
    return fingerprint({"base": str(QWEN_BASE),
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.sched_setaffinity(0, config["cpus"])
    lock_path = args.config.with_suffix(".lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if config["kind"] == "tosfit":
            from mindware.components.optimizers.tosfit_remote import DurableToSFiTActor
            from mindware.components.optimizers.tosfit_runtime import ToSFiTRuntime
            if config["identity"] != tosfit_identity():
                raise ValueError("ToSFiT implementation changed since preparation")
            app = DurableToSFiTActor(config["session_root"], config["identity"],
                lambda: ToSFiTRuntime("cuda:0", "cuda:1"),
                checkpoint_lock=config["checkpoint_lock"])
        elif config["kind"] == "optformer":
            from optformer_worker import configure_runtime, OptFormerEngine
            from mindware.components.optimizers.optformer_remote import OptFormerService
            configure_runtime()
            app = OptFormerService(OptFormerEngine(config["model_dir"]), config["journal"])
            if app.identity != config["identity"]:
                raise ValueError("OptFormer checkpoint changed since preparation")
        else:
            raise ValueError("Unknown runtime kind")
        httpd = server(app, config["port"], max_pending=config.get("max_pending", 8))
        print(json.dumps({"ready": True, "port": config["port"], "kind": config["kind"]}), flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
