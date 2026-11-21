"""Load only the upstream numerical modules, avoiding optional application imports."""

import hashlib
import importlib.util
from pathlib import Path
import sys
import types


UPSTREAM = Path("/root/data/cys/baseline/repos")


def load_module(name, path):
    path = Path(path)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bora_policy():
    return load_module("_ml_bora_policy", UPSTREAM / "bora-the-explorer/bora/policy.py")


def tosfit_core():
    root = UPSTREAM / "tosfit/src"
    gp = load_module("_ml_tosfit_gp", root / "gaussian_process.py")
    misc = load_module("_ml_tosfit_misc", root / "utils/misc.py")
    # lite imports only src.utils.misc. Restore the process's original namespace
    # immediately so other upstream packages can keep their own 'src' modules.
    names = ("src", "src.utils", "src.utils.misc")
    previous = {name: sys.modules.get(name) for name in names}
    package, utils = types.ModuleType("src"), types.ModuleType("src.utils")
    package.utils, utils.misc = utils, misc
    try:
        sys.modules.update({"src": package, "src.utils": utils, "src.utils.misc": misc})
        lite = load_module("_ml_tosfit_lite", root / "lite.py")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return gp, misc, lite
