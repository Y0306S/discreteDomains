"""Exercise success, failure, CUDA isolation, and timeout process cleanup."""

import os
from pathlib import Path
import subprocess
import tempfile
import time

from mindware.components.optimizers.isolated_evaluator import run_isolated


def success():
    import torch
    return {"pid": os.getpid(), "cuda_initialized": torch.cuda.is_initialized(), "score": 0.5}


def failure():
    raise ValueError("expected evaluator failure")


def slow(path):
    process = subprocess.Popen(["sleep", "60"])
    Path(path).write_text(str(process.pid))
    time.sleep(60)


def main():
    import torch
    if torch.cuda.is_available():
        torch.zeros(1, device="cuda")
        assert torch.cuda.is_initialized()
    result = run_isolated(success, (), {}, 30)
    assert result["result"]["score"] == 0.5, result
    assert not result["result"]["cuda_initialized"], result
    assert result["result"]["pid"] != os.getpid()
    result = run_isolated(failure, (), {}, 30)
    assert not result["timeout"] and "expected evaluator failure" in result["traceback"], result
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "child.pid"
        result = run_isolated(slow, (str(path),), {}, 10)
        assert result["timeout"] and result["elapsed_time"] < 16, result
        assert path.exists(), "Child did not reach objective before timeout"
        process = Path("/proc") / path.read_text()
        if process.exists():
            assert ") Z " in (process / "stat").read_text(), "Estimator descendant survived timeout"
    print("PASS: success, exception, CUDA isolation, timeout, and descendant cleanup")


if __name__ == "__main__":
    main()
