"""Small subprocess used to exercise interruption and evaluator cleanup."""
import os
from pathlib import Path
import time

from mindware.components.optimizers.isolated_evaluator import run_isolated


def objective():
    folder = Path(os.environ["ML_SESSION_DIR"])
    (folder / "objective_pid").write_text(str(os.getpid()))
    time.sleep(40)
    return {"objective_value": 0.25}


if __name__ == "__main__":
    result = run_isolated(objective, (), {}, 30)
    print(result, flush=True)
