"""Register source-only SeqIO tasks, then enter the original T5X trainer."""

import json
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mindware.components.optimizers.transfer_hpo.optformer_bridge import install_t5x_compat, register_task

install_t5x_compat()
register_task(json.loads(Path(os.environ["ML_OPTFORMER_MANIFEST"]).read_text()))
runpy.run_module("t5x.train", run_name="__main__", alter_sys=True)
