#!/usr/bin/env python3
"""Export only NumPy data and native-model bytes across the Python environments."""
import json
from pathlib import Path
import pickle
import sqlite3
import sys
sys.path.insert(0, '/root/data/cys/llmoptimizer_benchmark/tools')
import peptide_transfer_eval as adapter

case = Path(sys.argv[1])
recipe = json.loads((case / 'recipe.json').read_text())
sys.path.insert(0, recipe['cpu_upstream'])
with sqlite3.connect('file:' + str(case / 'runs/pep_235/runtime.sqlite') + '?mode=ro', uri=True) as db:
    value = adapter.storage.get(db, 'state')
with Path(__file__).with_name('rgpe_math_inputs.pkl').open('wb') as stream:
    pickle.dump({k: value[k] for k in ('features', 'model_snapshot')}, stream, protocol=4)
