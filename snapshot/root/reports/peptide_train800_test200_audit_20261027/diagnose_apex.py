#!/usr/bin/env python3
"""Compare raw, uncached APEX predictions across batch compositions."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path('/root/data/cys/BOLT/optimization/peptides')
CAMPAIGN = Path('/root/data/logs/peptide_bo_train800_20261027_v5b')


def main():
    os.sched_setaffinity(0, {16})
    os.environ.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.05)
    sys.path.insert(0, str(UPSTREAM))
    os.chdir(UPSTREAM)
    from apex_oracle.APEX_predict import APEX_models, predict_APEX
    recipe = json.loads((CAMPAIGN / 'recipe.json').read_text())
    pools = json.loads(Path(recipe['prepared_pool']).read_text())
    sequence = 'LEWPWKRLALGSIIARYL'
    pool = pools['pep_278']
    index = pool['sequences'].index(sequence)
    batches = [
        ('single', [sequence], 0),
        ('repeat_16', [sequence] * 16, 0),
        ('repeat_64', [sequence] * 64, 0),
        ('pool_64', pool['sequences'][-64:], index - (len(pool['sequences']) - 64)),
        ('pool_100', pool['sequences'], index),
        ('prefix_3', pool['sequences'][:3] + [sequence], 3),
    ]
    records = []
    with torch.inference_mode():
        for repeat in range(3):
            for label, sequences, target_index in batches:
                start = time.monotonic()
                result = predict_APEX(sequences)
                records.append(dict(batch=label, size=len(sequences), repeat=repeat,
                                    mic=float(result[target_index, 0]),
                                    seconds=time.monotonic() - start))
    original = -pool['scores'][index]
    report = dict(sequence=sequence, task='pep_278', initialization_index=index,
                  source=pool['sources'][index], original_mic=original,
                  raw_uncached=True, ensemble_models=len(APEX_models), records=records,
                  maximum_relative_difference_from_original=max(
                      abs(r['mic'] - original) / original for r in records),
                  torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name(),
                  predictor_sha256=hashlib.sha256(
                      (UPSTREAM / 'apex_oracle/APEX_predict.py').read_bytes()).hexdigest())
    temporary = ROOT / 'apex_batch_diagnosis.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(ROOT / 'apex_batch_diagnosis.json')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
