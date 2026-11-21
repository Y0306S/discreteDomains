#!/usr/bin/env python3
"""Subprocess worker: load one example from a cached TFDS dataset."""

import json
import os
import sys
import time


def main() -> None:
    data_dir = sys.argv[1]
    name = sys.argv[2]
    os.environ["TFDS_DATA_DIR"] = data_dir

    google_research = "/root/data/cys/blockdata_archive/LLMOptimizer/google-research"
    if google_research not in sys.path:
        sys.path.insert(0, google_research)
    from task_set import datasets as _task_set_datasets  # noqa: F401 — register custom TFDS builders

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    import tensorflow_datasets as tfds

    t0 = time.monotonic()
    try:
        builder = tfds.builder(name, data_dir=data_dir)
        split = "train" if "train" in builder.info.splits else list(builder.info.splits.keys())[0]
        ds = builder.as_dataset(split=split, shuffle_files=False)
        batch = ds.make_one_shot_iterator().get_next()
        with tf.Session() as sess:
            ex = sess.run(batch)
        shapes = {k: tuple(v.shape) for k, v in ex.items() if hasattr(v, "shape")}
        print(
            json.dumps(
                {
                    "status": "ok",
                    "split": split,
                    "keys": sorted(ex.keys()),
                    "shapes": shapes,
                    "elapsed_s": round(time.monotonic() - t0, 2),
                }
            )
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "error": str(exc),
                    "elapsed_s": round(time.monotonic() - t0, 2),
                }
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
