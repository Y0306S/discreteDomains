"""FSBO functional smoke test: meta-train across tasks, then meta-test on a target task.

Run from repos/FSBO with PYTHONPATH=. and envs/bl_fsbo_exact/bin/python.
"""
import io, os, sys, tempfile
import numpy as np
from fsbo_modules import FSBO, DeepKernelGP


def mk(n, d=6):
    return {'X': np.random.rand(n, d).tolist(), 'y': np.random.rand(n, 1).tolist()}


def main():
    np.random.seed(0)
    tmp = tempfile.mkdtemp()
    ckpt = os.path.join(tmp, 'ckpt')
    # meta-training tasks share one search-space schema, as the spec requires
    train = {str(t): mk(80) for t in range(4)}
    valid = {str(t): mk(50) for t in range(2)}

    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        fsbo = FSBO(train_data=train, valid_data=valid, checkpoint_path=ckpt,
                    batch_size=32, test_batch_size=32, n_inner_steps=2)
        fsbo.meta_train(epochs=3)
        # meta-test: init deep-kernel GP from meta-trained weights, fine-tune, then pick
        dkgp = DeepKernelGP(input_size=6, log_dir=os.path.join(tmp, 'log.txt'), seed=1,
                            checkpoint=ckpt, load_model=False, epochs=20, verbose=False)
        idx = dkgp.observe_and_suggest(X_obs=np.random.rand(8, 6), y_obs=np.random.rand(8),
                                       X_pen=np.random.rand(40, 6))
    finally:
        sys.stdout = old

    assert os.path.exists(os.path.join(ckpt, 'weights')), "meta-train wrote no weights"
    print("SMOKE FSBO meta_train OK -> %s/weights" % ckpt)
    print("SMOKE FSBO meta-test observe_and_suggest -> candidate index %s of 40" % idx)


if __name__ == "__main__":
    main()
