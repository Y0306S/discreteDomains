#!/usr/bin/env python3
"""Paper FSBO meta-training and target EI in the isolated FSBO environment."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.transfer_hpo.protocol import (
    acquisition_diagnostics, atomic_json, partition_tasks, sha256_file, validate_dump,
)


def arrays(record, dim):
    x = np.asarray(record["X"], dtype=np.float64)
    y = np.asarray(record["y_maximize"], dtype=np.float64)
    if x.ndim != 2 or x.shape != (len(y), dim) or len(y) < 2:
        raise ValueError("Invalid FSBO task shapes or fewer than two observations")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite FSBO data")
    return x, y


def cmd_meta_train(args):
    import torch
    from fsbo_modules import FSBO

    dump = json.loads(Path(args.dump).read_text())
    task_ids = validate_dump(dump)
    train_ids, valid_ids = partition_tasks(task_ids, args.validation_tasks)
    data = {tid: arrays(dump["tasks"][tid], dump["dim"]) for tid in task_ids}
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    state_path = ckpt / "training_state.pt"
    if not args.resume and any((ckpt / name).exists() for name in ("weights", "meta.json", "training_state.pt")):
        raise ValueError("Checkpoint directory already contains a run; use --resume or a new directory")
    config = {"hidden_size": [128, 128], "kernel": "rbf", "ard": False,
              "lr": args.lr, "batch_size": args.batch_size, "augmentation": "paper_eq10_11"}
    identity = {"method": "fsbo_paper", "format_version": 2, "group": dump["group"],
                "schema_fingerprint": dump["schema_fingerprint"], "dim": dump["dim"],
                "dump_sha256": sha256_file(args.dump), "train_tasks": train_ids,
                "validation_tasks": valid_ids, "seed": args.seed, "config": config}
    train_data = {t: {"X": data[t][0], "y": data[t][1][:, None]} for t in train_ids}
    valid_data = {t: {"X": data[t][0], "y": data[t][1][:, None]} for t in valid_ids}
    model = FSBO(train_data=train_data, valid_data=valid_data, checkpoint_path=str(ckpt),
                 hidden_size=config["hidden_size"], kernel="rbf", batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lower = min(data[t][1].min() for t in train_ids)
    upper = max(data[t][1].max() for t in train_ids)
    best, start, best_step = float("inf"), 0, None
    if args.resume:
        saved = torch.load(str(state_path), map_location=model.device)
        if saved["identity"] != identity:
            raise ValueError("Cannot resume with different data, schema, split, or model settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        best, start, best_step = saved["best"], saved["step"], saved["best_step"]
    if args.steps <= start:
        raise ValueError("--steps must exceed the number of already completed updates")

    # Fixed disjoint validation supports/queries keep checkpoint comparisons meaningful.
    valid_rng = np.random.RandomState(args.seed + 1)
    validation = []
    for tid in valid_ids:
        x, y = data[tid]
        order = valid_rng.permutation(len(y))
        n_support = min(args.batch_size, max(1, len(y) // 2))
        support = order[:n_support]
        query = order[n_support:n_support + args.batch_size]
        validation.append((x[support], y[support], x[query], y[query]))

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=model.device)

    def validate():
        losses = []
        model.feature_extractor.eval()
        model.model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            for sx, sy, qx, qy in validation:
                model.model.set_train_data(model.feature_extractor(tensor(sx)), tensor(sy), strict=False)
                posterior = model.likelihood(model.model(model.feature_extractor(tensor(qx))))
                losses.append(float(-posterior.log_prob(tensor(qy)) / len(qy)))
        result = float(np.mean(losses))
        if not np.isfinite(result):
            raise RuntimeError("Non-finite FSBO validation loss; no successful checkpoint")
        return result

    atomic_json(ckpt / "meta.json", dict(identity, status="training", steps_completed=start))
    for step in range(start + 1, args.steps + 1):
        task = train_ids[rng.randint(len(train_ids))]
        x, y = data[task]
        indices = rng.choice(len(y), size=min(args.batch_size, len(y)), replace=False)
        if lower < upper:
            lo, hi = np.sort(rng.uniform(lower, upper, size=2))
            while hi - lo <= np.finfo(float).eps * max(1.0, abs(lo), abs(hi)):
                lo, hi = np.sort(rng.uniform(lower, upper, size=2))
        else:
            lo, hi = lower, lower + 1.0
        labels = tensor((y[indices] - lo) / (hi - lo))
        model.feature_extractor.train()
        model.model.train()
        model.likelihood.train()
        optimizer.zero_grad()
        features = model.feature_extractor(tensor(x[indices]))
        model.model.set_train_data(features, labels, strict=False)
        loss = -model.mll(model.model(features), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite FSBO training loss")
        loss.backward()
        optimizer.step()
        if step % args.validate_every == 0 or step == args.steps:
            validation_loss = validate()
            if validation_loss < best:
                best, best_step = validation_loss, step
                temporary = ckpt / "weights.tmp"
                model.save_checkpoint(str(temporary))
                os.replace(str(temporary), str(ckpt / "weights"))
            saved = {"identity": identity, "step": step, "best": best, "best_step": best_step,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng": rng.get_state(), "torch_rng": torch.get_rng_state()}
            temporary = ckpt / "training_state.tmp"
            torch.save(saved, str(temporary))
            os.replace(str(temporary), str(state_path))
            atomic_json(ckpt / "meta.json", dict(identity, status="training", steps_completed=step,
                                                 requested_steps=args.steps, best_step=best_step,
                                                 validation_loss=validation_loss, best_validation_loss=best))
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "validation_loss": validation_loss, "best_step": best_step}), flush=True)
    weights = ckpt / "weights"
    if not weights.is_file():
        raise RuntimeError("Training finished without a checkpoint")
    model.load_checkpoint(str(weights))
    validate()
    atomic_json(ckpt / "meta.json", dict(identity, status="complete", steps_completed=args.steps,
                                        best_step=best_step, validation_loss=best,
                                        weights_sha256=sha256_file(weights)))
    return 0


def cmd_suggest(args):
    import copy
    import torch
    from fsbo_modules import DeepKernelGP
    from fsbo_utils import EI, totorch

    class StrictDeepKernelGP(DeepKernelGP):
        def train(self):
            self.load_checkpoint(str(Path(self.checkpoint) / "weights"))
            self.model.train()
            self.feature_extractor.train()
            self.likelihood.train()
            optimizer = torch.optim.Adam([
                {"params": self.model.parameters()}, {"params": self.feature_extractor.parameters()}
            ], lr=self.lr)
            best, previous, patience, best_weights = float("inf"), float("inf"), 0, None
            for _ in range(self.epochs):
                optimizer.zero_grad()
                features = self.feature_extractor(self.X_obs)
                self.model.set_train_data(features, self.y_obs, strict=False)
                loss = -self.mll(self.model(features), self.y_obs)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite FSBO target adaptation loss")
                loss.backward()
                optimizer.step()
                value = float(loss.detach())
                if value < best:
                    best, best_weights = value, copy.deepcopy(self.state_dict())
                patience = patience + 1 if np.allclose(value, previous, atol=self.loss_tol) else 0
                previous = value
                if patience > self.max_patience:
                    break
            if best_weights is None:
                raise RuntimeError("FSBO target adaptation performed no successful updates")
            self.load_state_dict(best_weights)

    req = json.loads(Path(args.request).read_text())
    ckpt = Path(args.ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    if meta.get("status") != "complete" or meta.get("method") != "fsbo_paper":
        raise ValueError("A completed paper-FSBO checkpoint is required")
    if sha256_file(ckpt / "weights") != meta["weights_sha256"]:
        raise ValueError("FSBO checkpoint digest mismatch")
    if req.get("schema_fingerprint") != meta["schema_fingerprint"]:
        raise ValueError("Target and FSBO checkpoint schemas differ")
    x = np.asarray(req["X_obs"], dtype=float)
    y = np.asarray(req["y_obs"], dtype=float)
    pending = np.asarray(req["X_pen"], dtype=float)
    if x.shape != (len(y), meta["dim"]) or len(y) < 2 or pending.ndim != 2 or pending.shape[1] != meta["dim"]:
        raise ValueError("Invalid target data or candidate shape")
    if not all(np.isfinite(a).all() for a in (x, y, pending)) or not len(pending):
        raise ValueError("Non-finite target data or empty candidate pool")
    model = StrictDeepKernelGP(input_size=meta["dim"], log_dir=str(Path(args.out).with_suffix(".log")),
                         hidden_size=meta["config"]["hidden_size"], kernel="rbf", ard=False,
                         seed=int(req.get("seed", 1)), checkpoint=str(ckpt), load_model=True,
                         epochs=int(req.get("finetune_epochs", 100)), verbose=False)
    scaling = req.get("target_scaling", "raw")
    if scaling == "minmax":
        # Same per-task MinMaxScaler as upstream meta-training (fsbo_modules.get_batch).
        lo, hi = float(y.min()), float(y.max())
        y = (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)
    elif scaling != "raw":
        raise ValueError("Unknown target_scaling: %r" % scaling)
    model.X_obs = totorch(x, model.device)
    model.y_obs = totorch(y, model.device).reshape(-1)
    model.train()
    mu, std = model.predict(totorch(pending, model.device))
    scores = EI(max(y), mu, std)
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite FSBO expected improvement")
    index = np.argmax(scores)
    atomic_json(args.out, {"index": int(index), "checkpoint_sha256": meta["weights_sha256"],
                          "acquisition": acquisition_diagnostics("EI", scores),
                          "posterior_mean": mu.reshape(-1).tolist(),
                          "posterior_stddev": std.reshape(-1).tolist(), "incumbent": float(max(y)),
                          "target_scaling": scaling})
    return 0


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    train = sub.add_parser("meta-train")
    train.add_argument("--dump", required=True)
    train.add_argument("--ckpt", required=True)
    train.add_argument("--validation-tasks", required=True)
    train.add_argument("--steps", type=positive, required=True)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--batch-size", type=positive, default=50)
    train.add_argument("--validate-every", type=positive, default=1)
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--resume", action="store_true")
    suggest = sub.add_parser("suggest")
    suggest.add_argument("--ckpt", required=True)
    suggest.add_argument("--request", required=True)
    suggest.add_argument("--out", required=True)
    args = parser.parse_args()
    return cmd_meta_train(args) if args.cmd == "meta-train" else cmd_suggest(args)


if __name__ == "__main__":
    sys.exit(main())
