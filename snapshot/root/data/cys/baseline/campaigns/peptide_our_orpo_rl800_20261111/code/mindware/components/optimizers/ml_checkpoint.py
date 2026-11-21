"""Durable ML optimizer rounds, including a proposal awaiting evaluation."""

import copy
import csv
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np


DEPENDENCIES = {
    "evaluator", "config_space", "KD", "logger", "params", "optimizer_params",
    "llm_client", "openai_client", "objective_function", "_experience_bank", "_ml_session",
    "extractor_client", "resolver", "tosfit_runtime", "bora_assistant", "_expected_bolt_identity",
}
KD_FIELDS = (
    "bo_history", "bo_history_mab", "configs", "perfs", "configs_mab", "perfs_mab",
    "param_tuning_history", "algo_selection_history",
)


def atomic_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = stream.name
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def progress(rows, budget):
    ids = [str(row["iter_idx"]) for row in rows]
    if len(ids) != len(set(ids)) or len(rows) > budget:
        raise ValueError("Duplicate trial records or exceeded evaluation budget")
    finite = sum(math.isfinite(float(row["perf"])) for row in rows)
    exhausted = len(rows) == budget
    return {"attempts": len(rows), "successful": finite, "failed": len(rows) - finite,
            "budget_exhausted": exhausted, "clean_complete": exhausted and finite == budget,
            "status": ("complete" if finite == budget else "complete_with_failures")
            if exhausted else "incomplete"}


def capture_rng(space):
    import torch
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state(), "space": space.random.get_state()}
    if torch.cuda.is_initialized():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(value, space):
    import torch
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    space.random.set_state(value["space"])
    if "cuda" in value:
        torch.cuda.set_rng_state_all(value["cuda"])


class MLSession:
    def __init__(self, optimizer, directory):
        self.optimizer = optimizer
        self.directory = Path(directory)
        self.path = self.directory / "checkpoint.pkl"
        self.identity = os.environ["ML_RUN_ID"]
        self.pending = None

    def save(self, pending=None):
        opt = self.optimizer
        extra = opt._checkpoint_state(self.directory) if hasattr(opt, "_checkpoint_state") else None
        state = {k: v for k, v in vars(opt).items() if k not in DEPENDENCIES}
        # This callback points back to the optimizer and its live model client.
        if state.get("_explore_policy") is not None:
            state["_explore_policy"] = copy.copy(state["_explore_policy"])
            state["_explore_policy"].tpe_peek = None
        kd = {key: getattr(opt.KD, key) for key in KD_FIELDS if hasattr(opt.KD, key)}
        if "algo_selection_history" in kd:
            kd["algo_selection_history"] = dict(kd["algo_selection_history"])
        rows = state.get("traj_rows", [])
        summary = progress(rows, int(opt.evaluation_num_limit))
        payload = {"version": 1, "identity": self.identity, "optimizer": state, "kd": kd,
                   "runtime_policy_id": os.environ.get("ML_RUNTIME_POLICY_ID"),
                   "rng": capture_rng(opt.config_space), "pending": pending,
                   "progress": summary, "trajectory": opt.traj_csv_path, "extra": extra,
                   "evaluator_paths": {k: v for k, v in vars(opt.evaluator).items()
                                       if k in ("output_dir", "timestamp", "model_dir", "data_path")}}
        client = getattr(opt, 'llm_client', None)
        if hasattr(client, 'inference_state'):
            payload['inference_state'] = client.inference_state()
        atomic_pickle(self.path, payload)
        if hasattr(opt, "_checkpoint_committed"):
            opt._checkpoint_committed(extra)
        self.pending = pending
        atomic_json(self.directory / "progress.json", {**summary, "identity": self.identity,
                    "runtime_policy_id": os.environ.get("ML_RUNTIME_POLICY_ID"),
                    "runtime_policy_revisions": getattr(opt, "runtime_policy_revisions", []),
                    "trajectory": opt.traj_csv_path, "pending": pending is not None})

    def restore(self):
        if not self.path.exists():
            return 0
        with self.path.open("rb") as stream:
            payload = pickle.load(stream)
        previous_identity = os.environ.get("ML_PREVIOUS_RUN_ID")
        migrating = payload.get("identity") != self.identity and payload.get("identity") == previous_identity
        if payload.get("version") != 1 or (payload.get("identity") != self.identity and not migrating):
            raise RuntimeError("Checkpoint belongs to a different run or protocol")
        opt = self.optimizer
        current_limits = {name: getattr(opt, name) for name in
                          ("per_run_time_limit", "time_limit", "evaluation_num_limit") if hasattr(opt, name)}
        opt.__dict__.update(payload["optimizer"])
        opt.__dict__.update(current_limits)
        if migrating:
            revisions = list(getattr(opt, "runtime_policy_revisions", []))
            revisions.append({"previous_identity": payload["identity"], "identity": self.identity,
                              "after_attempts": len(opt.traj_rows), **current_limits})
            opt.runtime_policy_revisions = revisions
            for row in opt.traj_rows:
                row.setdefault("evaluation_limit_seconds", payload["optimizer"].get("per_run_time_limit"))
                row.setdefault("evaluation_status", "SUCCESS" if math.isfinite(float(row["perf"])) else "FAILED")
                row.setdefault("runtime_policy_id", payload.get("runtime_policy_id") or
                               os.environ.get("ML_PREVIOUS_RUNTIME_POLICY_ID"))
        for name, value in payload["kd"].items():
            if name == "algo_selection_history":
                opt.KD.algo_selection_history.clear()
                opt.KD.algo_selection_history.update(value)
            else:
                setattr(opt.KD, name, value)
        if not getattr(opt, "target_only", False):
            opt.configs, opt.perfs = opt.KD.configs, opt.KD.perfs
        if getattr(opt, "_explore_policy", None) is not None:
            opt._explore_policy.tpe_peek = opt._peek_tpe_config
        opt.evaluator.__dict__.update(payload.get("evaluator_paths", {}))
        client = getattr(opt, "llm_client", None)
        remote = (client.restore_inference_state(payload.get('inference_state'), payload['rng'].get('cuda'))
                  if hasattr(client, 'restore_inference_state') else False)
        if "cuda" in payload["rng"] and not remote and client is not None and hasattr(client, "_ensure_loaded"):
            client._ensure_loaded()
        if hasattr(opt, "_checkpoint_restore"):
            opt._checkpoint_restore(payload.get("extra"))
        restore_rng({k: v for k, v in payload['rng'].items() if k != 'cuda' or not remote}, opt.config_space)
        self.pending = payload["pending"]
        summary = progress(opt.traj_rows, int(opt.evaluation_num_limit))
        if summary != payload["progress"]:
            raise RuntimeError("Checkpoint observation counts do not match")
        self.export_trajectory()
        atomic_json(self.directory / "progress.json", {**summary, "identity": self.identity,
                    "runtime_policy_id": os.environ.get("ML_RUNTIME_POLICY_ID"),
                    "runtime_policy_revisions": getattr(opt, "runtime_policy_revisions", []),
                    "trajectory": opt.traj_csv_path, "pending": self.pending is not None})
        if os.environ.get('ML_COMPACT_ARTIFACTS') == '1':
            from .run_artifact_store import compact_ml
            compact_ml(self.directory, summary['attempts'])
        return summary["attempts"]

    def export_trajectory(self):
        rows = self.optimizer.traj_rows
        path = Path(self.optimizer.traj_csv_path)
        if not rows:
            # A CSV ahead of the committed checkpoint is an uncommitted export.
            if path.exists():
                path.unlink()
            return
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)


def attach(optimizer):
    directory = os.environ.get("ML_SESSION_DIR")
    if not directory:
        return 0
    session = MLSession(optimizer, directory)
    optimizer._ml_session = session
    return session.restore()


def pending(optimizer):
    session = getattr(optimizer, "_ml_session", None)
    return session.pending if session else None


def prepare(optimizer, proposal):
    session = getattr(optimizer, "_ml_session", None)
    if session:
        if session.pending is None:
            session.save(proposal)
        os.environ["ML_EVALUATION_ID"] = str(len(optimizer.traj_rows))


def commit(optimizer, completed=True):
    session = getattr(optimizer, "_ml_session", None)
    if session:
        session.save(None if completed else session.pending)
        session.export_trajectory()
        if os.environ.get('ML_COMPACT_ARTIFACTS') == '1':
            from .run_artifact_store import compact_ml
            compact_ml(session.directory, len(optimizer.traj_rows))
