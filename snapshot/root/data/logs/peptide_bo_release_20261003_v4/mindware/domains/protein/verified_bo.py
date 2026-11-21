"""Audited peptide BO protocol, built on the original BOLT LOLBO implementation."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT = "peptide-lolbo-v3"
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
STATE_FIELDS = (
    "train_x", "train_y", "train_z", "train_c", "top_k_scores", "top_k_xs",
    "top_k_zs", "top_k_cs", "best_x_seen", "best_score_seen", "tr_state",
    "initial_model_training_complete", "progress_fails_since_last_e2e",
    "tot_num_e2e_updates", "new_best_found",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any, *, rotate: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if rotate and path.exists():
        previous = path.with_name(path.stem + ".prev" + path.suffix)
        link = previous.with_name(previous.name + ".tmp")
        link.unlink(missing_ok=True)
        os.link(path, link)
        os.replace(link, previous)
    os.replace(temporary, path)


def cpu_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def capture_rng() -> dict:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        if len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("Checkpoint and worker GPU counts differ")
        torch.cuda.set_rng_state_all(value["cuda"])


def snapshot(state: Any, identity: dict, progress: dict, rows: list) -> dict:
    return dict(format=FORMAT, identity=identity, progress=copy.deepcopy(progress),
                rows=copy.deepcopy(rows), rng=capture_rng(),
                state={k: cpu_copy(getattr(state, k)) for k in STATE_FIELDS if hasattr(state, k)},
                vae=cpu_copy(state.objective.vae.state_dict()),
                model=cpu_copy(state.model.state_dict()),
                constraints=[cpu_copy(m.state_dict()) for m in state.c_models],
                scores=copy.deepcopy(state.objective.xs_to_scores_dict),
                oracle_calls=int(state.objective.num_calls))


def restore(state: Any, saved: dict, identity: dict) -> tuple[dict, list]:
    if saved.get("format") != FORMAT or saved.get("identity") != identity:
        raise ValueError("Checkpoint task, assets, code, or protocol mismatch; legacy resume is forbidden")
    if len(saved["constraints"]) != len(state.c_models):
        raise ValueError("Checkpoint constraint model count differs")
    state.objective.vae.load_state_dict(saved["vae"], strict=True)
    state.model.load_state_dict(saved["model"], strict=True)
    for model, weights in zip(state.c_models, saved["constraints"]):
        model.load_state_dict(weights, strict=True)
    for name, value in saved["state"].items():
        setattr(state, name, copy.deepcopy(value))
    state.objective.xs_to_scores_dict = copy.deepcopy(saved["scores"])
    state.objective.num_calls = saved["oracle_calls"]
    state.objective.vae.eval()
    # Switching through train clears GPyTorch's variational prediction caches.
    state.model.train()
    state.model.eval()
    for model in state.c_models:
        model.train()
        model.eval()
    validate_state(state)
    restore_rng(saved["rng"])
    return copy.deepcopy(saved["progress"]), copy.deepcopy(saved["rows"])


def validate_state(state: Any) -> None:
    n = len(state.train_x)
    if state.train_y.shape != (n, 1) or state.train_z.shape != (n, 256):
        raise ValueError("Peptide sequences, outcomes, and 256-dimensional latents are misaligned")
    if state.train_c.shape != (n, 1):
        raise ValueError("Peptide constraints are misaligned")
    for value in (state.train_y, state.train_z, state.train_c):
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite peptide GP data")


def validate_sequence(sequence: str, max_length: int = 30, *, limit_cysteine: bool = True,
                      allow_unknown: bool = False) -> None:
    alphabet = ALPHABET | {"X"} if allow_unknown else ALPHABET
    if (not isinstance(sequence, str) or not sequence or len(sequence) > max_length
            or set(sequence) - alphabet or (limit_cysteine and sequence.count("C") > 1)):
        raise ValueError("Invalid peptide alphabet, length, or cysteine count: %r" % sequence)


def similarity(sequence: str, reference: str) -> float:
    from Levenshtein import distance
    return 1.0 - distance(sequence, reference) / len(reference)


def required_decoder_edits(reference: str) -> int:
    return max(0, reference.count("C") - 1) + sum(aa not in ALPHABET for aa in reference)


def encode(objective: Any, sequences: list[str], batch_size: int = 64) -> torch.Tensor:
    objective.vae.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            z, _ = objective.vae_forward(batch)
            z = z.detach().cpu().reshape(len(batch), -1)
            if z.shape[1] != 256 or not torch.isfinite(z).all():
                raise ValueError("VAE produced invalid latents; padding or truncation is forbidden")
            chunks.append(z)
    return torch.cat(chunks)


def unique_top_k(state: Any) -> None:
    feasible = torch.all(state.train_c <= 0, dim=1)
    order = torch.argsort(state.train_y[:, 0], descending=True).tolist()
    indices, seen = [], set()
    for i in order:
        sequence = str(state.train_x[i])
        if feasible[i] and sequence not in seen:
            indices.append(i)
            seen.add(sequence)
        if len(indices) == state.k:
            break
    if not indices:
        raise ValueError("No feasible peptide incumbent")
    state.top_k_scores = [float(state.train_y[i, 0]) for i in indices]
    state.top_k_xs = [str(state.train_x[i]) for i in indices]
    state.top_k_zs = [state.train_z[i].unsqueeze(0) for i in indices]
    state.top_k_cs = [state.train_c[i].unsqueeze(0) for i in indices]
    state.best_x_seen = state.top_k_xs[0]
    state.best_score_seen = state.top_k_scores[0]


def reencode_after_update(state: Any) -> None:
    before_x = list(state.train_x)
    before_y, before_c = state.train_y.clone(), state.train_c.clone()
    before_calls = int(state.objective.num_calls)
    before_scores = dict(state.objective.xs_to_scores_dict)
    state.update_models_e2e()
    state.train_z = encode(state.objective, before_x)
    unique_top_k(state)
    initial_epochs = state.init_n_epochs
    state.initial_model_training_complete = False
    state.init_n_epochs = state.num_update_epochs
    try:
        state.update_surrogate_model()
    finally:
        state.init_n_epochs = initial_epochs
    if (list(state.train_x) != before_x or not torch.equal(state.train_y, before_y)
            or not torch.equal(state.train_c, before_c)
            or int(state.objective.num_calls) != before_calls
            or state.objective.xs_to_scores_dict != before_scores):
        raise RuntimeError("VAE update changed observations or consumed oracle evaluations")
    validate_state(state)


def propose(state: Any, count: int) -> tuple[torch.Tensor, list[str]]:
    from lolbo.utils.bo_utils.turbo import generate_batch
    feasible = torch.all(state.train_c <= 0, dim=1)
    if not feasible.any():
        raise ValueError("Constrained acquisition has no feasible center")
    # Upstream uses Y.argmax() to choose the center even for constrained TS.
    # Its GPs still contain all observations; only the center candidates are filtered.
    with torch.no_grad():
        z = generate_batch(state=state.tr_state, model=state.model,
            X=state.train_z[feasible], Y=state.train_y[feasible], batch_size=count,
            acqf=state.acq_func, constraint_model_list=state.c_models)
        sequences = list(state.objective.vae_decode(z))
    if len(sequences) != count or z.shape != (count, 256) or not torch.isfinite(z).all():
        raise ValueError("Acquisition produced a malformed batch")
    return z, sequences


def score_batch(objective: Any, sequences: list[str]) -> dict:
    for sequence in sequences:
        validate_sequence(sequence)
    cache = objective.xs_to_scores_dict
    new = list(dict.fromkeys(x for x in sequences if x not in cache))
    constraints = objective.compute_constraints(sequences).cpu()
    if constraints.shape != (len(sequences), 1) or not torch.isfinite(constraints).all():
        raise ValueError("Invalid oracle constraints")
    with torch.no_grad():
        values = np.asarray(objective.query_oracle(new), dtype=float) if new else np.empty(0)
    if values.shape != (len(new),) or not np.isfinite(values).all():
        raise ValueError("Oracle returned missing or non-finite scores")
    cache.update(zip(new, values.tolist()))
    objective.num_calls += len(new)
    seen = set(cache) - set(new)
    hits = []
    for sequence in sequences:
        hits.append(sequence in seen)
        seen.add(sequence)
    return dict(sequences=sequences, scores=[float(cache[x]) for x in sequences],
                constraints=constraints, cache_hits=hits, new_scores=dict(zip(new, values.tolist())),
                fresh_evaluations=len(new), rng=capture_rng())


def next_batch_size(progress: dict, budget: int, batch_size: int) -> int:
    return max(0, min(batch_size, budget - int(progress["proposals"])))


def collect_pools(recipe: dict) -> tuple[dict, list]:
    manifest = json.loads(Path(recipe["pool_manifest"]).read_text())
    stride = manifest["num_mutations"]
    if (manifest["n_refs"] != len(manifest["per_ref"])
            or manifest["rows"] != stride * manifest["n_refs"]):
        raise ValueError("Initialization manifest dimensions disagree")
    selected, audit = {}, []
    with Path(manifest["init_path"]).open() as xs, Path(manifest["scores_path"]).open() as ys:
        for offset, meta in enumerate(manifest["per_ref"]):
            index = manifest["ref_start"] + offset
            if meta["ref_index"] != index:
                raise ValueError("Initialization manifest reference order changed")
            sequences = [line.strip() for line in itertools.islice(xs, stride)]
            scores = [float(line.strip()) for line in itertools.islice(ys, stride)]
            if len(sequences) != stride or len(scores) != stride:
                raise ValueError("Initialization pool is incomplete")
            eligible = []
            for i, (sequence, score) in enumerate(zip(sequences, scores)):
                validate_sequence(sequence, limit_cysteine=False, allow_unknown=True)
                if not math.isfinite(score) or similarity(sequence, meta["reference"]) < recipe["similarity"]:
                    raise ValueError("Initialization contains non-finite or infeasible points")
                if recipe["init_policy"] == "legacy_first" or (sequence.count("C") <= 1 and "X" not in sequence):
                    eligible.append(i)
            indices = eligible[:recipe["initialization"]]
            if len(indices) != recipe["initialization"]:
                raise ValueError("Task pep_%03d has only %d eligible initializations" % (index, len(eligible)))
            task = "pep_%03d" % index
            selected[task] = dict(reference=meta["reference"], sequences=[sequences[i] for i in indices],
                                 scores=[scores[i] for i in indices], source_row_indices=indices)
            audit.append(dict(task=task, reference=meta["reference"], available=len(eligible),
                selected=len(indices), distinct=len({sequences[i] for i in indices}),
                replaced_prefix_rows=sum(i >= recipe["initialization"] for i in indices),
                selected_multic=sum(sequences[i].count("C") > 1 for i in indices),
                selected_unknown=sum("X" in sequences[i] for i in indices),
                min_similarity=min(similarity(sequences[i], meta["reference"]) for i in indices)))
        if xs.readline() or ys.readline():
            raise ValueError("Initialization pool has trailing rows outside manifest")
    return selected, audit


def load_pool(recipe: dict, task: str) -> tuple[str, list[str], list[float]]:
    pool = json.loads(Path(recipe["prepared_pool"]).read_text())[task]
    reference, sequences, values = pool["reference"], pool["sequences"], pool["scores"]
    if len(sequences) != recipe["initialization"] or len(values) != len(sequences):
        raise ValueError("Prepared initialization pool is incomplete")
    for sequence, value in zip(sequences, values):
        validate_sequence(sequence, limit_cysteine=recipe["init_policy"] != "legacy_first",
                          allow_unknown=recipe["init_policy"] == "legacy_first")
        if not math.isfinite(value) or similarity(sequence, reference) < recipe["similarity"] - 1e-8:
            raise ValueError("Initialization contains non-finite or infeasible points")
    return reference, sequences, values


def create_state(recipe: dict, task: str, saved: dict | None = None) -> Any:
    upstream = Path(recipe["upstream"])
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    from apex_oracle.refseqs import REFERENCE_SEQUENCE
    from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
    from lolbo.lolbo import LOLBOState
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    reference, sequences, scores = load_pool(recipe, task)
    index = int(task.removeprefix("pep_"))
    if REFERENCE_SEQUENCE[index] != reference:
        raise ValueError("Initialization reference and BOLT constraint reference differ")
    objective = ApexConstrainedDiverseObjective(
        task_id="apex", task_specific_args=["bacteria_0"], path_to_vae_statedict=recipe["vae"],
        max_string_length=30, constraint_function_ids=["similarity"],
        constraint_thresholds=[recipe["similarity"]], constraint_types=[index])
    if saved is None:
        z = encode(objective, sequences)
    else:
        z = saved["state"]["train_z"][:len(sequences)].clone()
    state = LOLBOState(objective=objective, surrogate_type="gp_dkl", train_x=list(sequences),
        train_y=torch.tensor(scores, dtype=torch.float32).reshape(-1, 1), train_z=z,
        train_c=objective.compute_constraints(sequences), minimize=False, k=1000,
        num_update_epochs=recipe["gp_update_epochs"], init_n_epochs=recipe["gp_initial_epochs"],
        learning_rte=recipe["learning_rate"],
        bsz=recipe["batch_size"], acq_func="ts", verbose=False)
    unique_top_k(state)
    validate_state(state)
    return state
