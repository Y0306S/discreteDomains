"""Strict initialization and budget helpers for the new, isolated peptide campaign."""

from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path
import shutil
import sys

import torch

from mindware.domains.protein import verified_bo as bo

FORMAT = "peptide-lolbo-v5"
REPLACEMENTS = {"pep_049": ("pep_1000", "KTLKIIRLLF"),
                "pep_073": ("pep_1001", "RQKNHGIHFRVLAKALR"),
                "pep_082": ("pep_1002", "RMARNLVRYVQGLKKKKVI")}


def mutants(reference, count, existing=(), seed=0):
    if bo.required_decoder_edits(reference) > len(reference) // 4:
        raise ValueError("Reference has no decoder-compatible neighborhood")
    rng = random.Random(str(seed) + ":" + reference)
    seen, result = set(existing), []
    alphabet = sorted(bo.ALPHABET)
    unknown = {i for i, c in enumerate(reference) if c not in bo.ALPHABET}
    cysteines = [i for i, c in enumerate(reference) if c == "C"]
    limit = len(reference) // 4
    for _ in range(max(10000, count * 1000)):
        if len(result) == count:
            return result
        keep = rng.choice(cysteines) if cysteines else None
        mandatory = unknown | {i for i in cysteines if i != keep}
        edits = rng.randint(max(1, len(mandatory)), limit)
        optional = rng.sample([i for i in range(len(reference)) if i not in mandatory], edits - len(mandatory))
        chars = list(reference)
        for i in sorted(mandatory | set(optional)):
            choices = [a for a in alphabet if a != reference[i] and (i not in mandatory or a != "C")]
            chars[i] = rng.choice(choices)
        sequence = "".join(chars)
        if sequence in seen or sequence == reference or sequence.count("C") > 1:
            continue
        bo.validate_sequence(sequence)
        if bo.similarity(sequence, reference) < 0.75:
            continue
        seen.add(sequence)
        result.append(sequence)
    raise ValueError("Cannot construct requested distinct initialization mutants")


class CPUTransformer(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda" else node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            if node.args or node.keywords:
                raise ValueError("Unexpected CUDA call arguments")
            node.func.attr = "cpu"
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch" and node.func.attr == "load"):
            node.keywords = [k for k in node.keywords if k.arg not in ("map_location", "weights_only")]
            node.keywords += [ast.keyword("map_location", ast.Constant("cpu")),
                              ast.keyword("weights_only", ast.Constant(False))]
        return node


def make_cpu_upstream(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("lolbo", "uniref_vae", "apex_oracle", "your_tasks"):
        for path in (source / folder).rglob("*"):
            if "__pycache__" in path.parts:
                continue
            target = destination / path.relative_to(source)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.suffix == ".py":
                target.parent.mkdir(parents=True, exist_ok=True)
                tree = CPUTransformer().visit(ast.parse(path.read_text()))
                ast.fix_missing_locations(tree)
                target.write_text(ast.unparse(tree) + "\n")
            elif path.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(path.resolve())


def create_state(recipe, task, saved=None, remote_url=None):
    sys.path.insert(0, recipe["cpu_upstream"] if remote_url else recipe["upstream"])
    from lolbo.info_transformer_vae_objective import ApexConstrainedDiverseObjective
    from lolbo.lolbo import LOLBOState
    import os
    import numpy as np
    os.chdir(recipe["cpu_upstream"] if remote_url else recipe["upstream"])
    random.seed(recipe["seed"])
    np.random.seed(recipe["seed"])
    torch.manual_seed(recipe["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe["seed"])
    reference, sequences, scores = bo.load_pool(recipe, task)
    if reference != recipe["references"][task]:
        raise ValueError("Task/reference mismatch")
    objective = ApexConstrainedDiverseObjective(task_id="apex", task_specific_args=["bacteria_0"],
        path_to_vae_statedict=recipe["vae"], max_string_length=30)
    objective.compute_constraints = lambda xs: torch.tensor(
        [[recipe["similarity"] - bo.similarity(x, reference)] for x in xs], dtype=torch.float32)
    backend = None
    if remote_url:
        from mindware.domains.protein.peptide_rpc import RemoteObjective
        recipe_identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        backend = RemoteObjective(objective, remote_url, task + ":" + recipe_identity)
    z = bo.encode(objective, sequences) if saved is None else saved["state"]["train_z"][:len(sequences)].clone()
    state = LOLBOState(objective=objective, surrogate_type="gp_dkl", train_x=list(sequences),
        train_y=torch.tensor(scores, dtype=torch.float32).reshape(-1, 1), train_z=z,
        train_c=objective.compute_constraints(sequences), minimize=False, k=1000,
        num_update_epochs=recipe["gp_update_epochs"], init_n_epochs=recipe["gp_initial_epochs"],
        learning_rte=recipe["learning_rate"], bsz=recipe["batch_size"], acq_func="ts", verbose=False)
    bo.unique_top_k(state)
    bo.validate_state(state)
    return state, backend


def bounded_prefix(sequences, cache, remaining):
    seen = set(cache)
    accepted = []
    for sequence in sequences:
        if sequence not in seen:
            if remaining <= 0:
                break
            remaining -= 1
            seen.add(sequence)
        accepted.append(sequence)
    return accepted


def check_accounting(recipe, progress, rows, state=None):
    acquired = progress["proposals"]
    recentered = progress["recenter_proposals"]
    fresh = progress["fresh_evaluations"]
    if (acquired > recipe["proposals"] or fresh > recipe["fresh_budget"]
            or len(rows) != recipe["initialization"] + acquired + recentered
            or fresh + progress["cache_hits"] != acquired + recentered
            or progress["acquisition_fresh"] + progress["recenter_fresh"] != fresh):
        raise ValueError("Peptide v5 evaluation accounting diverged")
    if state is not None and state.objective.num_calls != fresh:
        raise ValueError("APEX call count differs from durable history")


def reencode(state):
    before_x, before_y = list(state.train_x), state.train_y.clone()
    before_calls = state.objective.num_calls
    state.train_z = bo.encode(state.objective, state.train_x)
    bo.unique_top_k(state)
    if state.train_x != before_x or not torch.equal(state.train_y, before_y) or state.objective.num_calls != before_calls:
        raise ValueError("Re-encoding changed observations or scores")
