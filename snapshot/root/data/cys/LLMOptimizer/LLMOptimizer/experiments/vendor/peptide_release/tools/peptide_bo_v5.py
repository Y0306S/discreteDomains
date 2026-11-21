#!/usr/bin/env python3
"""Fresh train800 peptide campaign with strict initialization and bounded native recentering."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import torch
import peptide_bo_campaign as legacy
from mindware.domains.protein import verified_bo as bo
from mindware.domains.protein import peptide_v5 as protocol

bo.FORMAT = protocol.FORMAT
UPSTREAM = Path("/root/data/cys/BOLT/optimization/peptides")
VAE = UPSTREAM / "uniref_vae/saved_models/dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118_pretrained_authors.pkl"


def prepare(args):
    campaign = args.campaign.resolve()
    if (campaign / "recipe.json").exists():
        raise ValueError("Use a new campaign; original recipes are immutable")
    campaign.mkdir(parents=True, exist_ok=True)
    split_path = Path("/root/blockdata/llmoptimizer_benchmark/docs/peptide_split_800_200.json")
    split = json.loads(split_path.read_text())
    bo.atomic_json(campaign / "original_split.json", split)
    refs_path = UPSTREAM / "apex_oracle/refseqs.py"
    refs = ast.literal_eval(ast.parse(refs_path.read_text()).body[0].value)
    references = {f"pep_{i:03d}": seq for i, seq in enumerate(refs)}
    train, test = list(split["train_ids"]), list(split["test_ids"])
    for old, (new, reference) in protocol.REPLACEMENTS.items():
        train[train.index(old)] = new
        references[new] = reference
    tasks = train + test
    source = UPSTREAM / "apex_oracle/init_data/generated/refs_0_1000_manifest.json"
    manifest = json.loads(source.read_text())
    for key in ("init", "scores"):
        if bo.digest(Path(manifest[key + "_path"])) != manifest[key + "_sha256"]:
            raise ValueError("Original initialization checksum mismatch")
    selected = {task: dict(reference=references[task], sequences=[], scores=[], sources=[]) for task in tasks}
    stride = manifest["num_mutations"]
    with Path(manifest["init_path"]).open() as xs, Path(manifest["scores_path"]).open() as ys:
        for i, meta in enumerate(manifest["per_ref"]):
            if meta["reference"] != refs[i] or meta["ref_index"] != i:
                raise ValueError("Original pool reference mismatch")
            task = f"pep_{i:03d}"
            for offset in range(stride):
                sequence, score = xs.readline().strip(), float(ys.readline())
                if task not in selected or len(selected[task]["sequences"]) >= args.initialization:
                    continue
                if sequence in selected[task]["sequences"]:
                    continue
                try:
                    bo.validate_sequence(sequence)
                except ValueError:
                    continue
                if not math.isfinite(score) or bo.similarity(sequence, references[task]) < 0.75:
                    raise ValueError("Original score or reference similarity invalid")
                selected[task]["sequences"].append(sequence)
                selected[task]["scores"].append(score)
                selected[task]["sources"].append(dict(kind="measured_original", row=i * stride + offset))
        if xs.readline() or ys.readline():
            raise ValueError("Unexpected original pool rows")
    from mindware.domains.protein.peptide_rpc import RemoteObjective
    from types import SimpleNamespace
    scorer = RemoteObjective(SimpleNamespace(vae_forward=None), args.server, "initialization:" + str(campaign))
    new_count = 0
    performed_calls = 0
    reused = json.loads(Path(args.reuse_initialization).read_text()) if args.reuse_initialization else None
    for task in tasks:
        pool = selected[task]
        missing = args.initialization - len(pool["sequences"])
        if reused is not None:
            previous = reused[task]
            n = len(pool["sequences"])
            if (previous["reference"] != pool["reference"] or len(previous["sequences"]) != args.initialization
                    or previous["sequences"][:n] != pool["sequences"] or previous["scores"][:n] != pool["scores"]):
                raise ValueError("Resumed preparation initialization mismatch")
            for x, y in zip(previous["sequences"], previous["scores"]):
                bo.validate_sequence(x)
                if bo.similarity(x, pool["reference"]) < 0.75 or not math.isfinite(y):
                    raise ValueError("Resumed initialization is invalid")
            selected[task] = previous
            new_count += missing
            continue
        if missing:
            generated = protocol.mutants(pool["reference"], missing, pool["sequences"], seed=0)
            scores = scorer.oracle(generated)
            performed_calls += len(generated)
            if len(scores) != missing or not all(math.isfinite(float(y)) for y in scores):
                raise ValueError("New initialization scoring failed")
            pool["sequences"].extend(generated)
            pool["scores"].extend(scores.tolist())
            pool["sources"].extend(dict(kind="new_compatible_mutant", seed=0) for _ in generated)
            new_count += missing
        if len(set(pool["sequences"])) != args.initialization:
            raise ValueError("Initialization not distinct")
    from Levenshtein import distance
    maximum = max(max(1-distance(references[a], references[b])/len(references[a]),
                      1-distance(references[a], references[b])/len(references[b])) for a in train for b in test)
    if maximum > 0.75:
        raise ValueError("Replacements violate train/test similarity boundary")
    bo.atomic_json(campaign / "initialization.json", selected)
    bo.atomic_json(campaign / "split.json", dict(train_ids=train, test_ids=test,
        references={task: references[task] for task in tasks}, replacements=protocol.REPLACEMENTS,
        original_split_sha256=bo.digest(split_path), maximum_cross_similarity=maximum))
    cpu_upstream = campaign / "upstream_cpu"
    if args.reuse_initialization and cpu_upstream.exists():
        import shutil
        shutil.rmtree(cpu_upstream)
    protocol.make_cpu_upstream(UPSTREAM, cpu_upstream)
    code_paths = [ROOT / "tools" / name for name in ["peptide_bo_v5.py", "peptide_bo_campaign.py", "peptide_vae_server.py"]]
    code_paths += [ROOT / "mindware/domains/protein" / name for name in ["verified_bo.py", "peptide_v5.py", "peptide_rpc.py"]]
    code_paths += [ROOT / "tests/domains/test_peptide_v5.py", ROOT / "tests/domains/test_verified_peptide_bo.py",
                   ROOT / "tools/check_peptide_v5.py"]
    code_paths += sorted(UPSTREAM.rglob("*.py")) + sorted(cpu_upstream.rglob("*.py"))
    assets = [VAE, source, refs_path, campaign / "original_split.json", Path(manifest["init_path"]), Path(manifest["scores_path"]),
        UPSTREAM / "uniref_vae/1mer_vocab.csv", campaign / "initialization.json", campaign / "split.json"]
    assets += sorted(p for p in (UPSTREAM / "apex_oracle/APEX_pathogen_models").iterdir() if p.is_file())
    recipe = dict(format=protocol.FORMAT, purpose="component_check" if args.component else "production",
        upstream=str(UPSTREAM), cpu_upstream=str(cpu_upstream), vae=str(VAE),
        prepared_pool=str(campaign / "initialization.json"), pool_manifest=str(source),
        initialization=args.initialization, proposals=args.proposals, fresh_budget=args.fresh_budget,
        batch_size=args.batch_size, e2e_every=args.e2e_every,
        e2e_policy="consecutive_non_improvement_joint_update_bounded_native_recenter",
        center_policy="best_observed_feasible", init_policy="first_valid_then_scored_compatible_topup",
        similarity=0.75, seed=1, gp_initial_epochs=20, gp_update_epochs=2, learning_rate=0.001,
        train=train, test=test, references={task: references[task] for task in tasks}, blocked_tasks={},
        budget_definition="maximum_acquisition_proposals_and_shared_fresh_acquisition_recenter_calls",
        initialization_new_calls=new_count, higher_is_better=True, oracle="negative_APEX_bacteria_0",
        initialization_calls_this_preparation=performed_calls,
        reused_initialization=(dict(path=str(Path(args.reuse_initialization).resolve()),
            sha256=bo.digest(Path(args.reuse_initialization))) if args.reuse_initialization else None),
        code={str(p): bo.digest(p) for p in code_paths}, assets={str(p): bo.digest(p) for p in assets},
        environment=legacy.versions(), prepared_at=time.time())
    bo.atomic_json(campaign / "recipe.json", recipe)
    print(json.dumps(dict(prepared=str(campaign), train=len(train), test=len(test), new_initialization_calls=new_count,
        initialization_calls_this_preparation=performed_calls,
        max_cross_similarity=maximum, launched=False)), flush=True)


def verify(campaign):
    recipe = json.loads((campaign / "recipe.json").read_text())
    if recipe["format"] != protocol.FORMAT:
        raise ValueError("Wrong peptide recipe")
    for section in ("code", "assets"):
        for path, expected in recipe[section].items():
            if bo.digest(Path(path)) != expected:
                raise ValueError("Pinned file changed: " + path)
    if legacy.versions() != recipe["environment"]:
        raise ValueError("Worker package versions differ")
    return recipe


def preflight(campaign):
    gate = json.loads((campaign / "preflight_passed.json").read_text())
    report_path = Path(gate["report"])
    report = json.loads(report_path.read_text())
    required = {"unit_tests", "strict_initialization", "real_acquisition", "vae_gradient_update",
        "bounded_recenter", "fresh_process_resume", "pending_without_rescore", "trust_region_contraction",
        "remote_cpu_worker", "archive"}
    if (gate["recipe_sha256"] != bo.digest(campaign / "recipe.json")
            or gate["report_sha256"] != bo.digest(report_path)
            or report["recipe_sha256"] != gate["recipe_sha256"] or not report["passed"]
            or not required <= report["checks"].keys() or not all(report["checks"][x] for x in required)):
        raise ValueError("New campaign has not passed its required preflight")


def archive(directory, identity):
    files = [directory / x for x in ("checkpoint.pt", "history.json", "progress.json")]
    extra = [p for p in directory.iterdir() if p not in files]
    known = {"checkpoint.prev.pt", "pending.pt", "checkpoint.pt.tmp", "checkpoint.prev.pt.tmp", "pending.pt.tmp",
             "history.json.tmp", "progress.json.tmp"}
    if any(not p.is_file() or p.name not in known for p in extra):
        raise ValueError("Unexpected files in completed task")
    manifest = dict(identity=identity, members={p.name: bo.digest(p) for p in files},
                    cleanup={p.name: bo.digest(p) for p in extra})
    target = directory.with_suffix(".tar.gz")
    temporary = target.with_name(target.name + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as packed:
        for path in files:
            packed.add(path, arcname=path.name, recursive=False)
        payload = json.dumps(manifest).encode()
        member = tarfile.TarInfo("manifest.json")
        member.size = len(payload)
        packed.addfile(member, io.BytesIO(payload))
    verify_archive(temporary, identity)
    os.replace(temporary, target)
    cleanup(directory, target, identity)
    return target


def verify_archive(path, identity):
    import hashlib
    with tarfile.open(path, "r:gz") as packed:
        names = {"checkpoint.pt", "history.json", "progress.json", "manifest.json"}
        members = packed.getmembers()
        if len(members) != 4 or {m.name for m in members} != names or not all(m.isfile() for m in members):
            raise ValueError("Invalid archive inventory")
        manifest = json.load(packed.extractfile("manifest.json"))
        if manifest["identity"] != identity:
            raise ValueError("Archive identity mismatch")
        for name, expected in manifest["members"].items():
            digest = hashlib.sha256()
            with packed.extractfile(name) as stream:
                for chunk in iter(lambda: stream.read(2**20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("Archive checksum mismatch")
        history = json.load(packed.extractfile("history.json"))
        if history["status"] != "complete" or history["identity"] != identity:
            raise ValueError("Archive is not complete")
        protocol.check_accounting(dict(initialization=history["initialization"], proposals=history["proposal_budget"],
            fresh_budget=history["fresh_budget"]), history, history["observations"])
        if history["proposals"] < history["proposal_budget"] and history["fresh_evaluations"] < history["fresh_budget"]:
            raise ValueError("Completed task did not exhaust either budget")
    return history


def cleanup(directory, target, identity):
    verify_archive(target, identity)
    with tarfile.open(target, "r:gz") as packed:
        manifest = json.load(packed.extractfile("manifest.json"))
    for name, expected in {**manifest["members"], **manifest["cleanup"]}.items():
        if Path(name).name != name:
            raise ValueError("Invalid cleanup path")
        path = directory / name
        if path.exists():
            if bo.digest(path) != expected:
                raise ValueError("Task changed after archive publication")
            path.unlink()
    if directory.exists():
        directory.rmdir()


def status(recipe, identity, state, progress, status_name):
    values = {k: v for k, v in progress.items() if k not in ("recenter_plan", "recenter_optimizer")}
    return dict(values, identity=identity, status=status_name,
        e2e_updates=state.tot_num_e2e_updates, best_feasible_mic=-float(state.best_score_seen),
        current_tr_length=state.tr_state.length)


def commit_pending(state, identity, progress, rows, pending, directory, recipe):
    commit_started = time.monotonic()
    path = directory / "pending.pt"
    if "scored" not in pending:
        pending["attempts"] = pending.get("attempts", 0) + 1
        bo.atomic_torch(path, pending)
        oracle_started = time.monotonic()
        pending["scored"] = bo.score_batch(state.objective, pending["sequences"])
        pending["oracle_seconds"] = time.monotonic() - oracle_started
        bo.atomic_torch(path, pending)
    else:
        scored = pending["scored"]
        state.objective.xs_to_scores_dict.update(scored["new_scores"])
        state.objective.num_calls += scored["fresh_evaluations"]
        bo.restore_rng(scored["rng"])
    scored, z = pending["scored"], pending["z"]
    if pending["phase"] == "recenter":
        state.model.train()
        params = [{"params": state.model.parameters(), "lr": recipe["learning_rate"]}]
        for model in state.c_models:
            model.train()
            params.append({"params": model.parameters(), "lr": recipe["learning_rate"]})
        optimizer = torch.optim.Adam(params, lr=recipe["learning_rate"])
        if progress["recenter_optimizer"] is not None:
            optimizer.load_state_dict(progress["recenter_optimizer"])
        device = next(state.model.parameters()).device
        y = torch.tensor(scored["scores"], dtype=torch.float32, device=device)
        loss = -state.mll(state.model(z.to(device)), y)
        for i, model in enumerate(state.c_models):
            loss = loss - state.c_mlls[i](model(z.to(device)), scored["constraints"][:, i].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
        optimizer.step()
        progress["recenter_optimizer"] = bo.cpu_copy(optimizer.state_dict())
        state.model.eval()
        for model in state.c_models:
            model.eval()
        progress["recenter_plan"]["cursor"] = pending["next_cursor"]
    acquired = pending["phase"] == "acquisition"
    state.update_next(z, torch.tensor(scored["scores"], dtype=torch.float32), scored["sequences"],
                      scored["constraints"], acquisition=acquired)
    bo.unique_top_k(state)
    n, fresh = len(scored["sequences"]), scored["fresh_evaluations"]
    progress["proposals" if acquired else "recenter_proposals"] += n
    progress["acquisition_fresh" if acquired else "recenter_fresh"] += fresh
    progress["fresh_evaluations"] += fresh
    progress["cache_hits"] += sum(scored["cache_hits"])
    if acquired:
        progress["rounds"] += 1
        if state.tr_state.restart_triggered:
            state.initialize_tr_state()
    progress["events"] += 1
    progress["seconds"] += pending.get("proposal_seconds", 0.0) + pending.get("oracle_seconds", 0.0)
    progress["seconds"] += time.monotonic() - commit_started
    if "oracle_started" in locals():
        progress["seconds"] -= pending["oracle_seconds"]
    for sequence, score, cached, constraint in zip(scored["sequences"], scored["scores"], scored["cache_hits"], scored["constraints"]):
        feasible = bo.similarity(sequence, recipe["references"][identity["task"]]) >= recipe["similarity"] - 1e-8
        if feasible != bool(constraint[0] <= 0):
            raise ValueError("Constraint and recorded feasibility differ")
        rows.append(dict(sequence=sequence, score=score, mic=-score, phase=pending["phase"],
            round=progress["rounds"], feasible=feasible, cached=cached, elapsed=progress["seconds"]))
    protocol.check_accounting(recipe, progress, rows, state)
    bo.validate_state(state)
    bo.atomic_torch(directory / "checkpoint.pt", bo.snapshot(state, identity, progress, rows), rotate=True)
    path.unlink()
    bo.atomic_json(directory / "progress.json", status(recipe, identity, state, progress, "running"))
    print(json.dumps(status(recipe, identity, state, progress, "running")), flush=True)


def run_task(args):
    campaign = args.campaign.resolve()
    recipe = verify(campaign)
    if recipe["purpose"] == "production":
        if not args.launch:
            raise ValueError("Production requires --launch")
        preflight(campaign)
    if args.task not in recipe["train"] + recipe["test"]:
        raise ValueError("Task outside the versioned split")
    sys.path.insert(0, recipe["cpu_upstream"] if args.server else recipe["upstream"])
    identity = dict(format=protocol.FORMAT, recipe_sha256=bo.digest(campaign / "recipe.json"), task=args.task)
    with legacy.lock(campaign / "locks" / (args.task + ".lock")):
        directory = campaign / "runs" / args.task
        packed = directory.with_suffix(".tar.gz")
        if packed.exists():
            cleanup(directory, packed, identity)
            return
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint, pending_path = directory / "checkpoint.pt", directory / "pending.pt"
        if not checkpoint.exists() and any(directory.iterdir()):
            raise ValueError("Task has data without a checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() else None
        state, backend = protocol.create_state(recipe, args.task, saved, args.server)
        stopping = False
        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        if saved is None:
            reference, sequences, scores = bo.load_pool(recipe, args.task)
            progress = dict(rounds=0, proposals=0, fresh_evaluations=0, cache_hits=0, acquisition_fresh=0,
                recenter_fresh=0, recenter_proposals=0, recenter_plan=None, recenter_optimizer=None, events=0, seconds=0.0)
            rows = [dict(sequence=x, score=float(y), mic=-float(y), phase="initialization", round=0,
                feasible=True, cached=True, elapsed=0.0) for x, y in zip(sequences, scores)]
            bo.atomic_torch(checkpoint, bo.snapshot(state, identity, progress, rows))
        else:
            progress, rows = bo.restore(state, saved, identity)
            if backend:
                backend.dirty()
        try:
            while progress["proposals"] < recipe["proposals"] and progress["fresh_evaluations"] < recipe["fresh_budget"]:
                if pending_path.exists():
                    pending = torch.load(pending_path, map_location="cpu", weights_only=False)
                    if pending["identity"] != identity:
                        raise ValueError("Pending event identity mismatch")
                    if pending["event"] <= progress["events"]:
                        pending_path.unlink()
                        continue
                    if pending["event"] != progress["events"] + 1:
                        raise ValueError("Pending event is out of order")
                    progress, rows = bo.restore(state, pending["before_oracle"], identity)
                    if backend:
                        backend.dirty()
                else:
                    started = time.monotonic()
                    if progress["recenter_plan"] is None and state.progress_fails_since_last_e2e >= recipe["e2e_every"]:
                        state.update_models_e2e()
                        if backend:
                            backend.dirty()
                        protocol.reencode(state)
                        old_epochs = state.init_n_epochs
                        state.initial_model_training_complete = False
                        state.init_n_epochs = state.num_update_epochs
                        state.update_surrogate_model()
                        state.init_n_epochs = old_epochs
                        inputs = list(state.train_x[-state.bsz:]) + list(state.top_k_xs)
                        batch = max(1, 2560 // max(map(len, inputs)))
                        progress["recenter_plan"] = dict(inputs=inputs * state.num_update_epochs, cursor=0, batch=batch)
                        progress["recenter_optimizer"] = None
                        bo.atomic_torch(checkpoint, bo.snapshot(state, identity, progress, rows), rotate=True)
                    plan = progress["recenter_plan"]
                    if plan is not None and plan["cursor"] >= len(plan["inputs"]):
                        progress["recenter_plan"] = None
                        progress["recenter_optimizer"] = None
                        bo.atomic_torch(checkpoint, bo.snapshot(state, identity, progress, rows), rotate=True)
                        continue
                    if plan is not None:
                        end = min(len(plan["inputs"]), plan["cursor"] + plan["batch"])
                        z = bo.encode(state.objective, plan["inputs"][plan["cursor"]:end])
                        with torch.no_grad():
                            sequences = state.objective.vae_decode(z)
                        phase = "recenter"
                    else:
                        state.update_surrogate_model()
                        n = min(recipe["batch_size"], recipe["proposals"] - progress["proposals"])
                        z, sequences = bo.propose(state, n)
                        phase, end = "acquisition", None
                    accepted = protocol.bounded_prefix(sequences, state.objective.xs_to_scores_dict,
                        recipe["fresh_budget"] - progress["fresh_evaluations"])
                    if not accepted:
                        raise ValueError("Empty event with unexhausted budgets")
                    pending = dict(identity=identity, event=progress["events"] + 1, phase=phase,
                        sequences=accepted, z=bo.cpu_copy(z[:len(accepted)]), next_cursor=end,
                        before_oracle=bo.snapshot(state, identity, progress, rows),
                        proposal_seconds=time.monotonic() - started)
                    bo.atomic_torch(pending_path, pending)
                commit_pending(state, identity, progress, rows, pending, directory, recipe)
                if stopping or (args.stop_after_events and progress["events"] >= args.stop_after_events):
                    print(json.dumps(dict(task=args.task, status="paused_at_checkpoint")), flush=True)
                    return
            values = status(recipe, identity, state, progress, "complete")
            bo.atomic_json(directory / "progress.json", values)
            bo.atomic_json(directory / "history.json", dict(values, observations=rows,
                reference=recipe["references"][args.task], initialization=recipe["initialization"],
                proposal_budget=recipe["proposals"], fresh_budget=recipe["fresh_budget"],
                initialization_actual_calls=0, initialization_source="strict_pinned_measured_pool",
                budget_definition=recipe["budget_definition"]))
            packed = archive(directory, identity)
            print(json.dumps(dict(task=args.task, status="complete", archive=str(packed))), flush=True)
        finally:
            if backend:
                with contextlib.suppress(Exception):
                    backend.close()


def worker(args):
    recipe = verify(args.campaign)
    preflight(args.campaign)
    tasks = recipe["train"]
    assigned = tasks[args.index::args.workers]
    ordered = assigned + [task for task in tasks if task not in assigned]
    stopping = False
    child = None
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for task in ordered:
        if stopping:
            return
        if (args.campaign / "runs" / (task + ".tar.gz")).exists():
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "run-task", "--campaign", str(args.campaign),
            "--task", task, "--cpu", str(args.cpu), "--launch"]
        if args.server:
            command += ["--server", args.server]
        else:
            command += ["--gpu", str(args.gpu)]
        child = subprocess.Popen(command, env=os.environ.copy())
        result = child.wait()
        if result not in (0, 73):
            print(json.dumps(dict(task=task, status="task_error", returncode=result)), flush=True)
    print(json.dumps(dict(status="worker_finished", host=socket.gethostname(), index=args.index)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--initialization", type=int, default=100)
    p.add_argument("--proposals", type=int, default=5000)
    p.add_argument("--fresh-budget", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--e2e-every", type=int, default=10)
    p.add_argument("--server", required=True)
    p.add_argument("--component", action="store_true")
    p.add_argument("--reuse-initialization", type=Path)
    sub.add_parser("verify")
    p = sub.add_parser("run-task")
    p.add_argument("--task", required=True)
    p.add_argument("--stop-after-events", type=int)
    p.add_argument("--launch", action="store_true")
    p.add_argument("--server")
    p.add_argument("--gpu", type=int)
    p.add_argument("--cpu", type=int, required=True)
    p = sub.add_parser("worker")
    p.add_argument("--server")
    p.add_argument("--gpu", type=int)
    p.add_argument("--cpu", type=int, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    for p in sub.choices.values():
        p.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        if min(args.initialization, args.proposals, args.fresh_budget, args.batch_size, args.e2e_every) <= 0:
            parser.error("Budgets and intervals must be positive")
        prepare(args)
    elif args.action == "verify":
        print(json.dumps(dict(valid=bool(verify(args.campaign)), launched=False)))
    else:
        os.sched_setaffinity(0, {args.cpu})
        os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="" if args.server else str(args.gpu))
        torch.set_num_threads(1)
        if args.action == "worker":
            worker(args)
        else:
            try:
                run_task(args)
            except BlockingIOError:
                sys.exit(73)


if __name__ == "__main__":
    main()
