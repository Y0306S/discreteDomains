#!/usr/bin/env python3
"""Prepare, verify, and run a locked peptide LOLBO campaign. Preparation never launches jobs."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.metadata
import io
import json
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

from mindware.domains.protein.verified_bo import (
    FORMAT, atomic_json, atomic_torch, collect_pools, cpu_copy, create_state, digest,
    load_pool, next_batch_size, propose, reencode_after_update, required_decoder_edits, restore, restore_rng,
    score_batch, similarity, snapshot, unique_top_k, validate_state,
)
import torch

UPSTREAM = Path("/root/data/cys/BOLT/optimization/peptides")
STEM = "dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118"
AUTHOR = Path("/root/data/cys/APEXGo/generation/checkpoints") / (STEM + ".pkl")
DEFAULT_CAMPAIGN = Path("/root/data/logs/peptide_bo_verified_20261003")


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def versions():
    names = ["torch", "gpytorch", "botorch", "linear-operator", "numpy", "lightning", "python-Levenshtein"]
    return {name: importlib.metadata.version(name) for name in names}


def prepare(args):
    campaign = args.campaign.resolve()
    if (campaign / "recipe.json").exists():
        raise ValueError("Campaign already prepared; verify it or select a new directory")
    vae = UPSTREAM / "uniref_vae/saved_models" / (STEM + "_pretrained_authors.pkl")
    if digest(vae) != digest(AUTHOR):
        raise ValueError("Peptide VAE differs from the authors' APEXGo checkpoint")
    pool_path = UPSTREAM / "apex_oracle/init_data/generated/refs_0_1000_manifest.json"
    pool = json.loads(pool_path.read_text())
    for key in ("init", "scores"):
        if digest(Path(pool[key + "_path"])) != pool[key + "_sha256"]:
            raise ValueError("Initialization pool digest mismatch")
    code = [Path(__file__), ROOT / "mindware/domains/protein/verified_bo.py",
            ROOT / "tools/check_peptide_bo_components.py", ROOT / "tests/domains/test_verified_peptide_bo.py"]
    code += sorted(UPSTREAM.rglob("*.py"))
    assets = [vae, AUTHOR, pool_path, Path(pool["init_path"]), Path(pool["scores_path"]),
              UPSTREAM / "uniref_vae/1mer_vocab.csv"]
    assets += sorted(p for p in (UPSTREAM / "apex_oracle/APEX_pathogen_models").iterdir() if p.is_file())
    recipe = dict(format=FORMAT, purpose="component_check" if args.component else "production",
        upstream=str(UPSTREAM), vae=str(vae),
        pool_manifest=str(pool_path), initialization=args.initialization, proposals=args.proposals,
        prepared_pool=str(campaign / "initialization.json"), init_policy=args.init_policy,
        batch_size=args.batch_size, e2e_every=args.e2e_every, e2e_policy="periodic_joint_update_encode_only",
        center_policy="best_observed_feasible", gp_initial_epochs=20, gp_update_epochs=2,
        learning_rate=0.001, latent_dimensions=256,
        decoder_policy="upstream_at_most_one_cysteine", initialization_unknown_policy="preserve_scored_X",
        budget_definition="initialization_rows_plus_proposed_candidates_including_cached_duplicates",
        similarity=0.75, seed=1, oracle="negative_APEX_bacteria_0", higher_is_better=True,
        train=["pep_%03d" % i for i in range(900)], test=["pep_%03d" % i for i in range(900, 1000)],
        code={str(p): digest(p) for p in code}, assets={str(p): digest(p) for p in assets},
        environment=versions(), prepared_at=time.time(),
        launch_policy="explicit_--launch_and_recipe_bound_preflight")
    selected, pools = collect_pools(recipe)
    if set(selected) != set(recipe["train"] + recipe["test"]):
        raise ValueError("Pool and fixed train900/test100 split differ")
    recipe["blocked_tasks"] = {task: "Decoder residue/cysteine rules require %d edits; similarity allows at most %d" % (
        required_decoder_edits(value["reference"]), int(len(value["reference"]) * (1 - recipe["similarity"])))
        for task, value in selected.items()
        if required_decoder_edits(value["reference"]) > int(len(value["reference"]) * (1 - recipe["similarity"]))}
    campaign.mkdir(parents=True, exist_ok=True)
    atomic_json(campaign / "initialization.json", selected)
    recipe["assets"][recipe["prepared_pool"]] = digest(Path(recipe["prepared_pool"]))
    atomic_json(campaign / "recipe.json", recipe)
    atomic_json(campaign / "pool_audit.json", dict(tasks=pools, verified=1000, all_feasible=True))
    print(json.dumps(dict(prepared=str(campaign), tasks=1000, blocked=recipe["blocked_tasks"], launched=False,
                         budget=args.initialization + args.proposals)), flush=True)


def verify_recipe(campaign):
    path = campaign / "recipe.json"
    recipe = json.loads(path.read_text())
    if recipe["format"] != FORMAT:
        raise ValueError("Unknown peptide protocol")
    for key in ("code", "assets"):
        for name, expected in recipe[key].items():
            if digest(Path(name)) != expected:
                raise ValueError("Pinned file changed: " + name)
    if versions() != recipe["environment"]:
        raise ValueError("Worker package versions differ from the prepared environment")
    return recipe


def verify_preflight(campaign):
    hold = campaign / "launch_hold.json"
    if hold.exists():
        raise ValueError("Production launch is on hold: " + json.loads(hold.read_text())["reason"])
    gate = json.loads((campaign / "preflight_passed.json").read_text())
    if gate["recipe_sha256"] != digest(campaign / "recipe.json"):
        raise ValueError("Preflight belongs to a different recipe")
    report_path = Path(gate["report"])
    if digest(report_path) != gate["report_sha256"]:
        raise ValueError("Preflight report changed")
    report = json.loads(report_path.read_text())
    if not report.get("passed") or report.get("recipe_sha256") != gate["recipe_sha256"]:
        raise ValueError("Component preflight did not pass for this recipe")
    required = {"unit_tests", "oracle_scores", "two_references", "gp_acquisition", "vae_update",
                "encode_only", "resume", "fresh_process_resume", "partial_batch", "archive"}
    if not required <= report["checks"].keys() or not all(report["checks"][k] for k in required):
        raise ValueError("Component preflight lacks required checks")


def gpu_available(gpu, min_free_mib=12000):
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    match = [row.split(",") for row in rows.splitlines() if int(row.split(",")[0]) == gpu]
    if len(match) != 1:
        raise ValueError("Requested GPU does not exist")
    uuid, free = match[0][1].strip(), int(match[0][2])
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    if any(row.split(",")[0].strip() == uuid for row in processes.splitlines()):
        raise ValueError("GPU %d is occupied; existing work will not be interrupted" % gpu)
    if free < min_free_mib:
        raise ValueError("Insufficient free GPU memory")
    return dict(index=gpu, uuid=uuid, free_mib=free)


def cpu_siblings(cpus):
    result = set()
    for cpu in cpus:
        text = Path("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu).read_text()
        for part in text.strip().split(","):
            bounds = [int(v) for v in part.split("-")]
            result.update(range(bounds[0], bounds[-1] + 1))
    return result


def cpu_available(cpus):
    if not cpus or not cpus <= os.sched_getaffinity(0):
        raise ValueError("Invalid CPU allocation")
    siblings = cpu_siblings(cpus)
    ignored = {os.getpid(), os.getppid()}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) in ignored:
            continue
        try:
            affinity = os.sched_getaffinity(int(process.name))
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
        # Broad-affinity system processes are handled by the utilization sample below.
        if command and len(affinity) <= 16 and affinity & siblings:
            raise ValueError("CPU allocation overlaps pinned PID %s: %s" % (process.name, command[:180]))

    def sample():
        values = {}
        for line in Path("/proc/stat").read_text().splitlines():
            fields = line.split()
            if fields[0].startswith("cpu") and fields[0][3:].isdigit():
                cpu = int(fields[0][3:])
                if cpu in siblings:
                    times = [int(v) for v in fields[1:9]]
                    values[cpu] = (sum(times), times[3] + times[4])
        return values
    before = sample()
    time.sleep(1)
    after = sample()
    for cpu in siblings:
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        if total and idle / total < 0.8:
            raise ValueError("CPU %d or its SMT sibling is busy" % cpu)
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) < 16 * 1024 * 1024:
        raise ValueError("Less than 16 GiB system RAM available")


@contextlib.contextmanager
def resources(args):
    cpus = {int(value) for value in args.cpus.split(",")}
    locks = Path("/root/data/logs/peptide_bo_resource_locks") / socket.gethostname()
    with contextlib.ExitStack() as stack:
        stack.enter_context(lock(locks / ("gpu%d.lock" % args.gpu)))
        for cpu in sorted(cpu_siblings(cpus)):
            stack.enter_context(lock(locks / ("cpu%d.lock" % cpu)))
        cpu_available(cpus)
        gpu_available(args.gpu)
        os.sched_setaffinity(0, cpus)
        os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS=str(len(cpus)),
                          OPENBLAS_NUM_THREADS=str(len(cpus)), MKL_NUM_THREADS=str(len(cpus)))
        torch.set_num_threads(len(cpus))
        yield cpus


def verify_archive(path, identity=None):
    import hashlib
    with tarfile.open(path, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if identity is not None and manifest["identity"] != identity:
            raise ValueError("Completed archive belongs to another run")
        members = archive.getmembers()
        expected_names = {"checkpoint.pt", "history.json", "progress.json"}
        if (set(manifest["members"]) != expected_names
                or {m.name for m in members} != expected_names | {"manifest.json"}
                or len(members) != 4 or not all(m.isfile() for m in members)):
            raise ValueError("Archive member inventory mismatch")
        for member, expected in manifest["members"].items():
            h = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError("Corrupt archive member: " + member)
        history = json.load(archive.extractfile("history.json"))
        if history["status"] != "complete" or history["identity"] != manifest["identity"]:
            raise ValueError("Archive does not contain a completed matching history")
        if (len(history["observations"]) != history["initialization"] + history["proposals"]
                or history["proposals"] != history["proposal_budget"]
                or history["fresh_evaluations"] + history["cache_hits"] != history["proposals"]):
            raise ValueError("Archive observation accounting is inconsistent")
    return history


def cleanup_archived(directory, target, identity):
    verify_archive(target, identity)
    with tarfile.open(target, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    for name, expected in {**manifest["members"], **manifest["cleanup"]}.items():
        if Path(name).name != name:
            raise ValueError("Unsafe archive cleanup name")
        path = directory / name
        if path.exists() and digest(path) != expected:
            raise ValueError("Archive source changed before cleanup: " + str(path))
    for name in {**manifest["members"], **manifest["cleanup"]}:
        (directory / name).unlink(missing_ok=True)
    if directory.exists():
        directory.rmdir()


def archive_run(directory, identity):
    target = directory.with_suffix(".tar.gz")
    files = [directory / name for name in ("checkpoint.pt", "history.json", "progress.json")]
    known = {"checkpoint.prev.pt", "checkpoint.pt.tmp", "checkpoint.prev.pt.tmp",
             "pending.pt", "pending.pt.tmp", "history.json.tmp", "progress.json.tmp"}
    extra = [p for p in directory.iterdir() if p not in files]
    if any(p.name not in known or not p.is_file() for p in extra):
        raise ValueError("Unknown files in completed task; refusing cleanup")
    manifest = dict(identity=identity, members={p.name: digest(p) for p in files},
                    cleanup={p.name: digest(p) for p in extra})
    temporary = target.with_name(target.name + ".tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=1) as archive:
        for path in files:
            archive.add(path, arcname=path.name, recursive=False)
        payload = json.dumps(manifest, allow_nan=False).encode()
        member = tarfile.TarInfo("manifest.json")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    verify_archive(temporary, identity)
    os.replace(temporary, target)
    cleanup_archived(directory, target, identity)
    return target


def run_task(args):
    campaign = args.campaign.resolve()
    recipe = verify_recipe(campaign)
    # Checkpoints pickle upstream TurboState; its module must exist before torch.load.
    sys.path.insert(0, recipe["upstream"])
    if args.task not in recipe["train"] + recipe["test"]:
        raise ValueError("Task is outside the fixed peptide split")
    if args.task in recipe["blocked_tasks"]:
        raise ValueError(args.task + ": " + recipe["blocked_tasks"][args.task])
    if recipe["purpose"] == "production":
        if not args.launch:
            raise ValueError("Production is prepared only; launching requires explicit --launch")
        verify_preflight(campaign)
    identity = dict(recipe_sha256=digest(campaign / "recipe.json"), task=args.task, format=FORMAT)
    archive = campaign / "runs" / (args.task + ".tar.gz")
    with lock(campaign / "locks" / (args.task + ".lock")):
        if archive.exists():
            cleanup_archived(campaign / "runs" / args.task, archive, identity)
            print(json.dumps(dict(task=args.task, status="already_complete")), flush=True)
            return
        directory = campaign / "runs" / args.task
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "checkpoint.pt"
        pending_path = directory / "pending.pt"
        if not checkpoint.exists() and any(directory.iterdir()):
            raise ValueError("Task data exists without its checkpoint; refusing to silently start over")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() else None
        if saved is not None and saved.get("identity") != identity:
            raise ValueError("Checkpoint identity changed")
        state = create_state(recipe, args.task, saved)
        reference, init_x, init_y = load_pool(recipe, args.task)
        if saved is None:
            progress = dict(rounds=0, proposals=0, fresh_evaluations=0, cache_hits=0, seconds=0.0)
            rows = [dict(sequence=x, score=float(y), mic=-float(y), phase="initialization",
                         round=0, feasible=True, cached=True, elapsed=0.0) for x, y in zip(init_x, init_y)]
            atomic_torch(checkpoint, snapshot(state, identity, progress, rows))
        else:
            progress, rows = restore(state, saved, identity)
        stopping = False

        def shutdown(_signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        while next_batch_size(progress, recipe["proposals"], recipe["batch_size"]):
            started = time.monotonic()
            n = next_batch_size(progress, recipe["proposals"], recipe["batch_size"])
            next_round = progress["rounds"] + 1
            if pending_path.exists():
                pending = torch.load(pending_path, map_location="cpu", weights_only=False)
                if pending["identity"] != identity:
                    raise ValueError("Pending proposal identity changed")
                if pending["round"] == progress["rounds"]:
                    pending_path.unlink()
                    continue
                if pending["round"] != next_round:
                    raise ValueError("Pending proposal is not the next round")
                if len(pending["sequences"]) != n:
                    raise ValueError("Pending batch has the wrong budget")
                # This checkpoint is after the GP update and before oracle evaluation.
                progress, rows = restore(state, pending["before_oracle"], identity)
            else:
                state.update_surrogate_model()
                z, sequences = propose(state, n)
                pending = dict(identity=identity, round=next_round, sequences=sequences, z=cpu_copy(z),
                    before_oracle=snapshot(state, identity, progress, rows), oracle_attempts=[])
                atomic_torch(pending_path, pending)
            if "scored" not in pending:
                pending["oracle_attempts"].append(dict(started_at=time.time(),
                    fresh_sequences=list(dict.fromkeys(x for x in pending["sequences"]
                        if x not in state.objective.xs_to_scores_dict))))
                atomic_torch(pending_path, pending)
                pending["scored"] = score_batch(state.objective, pending["sequences"])
                atomic_torch(pending_path, pending)
            else:
                scored = pending["scored"]
                state.objective.xs_to_scores_dict.update(scored["new_scores"])
                state.objective.num_calls += scored["fresh_evaluations"]
                restore_rng(scored["rng"])
            scored = pending["scored"]
            state.update_next(pending["z"], torch.tensor(scored["scores"], dtype=torch.float32),
                scored["sequences"], scored["constraints"], acquisition=True)
            unique_top_k(state)
            progress["proposals"] += n
            progress["rounds"] = next_round
            progress["fresh_evaluations"] += scored["fresh_evaluations"]
            progress["cache_hits"] += sum(scored["cache_hits"])
            if next_round % recipe["e2e_every"] == 0:
                reencode_after_update(state)
            if state.tr_state.restart_triggered:
                state.initialize_tr_state()
            progress["seconds"] += time.monotonic() - started
            for i, (sequence, score) in enumerate(zip(scored["sequences"], scored["scores"])):
                feasible = similarity(sequence, reference) >= recipe["similarity"] - 1e-8
                if feasible != bool(scored["constraints"][i, 0] <= 0):
                    raise ValueError("Reported and modeled peptide constraints differ")
                rows.append(dict(sequence=sequence, score=score, mic=-score, phase="acquisition",
                    round=next_round, feasible=feasible, cached=scored["cache_hits"][i],
                    elapsed=progress["seconds"], oracle_attempts=len(pending["oracle_attempts"])))
            validate_state(state)
            if (len(rows) != recipe["initialization"] + progress["proposals"]
                    or progress["fresh_evaluations"] + progress["cache_hits"] != progress["proposals"]
                    or state.objective.num_calls != progress["fresh_evaluations"]):
                raise RuntimeError("Peptide evaluation accounting diverged")
            atomic_torch(checkpoint, snapshot(state, identity, progress, rows), rotate=True)
            pending_path.unlink()
            status = dict(identity=identity, **progress, e2e_updates=state.tot_num_e2e_updates,
                best_feasible_mic=-float(state.best_score_seen), status="running")
            atomic_json(directory / "progress.json", status)
            print(json.dumps(status), flush=True)
            if stopping or (args.stop_after_rounds and next_round >= args.stop_after_rounds):
                print(json.dumps(dict(task=args.task, status="paused_at_checkpoint")), flush=True)
                return
        status = dict(identity=identity, **progress, e2e_updates=state.tot_num_e2e_updates,
            best_feasible_mic=-float(state.best_score_seen), status="complete")
        atomic_json(directory / "progress.json", status)
        atomic_json(directory / "history.json", dict(**status, reference=reference, observations=rows,
            initialization=recipe["initialization"], proposal_budget=recipe["proposals"],
            initialization_actual_calls=0, initialization_source="pinned_precomputed_pool",
            budget_definition=recipe["budget_definition"]))
        result = archive_run(directory, identity)
        print(json.dumps(dict(task=args.task, status="complete", archive=str(result))), flush=True)


def worker(args):
    campaign = args.campaign.resolve()
    recipe = verify_recipe(campaign)
    if recipe["purpose"] == "production":
        if not args.launch:
            raise ValueError("Production is prepared only; launching requires explicit --launch")
        verify_preflight(campaign)
    with resources(args) as cpus:
        env = dict(os.environ, PEPTIDE_WORKER_PARENT=str(os.getpid()))
        tasks = recipe["train"] + recipe["test"] if args.split == "all" else recipe[args.split]
        stopping = False
        child = None

        def shutdown(_signum, _frame):
            nonlocal stopping
            stopping = True
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        for task in tasks:
            if stopping:
                return
            if task in recipe["blocked_tasks"]:
                print(json.dumps(dict(task=task, status="blocked", reason=recipe["blocked_tasks"][task])), flush=True)
                continue
            cpu_available(cpus)
            gpu_available(args.gpu)
            command = [sys.executable, str(Path(__file__).resolve()), "run-task", "--campaign", str(campaign),
                       "--task", task, "--threads", str(len(cpus))]
            if args.launch:
                command.append("--launch")
            child = subprocess.Popen(command, env=env)
            result = child.wait()
            if stopping:
                return
            if result not in (0, 73):
                raise RuntimeError("Peptide task failed; worker stopped: " + task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--initialization", type=int, default=100)
    p.add_argument("--proposals", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--e2e-every", type=int, default=10)
    p.add_argument("--init-policy", choices=("first_valid", "legacy_first"), default="legacy_first")
    p.add_argument("--component", action="store_true")
    sub.add_parser("verify")
    p = sub.add_parser("run-task")
    p.add_argument("--task", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--stop-after-rounds", type=int)
    p.add_argument("--launch", action="store_true")
    p.add_argument("--gpu", type=int)
    p.add_argument("--cpus")
    p = sub.add_parser("worker")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--cpus", required=True, help="Comma-separated CPU IDs reserved for this worker")
    p.add_argument("--split", choices=("train", "test", "all"), default="all")
    p.add_argument("--launch", action="store_true")
    for p in sub.choices.values():
        p.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    args = parser.parse_args()
    if args.action == "prepare":
        if not 2 <= args.initialization <= 1000 or min(args.proposals, args.batch_size, args.e2e_every) < 1:
            parser.error("Invalid initialization or proposal budget")
        prepare(args)
    elif args.action == "verify":
        print(json.dumps(dict(valid=bool(verify_recipe(args.campaign.resolve())), launched=False)))
    elif args.action == "worker":
        worker(args)
    else:
        try:
            if os.environ.get("PEPTIDE_WORKER_PARENT") == str(os.getppid()):
                torch.set_num_threads(args.threads)
                run_task(args)
            else:
                if args.gpu is None or not args.cpus:
                    parser.error("Direct run-task requires --gpu and --cpus")
                with resources(args):
                    run_task(args)
        except BlockingIOError:
            print(json.dumps(dict(task=args.task, status="locked_by_another_worker")), flush=True)
            sys.exit(73)


if __name__ == "__main__":
    main()
