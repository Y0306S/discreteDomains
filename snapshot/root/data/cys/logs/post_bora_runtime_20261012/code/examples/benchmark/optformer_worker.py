#!/usr/bin/env python3
"""Original OptFormer training and EI policy on permitted ML histories."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mindware.components.optimizers.transfer_hpo.protocol import (
    TRAIN_MANIFEST, atomic_json, partition_tasks, sha256_file, sha256_directory,
)

OPT_REPO = Path("/root/data/cys/baseline/repos/optformer-neurips22")
T5X_REPO = Path("/root/data/cys/baseline/repos/t5x-2023-07")
VOCABULARY = "/root/data/cys/baseline/assets/optformer/sentencepiece.model"


def validate_resume(previous, identity, reconfigure_execution=False):
    changes = {key: {"old": previous.get(key), "new": value}
               for key, value in identity.items() if previous.get(key) != value}
    allowed = {"microbatches"} if reconfigure_execution else set()
    if set(changes) - allowed:
        raise ValueError("Cannot resume OptFormer with changed identity fields: " +
                         ", ".join(sorted(set(changes) - allowed)))
    return changes


def configure_runtime(*, deterministic_inference=True):
    # Avoid process-dependent bfloat16 GEMM choices in persistent replicas.
    flags = os.environ.get("XLA_FLAGS", "").split()
    autotune = [flag for flag in flags if flag.startswith("--xla_gpu_autotune_level=")]
    if deterministic_inference and autotune and autotune != ["--xla_gpu_autotune_level=0"]:
        raise ValueError("OptFormer inference requires deterministic GPU autotuning settings")
    if deterministic_inference and not autotune:
        os.environ["XLA_FLAGS"] = " ".join(flags + ["--xla_gpu_autotune_level=0"])
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def model_config(manifest):
    lines = [
        "from __gin__ import dynamic_registration",
        "from optformer.data import converters",
        "from optformer.data import tasks as t5_tasks",
        "from t5x.examples.t5 import network",
        "include 't5x/examples/t5/t5_1_1/base.gin'",
        "include 'optformer/t5x/configs/models/t5_1_1_overrides.gin'",
        "network.T5Config.dtype = %ml_optformer.bfloat16",
        "DROPOUT_RATE = 0.3",
        "MODEL_DIR = '/tmp/optformer_inference'",
        "STEP_OFFSET = 0",
        "TASK_FEATURE_LENGTHS = " + repr(manifest["lengths"]),
        "STUDY_CONVERTER = @converters.OptFormerConverter()",
        "converters.OptFormerConverter.max_trials = None",
        "converters.OptFormerConverter.discard_const_objective = False",
        "converters.OptFormerConverter.minimum_config = False",
        "t5_tasks.get_vocabulary.sentencepiece_model_file = " + repr(manifest["vocabulary"]),
    ]
    if manifest["model_size"] == "tiny":
        lines.extend([
            "network.T5Config.emb_dim = 32", "network.T5Config.num_heads = 2",
            "network.T5Config.num_encoder_layers = 1", "network.T5Config.num_decoder_layers = 1",
            "network.T5Config.head_dim = 16", "network.T5Config.mlp_dim = 64",
            "network.T5Config.dtype = 'float32'",
        ])
    return "\n".join(lines) + "\n"


def load_model(directory, meta):
    from mindware.components.optimizers.transfer_hpo.optformer_bridge import install_converter_compat, install_t5x_compat
    install_converter_compat()
    install_t5x_compat()
    from optformer.t5x.inference_utils import InferenceModel
    return InferenceModel.from_checkpoint(
        str(directory / meta["checkpoint"]), batch_size=1,
        model_gin_file=str(directory / "model.gin"),
        gin_search_paths=[str(OPT_REPO), str(T5X_REPO)],
    )


def cmd_train(args):
    from optformer.data import tasks
    from mindware.components.optimizers.transfer_hpo.optformer_bridge import (
        load_records, make_converter, study_from_record, checked_conversion,
    )
    if args.microbatches and args.batch_size % args.microbatches:
        raise ValueError("--microbatches must divide --batch-size")
    if (args.reconfigure_execution or args.benchmark) and not args.resume:
        raise ValueError("Execution reconfiguration and benchmarks require --resume")
    records = load_records(args.sequences)
    train_ids, valid_ids = partition_tasks([r["task_id"] for r in records], args.validation_tasks)
    vocabulary = tasks.get_vocabulary(args.vocabulary, expected_vocab_size=33100)
    converter = make_converter()
    max_input, max_target = 0, 0
    for record in records:
        converted, lengths = checked_conversion(converter, study_from_record(record), vocabulary)
        max_input = max(max_input, lengths["inputs"])
        max_target = max(max_target, lengths["targets"] + converted.num_parameters + 3)
    # Reserve metadata headroom for parameter-order augmentation and one next action.
    target = ((max_target + 127) // 128) * 128
    lengths = {"inputs": ((max_input + 255) // 128) * 128,
               **{key: target for key in ("targets", "target_inputs", "targets_types", "targets_masks")}}
    identity = {"method": "optformer_ml", "format_version": 2, "model_size": args.model_size,
                "numeric_converter": "continuous_numeric_v1",
                "sequences": str(Path(args.sequences).resolve()), "sequences_sha256": sha256_file(args.sequences),
                "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
                "train_tasks": train_ids, "validation_tasks": valid_ids,
                "vocabulary": str(Path(args.vocabulary).resolve()), "vocabulary_sha256": sha256_file(args.vocabulary),
                "lengths": lengths, "seed": args.seed, "batch_size": args.batch_size,
                "microbatches": args.microbatches, "eval_steps": args.eval_steps,
                "eval_batch_size": args.eval_batch_size or args.batch_size,
                "schema_fingerprints": sorted({r["schema_fingerprint"] for r in records}),
                "selection": "last completed training step", "source_weights": "random initialization"}
    if args.preflight:
        print(json.dumps(identity, indent=2))
        return
    directory = Path(args.model_dir).resolve()
    meta_path = directory / "meta.json"
    previous, completed, changes = {}, 0, {}
    if args.resume:
        previous = json.loads(meta_path.read_text())
        if previous.get("purpose") == "throughput_benchmark" and not args.benchmark:
            raise ValueError("Benchmark weights cannot be resumed as production training")
        changes = validate_resume(previous, identity, args.reconfigure_execution)
        checkpoints = list(directory.glob("checkpoint_[0-9]*"))
        completed = max((int(p.name.split("_")[-1]) for p in checkpoints), default=0)
        if args.steps <= completed:
            raise ValueError("--steps must exceed the last saved checkpoint")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("Model directory is not empty; use a new directory or --resume")
    directory.mkdir(parents=True, exist_ok=True)
    runtime = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "jax_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
               "microbatches": args.microbatches, "stats_every": args.stats_every}
    history = list(previous.get("execution_history", []))
    history.append({"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "resumed_from_step": completed, "changes": changes, "runtime": runtime})
    metadata = dict(identity, requested_steps=args.steps, resumed_from_step=completed,
                    purpose="throughput_benchmark" if args.benchmark else "production_training",
                    runtime=runtime, execution_history=history)
    atomic_json(meta_path, dict(metadata, status="training"))
    (directory / "model.gin").write_text(model_config(identity))
    training = [
        "from __gin__ import dynamic_registration",
        "import __main__ as train_script",
        "from t5x import utils",
        "from t5x import trainer",
        "include 'optformer/t5x/configs/runs/train.gin'",
        "include " + repr(str(directory / "model.gin")),
        "MIXTURE_OR_TASK_NAME = 'ml_train80'", "USE_CACHED_TASKS = False",
        "MODEL_DIR = " + repr(str(directory)), "TRAIN_STEPS = " + str(args.steps),
        "BATCH_SIZE = " + str(args.batch_size), "RANDOM_SEED = " + str(args.seed),
        "train/utils.DatasetConfig.seed = " + str(args.seed),
        "train/utils.DatasetConfig.pack = False",
        "train_eval/utils.DatasetConfig.batch_size = " + str(args.eval_batch_size or args.batch_size),
        "LEARNING_RATE = 0.3", "EVAL_STEPS = " + str(args.eval_steps),
        "trainer.Trainer.num_microbatches = " + repr(args.microbatches),
        "train_script.train.eval_period = " + str(args.save_every),
        "train_script.train.stats_period = " + str(args.stats_every),
        "utils.SaveCheckpointConfig.period = " + str(args.save_every),
        "utils.SaveCheckpointConfig.keep = 2",
    ]
    (directory / "train.gin").write_text("\n".join(training) + "\n")
    env = os.environ.copy()
    env["ML_OPTFORMER_MANIFEST"] = str(meta_path)
    env["PYTHONPATH"] = os.pathsep.join([str(OPT_REPO), str(T5X_REPO), str(ROOT), env.get("PYTHONPATH", "")])
    try:
        subprocess.run([sys.executable, str(ROOT / "examples/benchmark/optformer_t5x_train.py"),
                        "--gin_file=" + str(directory / "train.gin"),
                        "--gin_search_paths=" + str(OPT_REPO), "--gin_search_paths=" + str(T5X_REPO)],
                       env=env, check=True)
    except subprocess.CalledProcessError as exc:
        atomic_json(meta_path, dict(metadata, status="failed",
                                    returncode=exc.returncode))
        raise
    checkpoint = directory / ("checkpoint_%d" % args.steps)
    if not checkpoint.exists():
        raise RuntimeError("T5X finished without the requested final checkpoint")
    meta = dict(metadata, checkpoint=checkpoint.name, steps_completed=args.steps,
                checkpoint_sha256=sha256_directory(checkpoint),
                model_config_sha256=sha256_file(directory / "model.gin"))
    atomic_json(meta_path, dict(meta, status="verifying"))
    verify_env = dict(env, CUDA_VISIBLE_DEVICES="", JAX_PLATFORM_NAME="cpu")
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify-checkpoint",
                        "--model-dir", str(directory)], env=verify_env, check=True)
    except subprocess.CalledProcessError as exc:
        atomic_json(meta_path, dict(meta, status="failed_verification", returncode=exc.returncode))
        raise
    atomic_json(meta_path, dict(meta, status="benchmark_complete" if args.benchmark else "complete"))


def cmd_verify_checkpoint(args):
    directory = Path(args.model_dir)
    meta = json.loads((directory / "meta.json").read_text())
    if (sha256_directory(directory / meta["checkpoint"]) != meta["checkpoint_sha256"]
            or sha256_file(directory / "model.gin") != meta["model_config_sha256"]):
        raise ValueError("Checkpoint verification digest mismatch")
    model = load_model(directory, meta)
    import jax
    import numpy as np
    if not all(np.isfinite(np.asarray(value)).all() for value in jax.tree_util.tree_leaves(model._params)):
        raise ValueError("Restored checkpoint contains non-finite parameters")
    print(json.dumps({"verified_checkpoint": meta["checkpoint"], "finite_parameters": True}), flush=True)


def verified_metadata(directory):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    if (meta.get("status") != "complete" or meta["method"] != "optformer_ml"
            or meta.get("numeric_converter") != "continuous_numeric_v1"
            or sha256_directory(directory / meta["checkpoint"]) != meta["checkpoint_sha256"]
            or sha256_file(directory / "model.gin") != meta["model_config_sha256"]
            or sha256_file(meta["vocabulary"]) != meta["vocabulary_sha256"]
            or sha256_file(meta["sequences"]) != meta["sequences_sha256"]
            or sha256_file(TRAIN_MANIFEST) != meta["train_manifest_sha256"]):
        raise ValueError("OptFormer checkpoint or provenance mismatch")
    return meta


def suggest_with_model(model, meta, request):
    import numpy as np
    from optformer.t5x.policies import OptFormerDesigner, PolicyType
    from vizier import algorithms as vza
    from mindware.components.optimizers.transfer_hpo.optformer_bridge import (
        study_from_record, problem_from_record, config_from_suggestion, checked_conversion,
    )
    from mindware.components.optimizers.transfer_hpo.protocol import acquisition_diagnostics
    # Weights and compiled functions are shared; each call owns its study and RNG.
    model._dataset_builder.dataset([])
    model.set_seed(int(request["seed"]))
    cs, _ = problem_from_record(request)
    study = study_from_record(request)
    class CheckedDesigner(OptFormerDesigner):
        def _ei(self, *args, **kwargs):
            values = super()._ei(*args, **kwargs)
            if not np.isfinite(values).all():
                raise RuntimeError("OptFormer returned non-finite expected improvement")
            self.ei_diagnostics = acquisition_diagnostics("expected_improvement", values)
            return values

    designer = CheckedDesigner(study.problem, model, designer_name=None,
                                policy_type=PolicyType.RANK_SAMPLES,
                                num_samples=int(request.get("num_samples", 100)),
                                ranking_config={"type": "ei"},
                                study_converter_config={"randomize_parameter_order": False, "max_trials": None})
    converted, _ = checked_conversion(model.study_converter, study, model.vocab, meta["lengths"])
    if len(study.trials) >= model.max_trials(converted.num_parameters):
        raise ValueError("Target history and next action exceed the trained OptFormer context")
    completed = vza.CompletedTrials(study.trials)
    # The released policy predates Vizier's rename from trials to completed.
    if not hasattr(completed, "trials"):
        completed = SimpleNamespace(trials=completed.completed)
    designer.update(completed)
    suggestions = designer.suggest(1)
    suggestion = suggestions[0] if suggestions else None
    config, resolution = config_from_suggestion(cs, suggestion, return_diagnostics=True)
    return {"config": config,
                           "configuration_resolution": resolution,
                           "acquisition_applies_to_evaluated_config": not resolution["configuration_changed"],
                           "acquisition": designer.ei_diagnostics if suggestions else None,
                           "successful_history_count": len(study.trials),
                           "checkpoint_sha256": meta["checkpoint_sha256"]}


class OptFormerEngine:
    def __init__(self, directory):
        self.meta = verified_metadata(directory)
        self.model = load_model(Path(directory), self.meta)

    def suggest(self, request):
        try:
            return suggest_with_model(self.model, self.meta, request)
        finally:
            self.model._dataset_builder.dataset([])


def cmd_suggest(args):
    engine = OptFormerEngine(args.model_dir)
    atomic_json(args.out, engine.suggest(json.loads(Path(args.request).read_text())))


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--sequences", required=True)
    train.add_argument("--validation-tasks", required=True)
    train.add_argument("--model-dir", required=True)
    train.add_argument("--steps", required=True, type=positive)
    train.add_argument("--vocabulary", default=VOCABULARY)
    train.add_argument("--batch-size", type=positive, default=256)
    train.add_argument("--model-size", choices=("base", "tiny"), default="base")
    train.add_argument("--seed", type=int, default=123)
    train.add_argument("--save-every", type=positive, default=1000)
    train.add_argument("--stats-every", type=positive, default=1)
    train.add_argument("--microbatches", type=positive)
    train.add_argument("--eval-steps", type=positive, default=20)
    train.add_argument("--eval-batch-size", type=positive)
    train.add_argument("--preflight", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--reconfigure-execution", action="store_true")
    train.add_argument("--benchmark", action="store_true")
    verify = sub.add_parser("verify-checkpoint")
    verify.add_argument("--model-dir", required=True)
    suggest = sub.add_parser("suggest")
    suggest.add_argument("--model-dir", required=True)
    suggest.add_argument("--request", required=True)
    suggest.add_argument("--out", required=True)
    args = parser.parse_args()
    configure_runtime(deterministic_inference=args.command != "train")
    if args.command == "verify-checkpoint":
        return cmd_verify_checkpoint(args)
    return cmd_train(args) if args.command == "train" else cmd_suggest(args)


if __name__ == "__main__":
    main()
