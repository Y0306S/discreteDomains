"""Lossless ML ConfigSpace bridge to the original OptFormer Vizier converter."""

import json
import math
import functools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ConfigSpace import Configuration, CategoricalHyperparameter, Constant
from ConfigSpace import UniformFloatHyperparameter, UniformIntegerHyperparameter
from vizier import pyvizier as vz

from .protocol import allowed_task_ids, checked_source_path, sha256_file
from .schema import cs_from_spec, schema_fingerprint


def category(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def problem_from_record(record):
    cs = cs_from_spec(record["configspace"])
    if schema_fingerprint(cs) != record["schema_fingerprint"]:
        raise ValueError("OptFormer schema round-trip mismatch")
    if cs.get_conditions() or cs.get_forbiddens():
        raise ValueError("Original OptFormer bridge requires an unconditional, unconstrained space")
    problem = vz.ProblemStatement()
    problem.metadata["name"] = record["task_id"]
    problem.metadata["algorithm"] = record.get("algorithm_condition", "unknown_mixed")
    problem.metric_information.append(vz.MetricInformation(name="score", goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    root = problem.search_space.root
    for hp in cs.get_hyperparameters():
        if isinstance(hp, CategoricalHyperparameter):
            root.add_categorical_param(hp.name, [category(v) for v in hp.choices])
        elif isinstance(hp, Constant):
            root.add_categorical_param(hp.name, [category(hp.value)])
        elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            scale = vz.ScaleType.LOG if hp.log else vz.ScaleType.LINEAR
            if hp.q is not None:
                values = np.arange(hp.lower, hp.upper + hp.q / 2, hp.q).tolist()
                if len(values) > 1000:
                    raise ValueError("Quantized parameter exceeds OptFormer's 1000-token support")
                root.add_discrete_param(hp.name, values, scale_type=scale)
            elif isinstance(hp, UniformIntegerHyperparameter):
                root.add_int_param(hp.name, hp.lower, hp.upper, scale_type=scale)
            else:
                root.add_float_param(hp.name, hp.lower, hp.upper, scale_type=scale)
        else:
            raise ValueError("Unsupported OptFormer hyperparameter: %s" % hp)
    return cs, problem


def trial_from_config(cs, config, objective, index):
    validated = Configuration(cs, values=config)
    cs.check_configuration(validated)
    values = {}
    for hp in cs.get_hyperparameters():
        value = validated[hp.name]
        values[hp.name] = category(value) if isinstance(hp, (CategoricalHyperparameter, Constant)) else value
    trial = vz.Trial(id=index + 1, parameters=values)
    if not math.isfinite(float(objective)):
        raise ValueError("OptFormer requires finite completed observations")
    trial.complete(vz.Measurement(metrics={"score": float(objective)}))
    return trial


def study_from_record(record):
    cs, problem = problem_from_record(record)
    if len(record["configs"]) != len(record["y_maximize"]):
        raise ValueError("OptFormer configurations and objectives do not align")
    trials = [trial_from_config(cs, cfg, y, i)
              for i, (cfg, y) in enumerate(zip(record["configs"], record["y_maximize"]))]
    return vz.ProblemAndTrials(problem=problem, trials=trials)


def config_from_suggestion(cs, suggestion, *, return_diagnostics=False):
    from mindware.components.optimizers.strict_recommendation import StrictRecommendationResolver

    parameters = getattr(suggestion, "parameters", None)
    raw_values = {} if parameters is None else {
        name: parameter.value for name, parameter in parameters.items()}
    # Keep malformed values serializable without losing which parameter was invalid.
    raw_values = {name: repr(value) if isinstance(value, float) and not math.isfinite(value) else value
                  for name, value in raw_values.items()}
    values, decode_errors = dict(raw_values), {}
    for hp in cs.get_hyperparameters():
        if hp.name not in values:
            continue
        raw = values[hp.name]
        if isinstance(hp, (CategoricalHyperparameter, Constant)):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError) as exc:
                decode_errors[hp.name] = str(exc)
                del values[hp.name]
                continue
        elif (isinstance(hp, UniformIntegerHyperparameter) and type(raw) in (int, float)
              and math.isfinite(raw) and int(raw) == raw):
            raw = int(raw)
        values[hp.name] = raw
    resolver = StrictRecommendationResolver(cs, default_on_error=True)
    config = dict(resolver.resolve(values))
    diagnostics = dict(resolver.last_record, raw_suggestion=raw_values, decode_errors=decode_errors)
    diagnostics["configuration_changed"] = bool(
        decode_errors or diagnostics.get("repairs") or diagnostics.get("defaulted")
        or diagnostics.get("fallback"))
    if return_diagnostics:
        return config, diagnostics
    return config


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    allowed, seen = allowed_task_ids(), set()
    for record in records:
        task = record["task_id"]
        expected = "%s/%s/%s" % (record["task_type"], record["dataset"], record["algorithm"])
        if task != expected or task not in allowed or task in seen:
            raise ValueError("Duplicate or non-train80 source: %s" % task)
        seen.add(task)
        source = checked_source_path(record["source_path"])
        if sha256_file(source) != record["source_sha256"]:
            raise ValueError("OptFormer source history changed")
        if not record.get("ordered") or record.get("shuffled"):
            raise ValueError("OptFormer requires ordered source trajectories")
        if not record["configs"]:
            raise ValueError("Empty OptFormer source")
        study_from_record(record)
    if not records:
        raise ValueError("No OptFormer sources")
    return records


def make_converter(training=False):
    install_converter_compat()
    from optformer.data.converters import OptFormerConverter
    return OptFormerConverter(
        min_trials=1, max_trials=None, discard_const_objective=False,
        randomize_parameter_order=training,
        rand_objective_scale_range=(0.3, 1.0) if training else None,
    )


def install_converter_compat():
    from optformer.data import converters
    from vizier.pyvizier import converters as vizier_converters
    # The original OptFormer quantizer assumes all numeric inputs are continuous.
    # Newer Vizier treats short integer ranges as categorical indices by default.
    converters.converters = SimpleNamespace(DefaultModelInputConverter=functools.partial(
        vizier_converters.DefaultModelInputConverter, max_discrete_indices=0))


def install_t5x_compat():
    import gin
    import jax
    from t5x import partitioning

    if getattr(partitioning.default_mesh, "_ml_gpu_compat", False):
        return
    # TensorFlow can register a different numpy bfloat16 type than JAX.
    gin.constant("ml_optformer.bfloat16", jax.numpy.bfloat16)
    original = partitioning.default_mesh

    @functools.wraps(original)
    def default_mesh(num_partitions, model_parallel_submesh=None, backend=None):
        # Old T5X inspects TPU coordinates before dispatching to its GPU mesh.
        if jax.devices(backend)[-1].platform == "gpu" and model_parallel_submesh is None:
            return partitioning.get_gpu_mesh(num_partitions)
        return original(num_partitions, model_parallel_submesh, backend)

    default_mesh._ml_gpu_compat = True
    partitioning.default_mesh = default_mesh


def checked_conversion(converter, study, vocabulary, lengths=None):
    converted = converter.study_to_texts(study)
    if not converted.inputs or not converted.targets:
        raise ValueError("Upstream OptFormer rejected a study")
    expected = len(study.trials) * (converted.num_parameters + 3)
    measured = {key: len(vocabulary.encode(getattr(converted, key)))
                for key in ("inputs", "targets", "target_inputs")}
    # Each trial has parameter tokens, '*', objective token and '|'.
    if measured["targets"] != expected:
        raise ValueError("OptFormer silently dropped observations or changed token layout: %s != %s" %
                         (measured["targets"], expected))
    if lengths and any(measured[key] > lengths[key] for key in measured):
        raise ValueError("OptFormer sequence would be truncated: %s" % measured)
    return converted, measured


def register_task(manifest):
    from optformer.data import tasks
    import seqio
    records = load_records(manifest["sequences"])
    if sha256_file(manifest["sequences"]) != manifest["sequences_sha256"]:
        raise ValueError("OptFormer training sequences changed")
    valid = set(manifest["validation_tasks"])
    vocabulary = tasks.get_vocabulary(manifest["vocabulary"], expected_vocab_size=33100)

    def dataset_fn(split, shuffle_files=False, seed=None):
        del shuffle_files
        if seed is not None:
            np.random.seed(seed)
        selected = [r for r in records if (r["task_id"] in valid) == (split == "validation")]
        studies = [study_from_record(record) for record in selected]
        converter = make_converter(training=split == "train")

        class CheckedConverter:
            def study_to_texts(self, study):
                return checked_conversion(converter, study, vocabulary, manifest["lengths"])[0]

        def generator():
            yield from studies

        return tasks.vizier_dataset_from_generator(generator, CheckedConverter(), suppress_error=False)

    tasks.add_tasks("ml_train80", vocabulary=vocabulary, masked_types=["separator"],
                    dataset_fn=dataset_fn, splits=("train", "validation"))
    # Upstream's 100,000-example shuffle buffer needlessly repeats this small corpus.
    task = seqio.TaskRegistry.get("ml_train80").replace(shuffle_buffer_size=max(1, len(records) - len(valid)))
    seqio.TaskRegistry.remove("ml_train80")
    seqio.TaskRegistry.add_provider("ml_train80", task)
