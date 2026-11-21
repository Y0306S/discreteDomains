"""CPU diagnostics of existing baselines; model replies and oracles are fixtures.

Run each component in its existing environment. A passed component probe does
not establish a working Qwen/ML integration. Production implementations are not
modified by this script.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/root/data/cys/baseline/repos")
sys.path.insert(0, str(ROOT))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def error_from(call):
    try:
        call()
    except Exception as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    return None


def bora_checks(record):
    import numpy as np
    from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformIntegerHyperparameter
    from ConfigSpace import Configuration
    from mindware.components.optimizers import bora_optimizer as implementation
    from mindware.components.optimizers import bopro_optimizers_utils
    from mindware.components.optimizers.transfer_hpo.schema import build_hpo_cs, cs_from_spec
    from mindware.third_party.opro.bora_acq import BORALLM_ACQ

    cls = implementation.BORAOptimizer
    logger = logging.getLogger("bora_probe")
    fixed = build_hpo_cs("cls", "lightgbm")
    with tempfile.TemporaryDirectory(prefix="bora_constructor_probe_") as directory:
        with patch.object(implementation.KnowledgeDatabase, "instance") as kd, \
                patch.object(implementation, "BORALLM_ACQ") as acq:
            failure = error_from(lambda: cls(
                SimpleNamespace(fixed_config=None), fixed, name="hpo", timestamp=time.time(),
                output_dir=directory, dataset_profile={"metric": "bal_acc"}, evaluation_limit=2))
        record("fixed_algorithm_constructor", failure is None, error=failure,
               retrieval_calls=kd.call_count, llm_constructor_calls=acq.call_count)

    failure = error_from(lambda: bopro_optimizers_utils._dict_to_cfg(
        dict(fixed.get_default_configuration()), fixed))
    record("fixed_algorithm_evaluation_conversion", failure is None, error=failure)

    # A categorical fixture lets us inspect later methods past the Constant crash.
    space = ConfigurationSpace(seed=123)
    space.add_hyperparameters([
        CategoricalHyperparameter("algorithm", ["fixture"]),
        UniformIntegerHyperparameter("fixture:depth", 3, 15, default_value=9),
        CategoricalHyperparameter("fixture:criterion", ["gini", "entropy"]),
    ])
    cfg = space.get_default_configuration()
    opt = cls.__new__(cls)
    opt.config_space, opt.logger = space, logger
    for name, raw in (
        ("invalid_numeric_is_rejected", {"depth": -1}),
        ("invalid_category_is_rejected", {"criterion": "not_a_criterion"}),
        ("missing_tunables_reach_extraction", {}),
    ):
        try:
            converted = dict(opt._safe_dict_to_config(raw))
            record(name, False, input=raw, observed=converted,
                   reason="Accepted without extraction; value changed or defaulted")
        except Exception as exc:
            record(name, True, error=str(exc))

    parser = BORALLM_ACQ.__new__(BORALLM_ACQ)
    parser.logger = logger
    responses = {
        "complete_json": json.dumps({"hypotheses": [{"config": dict(cfg)}]}),
        "prose": "I recommend depth 7 and criterion entropy.",
        "missing_fields": json.dumps({"hypotheses": [{"config": {}}]}),
        "out_of_range": json.dumps({"hypotheses": [{"config": {"fixture:depth": -1}}]}),
    }
    for name, text in responses.items():
        parsed = parser._parse_response(text)
        record("parser_" + name, parsed["valid"] if name == "complete_json" else False,
               parsed_valid=parsed["valid"], parsed_configs=parsed["configs"],
               extractor_present=False)

    coverage = json.loads((ROOT / "benchmark_data/ml_train_80_transfer_v2/coverage.json").read_text())
    failures = []
    for group in coverage["groups"]:
        dump = json.loads(Path(group["dump"]).read_text())
        cs = cs_from_spec(dump["configspace"])
        raw = json.loads(json.dumps(dict(cs.get_default_configuration())))
        failure = error_from(lambda: cs.check_configuration(Configuration(cs, values=raw)))
        if failure:
            failures.append({"group": group["group"], "error": failure})
    record("actual_ml_schema_default_json_roundtrips", not failures,
           schemas=len(coverage["groups"]), failures=failures)

    with tempfile.TemporaryDirectory(prefix="bora_wiring_probe_") as directory:
        kd = SimpleNamespace(fetch_best_conf_knowledge=Mock(return_value={}))
        with patch.object(implementation.KnowledgeDatabase, "instance", return_value=kd), \
                patch.object(implementation, "MyAdvisor", return_value=SimpleNamespace()), \
                patch.object(implementation, "BORALLM_ACQ", return_value=SimpleNamespace()):
            constructed = cls(
                SimpleNamespace(fixed_config=None), space, name="hpo", timestamp=time.time(),
                output_dir=directory, dataset_profile={"metric": "bal_acc"},
                chat_engine="qwen_probe_sentinel", evaluation_limit=2)
        record("requested_model_is_respected", constructed.chat_engine == "qwen_probe_sentinel",
               requested="qwen_probe_sentinel", observed=constructed.chat_engine)
        record("no_cross_task_retrieval", kd.fetch_best_conf_knowledge.call_count == 0,
               retrieval_calls=kd.fetch_best_conf_knowledge.call_count)

    def fixture(warmup=False):
        instance = cls.__new__(cls)
        instance.config_space, instance.logger = space, logger
        instance.configs = [] if warmup else [cfg] * 6
        instance.perfs = [] if warmup else [1.0] * 6
        instance.lower_is_better = False
        instance.comments_history, instance.llm_query_cost, instance.llm_query_time = [], [], []
        instance.acq_func = SimpleNamespace(
            get_total_llm_cost=lambda: 0.0,
            get_action_a2_suggestions=Mock(return_value=([dict(cfg)], "fixture", 0.0, 0.0)),
            get_action_a3_selection=Mock(return_value=([], "", 0.0, 0.0)))
        instance.advisor = SimpleNamespace(init_num=5, get_suggestion=Mock(return_value=cfg),
                                           surrogate_model=SimpleNamespace(train=Mock()))
        instance.history = SimpleNamespace(update_observation=Mock())
        instance.n_bo_candidates, instance.n_lbo_selected, instance.n_llm_suggest = 5, 1, 1
        instance._calculate_uncertainty = Mock(return_value=(1.0, 1.0))
        instance._check_plateau = Mock(return_value=True)
        instance._update_trust_mechanism = Mock()
        instance.update_saver, instance._record_event = Mock(), Mock()
        instance.early_stopped_flag = instance.timeout_flag = False
        instance.time_limit, instance.timestamp = None, time.time()
        instance.evaluation_num_limit = 100
        instance.incumbent_perf, instance.incumbent_config = -np.inf, dict(cfg)
        instance.max_llm_cost, instance._a2_train_counter = 1.0, 0
        instance.sigma_max_global = 1.0
        instance.sigma_lower_ratio, instance.sigma_upper_ratio = 0.3, 0.5
        return instance

    instance = fixture()
    result = instance._perform_action_a3()
    failure = error_from(lambda: unpack_action(result))
    record("a3_failure_returns_four_fields", len(result) == 4,
           returned_fields=len(result), error=failure)

    invented = dict(cfg, **{"fixture:depth": 10})
    instance.acq_func.get_action_a3_selection.return_value = ([invented], "fixture", 0.0, 0.0)
    result = instance._perform_action_a3()
    accepted = [dict(c) for c in result[0]]
    record("a3_selection_is_from_supplied_candidates", all(c == dict(cfg) for c in accepted),
           supplied=[dict(cfg)], accepted=accepted)

    def fake_evaluate(instance, raw, cs):
        return SimpleNamespace(config=Configuration(cs, values=raw), objectives=[-1.0]), 0.0

    instance = fixture(warmup=True)
    instance.evaluation_num_limit = 2
    with patch.object(implementation, "_evaluate_dict", side_effect=fake_evaluate) as oracle:
        for _ in range(3):
            instance.iterate()
    record("evaluation_limit_is_enforced", oracle.call_count <= 2,
           configured_limit=2, attempted_iterations=3, observed_evaluations=oracle.call_count)

    instance = fixture()
    with patch.object(implementation, "_evaluate_dict", side_effect=fake_evaluate):
        instance.iterate()
        instance.iterate()
    observed_actions = [call.kwargs["note"] for call in instance._record_event.call_args_list]
    policy_module = load_file("bora_upstream_policy_probe", UPSTREAM / "bora-the-explorer/bora/policy.py")
    policy = policy_module.Policy(dim=2, m_init=1)
    policy.initialize(iteration=0, max_target=1.0, uncertainty=1.0)
    policy.update(last_targets=[1.0], max_target=1.0, iteration=1, uncertainty=1.0)
    upstream_actions = [policy.get_action().name, policy.get_action().name]
    record("forced_bo_after_assistant", instance.acq_func.get_action_a2_suggestions.call_count == 1,
           observed_actions=observed_actions, upstream_actions=upstream_actions,
           note="Stationary observations; trust update isolated with a fixture")


def unpack_action(result):
    candidates, note, cost, duration = result
    return candidates, note, cost, duration


def tosfit_checks(record):
    import torch
    torch.set_num_threads(1)
    source = UPSTREAM / "tosfit/src"
    gp_module = load_file("tosfit_gp_probe", source / "gaussian_process.py")
    gp = gp_module.LinearGaussianProcess(4, nar=0.01, dtype=torch.float64)
    features = torch.eye(4, dtype=torch.float64)
    rewards = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float64)
    gp.add_observations(features, rewards)
    means, variances = gp.posterior_mean(features), gp.posterior_var(features)
    record("isolated_gp_conditioning", bool(torch.isfinite(means).all()
           and torch.isfinite(variances).all() and (variances >= 0).all()
           and torch.allclose(means, rewards, atol=0.001)),
           means=means.tolist(), variances=variances.tolist())

    # Avoid unrelated eager imports in src/__init__.py. Load actual core modules;
    # replace only logging and generation for the oracle-accounting loop probe.
    src = ModuleType("src")
    src.__path__ = [str(source)]
    utils = ModuleType("src.utils")
    utils.__path__ = [str(source / "utils")]
    wandb = ModuleType("wandb")
    wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    wandb.run, wandb.log = SimpleNamespace(summary={}), Mock()
    with patch.dict(sys.modules, {"src": src, "src.utils": utils, "wandb": wandb}):
        misc = load_file("src.utils.misc", source / "utils/misc.py")
        lite = load_file("src.lite", source / "lite.py")
        logits = torch.nn.Parameter(torch.tensor([0.1, -0.1, 0.2, 0.0], dtype=torch.float64))
        before = logits.detach().clone()
        log_probs = logits.log_softmax(dim=0)
        advantage = lite.pom_objective_advantage_function(
            means, variances.sqrt(), log_probs.detach(), log_probs.detach(), lite.inv_vapor_exp, 0.0)
        misc.enable_direct_grad([logits], lambda p: torch.optim.SGD(p, lr=0.01, maximize=True))
        (advantage * log_probs).mean().backward()
        record("isolated_policy_gradient_and_sgd_hook", bool(torch.isfinite(logits).all()
               and not torch.equal(before, logits.detach())),
               max_parameter_change=float((before - logits.detach()).abs().max()),
               note="Four-logit CPU fixture, not Qwen training or an ML integration")

        transformers = ModuleType("transformers")
        transformers.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
        transformers.AutoModelForCausalLM = transformers.AutoTokenizer = object
        language_model = ModuleType("src.language_model")
        with patch.dict(sys.modules, {"transformers": transformers, "src.language_model": language_model}):
            module = load_file("src.tosfit", source / "tosfit.py")

        proposals = 0
        def generate(*args, **kwargs):
            nonlocal proposals
            proposals += 1
            return (torch.full((16, 1), proposals, dtype=torch.long),
                    torch.ones((16, 4), dtype=torch.float64), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        oracle_calls = []
        def reward(tokens):
            oracle_calls.extend(tokens[:, 0].tolist())
            return 100.0 - tokens[:, 0].float(), ["fixture"] * len(tokens)

        class Tokenizer:
            eos_token_id = 0
            def __call__(self, *args, **kwargs):
                return {"input_ids": torch.tensor([[1]])}
            def decode(self, *args, **kwargs):
                return "fixture"

        with patch.object(module, "_fit_pom_with_variational_flite", side_effect=generate):
            module.tosfit(
                reward_function=reward, reward_for_invalid_generation=-1.0, answer=None,
                reward_model=SimpleNamespace(add_observations=Mock()), feature_embedding_model=None,
                embedding_aggregation=None, kernel_feature_transformation=None,
                inverse_pom_activation_exp=lite.inv_vapor_exp, total_prompt="fixture", tokenizer=Tokenizer(),
                generator=SimpleNamespace(parameters=lambda: []), reference_generator=None,
                temperature=1.0, max_new_tokens=2, optimizer_constructor=None, num_samples=12,
                bo_batch_size=1, fine_tune_steps_per_bo_step=1, batch_size=16, mini_batch_size=1, alpha=0.0)
        record("nominal_proposals_equal_oracle_calls", len(oracle_calls) == 12,
               nominal_proposals=12, actual_oracle_calls=len(oracle_calls),
               repeated_incumbent_calls=oracle_calls.count(1) - 1,
               note="Actual upstream loop; generation, reward and logger are fixtures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, choices=("bora", "tosfit"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Run with CUDA_VISIBLE_DEVICES= for these CPU diagnostics")
    report = {"component": args.component, "live_llm_calls": 0, "qwen_training": False,
              "production_files_modified": False, "checks": []}
    def record(name, passed, **details):
        item = {"check": name, "passed": bool(passed), **details}
        report["checks"].append(item)
        print(json.dumps(item), flush=True)
    try:
        (bora_checks if args.component == "bora" else tosfit_checks)(record)
        report["probe_completed"] = True
    except Exception as exc:
        report["probe_completed"] = False
        report["probe_error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        sources = ([ROOT / "mindware/components/optimizers/bora_optimizer.py",
                    ROOT / "mindware/third_party/opro/bora_acq.py"] if args.component == "bora"
                   else [UPSTREAM / "tosfit/src/tosfit.py", UPSTREAM / "tosfit/src/lite.py"])
        report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
