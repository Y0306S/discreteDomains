"""Official BORA assistant/acquisition with mixed ConfigSpace and local-Qwen I/O."""

import builtins
import copy
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from ConfigSpace.util import deactivate_inactive_hyperparameters
from ConfigSpace.hyperparameters import IntegerHyperparameter, FloatHyperparameter

from mindware.components.optimizers.online_upstream import UPSTREAM, load_module
from mindware.components.optimizers.strict_recommendation import decode_object


ROOT = UPSTREAM / "bora-the-explorer"
BORA_CONTEXT_TOKENS = 32768
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from bora.acquisition import ExpectedImprovement
from bora.experiment import Type

# Isolate the Comment subclass and relative prompt paths from the upstream package.
official = load_module("bora.ml_assistant_upstream", ROOT / "bora/assistant.py")


def _prompt_open(path, *args, **kwargs):
    value = Path(path)
    if not value.is_absolute() and value.parts[:2] == ("bora", "prompts"):
        value = ROOT / value
        return io.StringIO(value.read_text())
    return builtins.open(value, *args, **kwargs)


official.open = _prompt_open
official.hashable = lambda values: tuple(values)


def same_configuration(left, right, space):
    left, right = dict(left), dict(right)
    if left.keys() != right.keys():
        return False
    for name in left:
        if isinstance(space[name], FloatHyperparameter):
            if not math.isclose(float(left[name]), float(right[name]), rel_tol=1e-12, abs_tol=1e-15):
                return False
        elif left[name] != right[name]:
            return False
    return True


class MLComment(official.Comment):
    def __init__(self, response, iteration, experiment, *args, **kwargs):
        self._iteration, self._experiment = iteration, experiment
        self._comment, self._hypotheses = None, []
        self.type = official.CommentType.POINTS
        self._keys = experiment.keys
        self._pbounds = experiment.pbounds
        if response is not None:
            try:
                self._process_response(response)
            except (ValueError, KeyError, TypeError, AttributeError):
                self._comment, self._hypotheses = None, []

    def _chat_completion(self, messages):
        return self._experiment.bridge.create(messages=messages).choices[0].message.content

    def _clean_hypothesis_keys(self, hypothesis):
        if not isinstance(hypothesis, dict) or not all(
                k in hypothesis for k in ("name", "rationale", "confidence", "points")):
            return False, {}
        points = hypothesis["points"]
        if isinstance(points, dict):
            self._experiment.owner._record_decision({"stage": "bora_json_normalization",
                "normalization": "single point object to one-element points list",
                "original_point": points})
            points = [points]
        if not isinstance(points, list):
            return False, {}
        result = {k: hypothesis[k] for k in ("name", "rationale", "confidence")}
        result["points"] = []
        for point in points:
            resolver = self._experiment.owner.resolver
            config = resolver.resolve(json.dumps(point))
            self._experiment.owner._record_decision(dict(resolver.last_record, hypothesis=result["name"]))
            result["points"].append(dict(config))
        return bool(result["points"]), result


official.Comment = MLComment


class QwenBridge:
    def __init__(self, owner):
        self.owner = owner
        self.chat = SimpleNamespace(completions=self)

    def _complete(self, messages):
        import torch

        client = self.owner.llm_client
        self.owner.bora_last_raw = ""
        tokenizer = getattr(client, "_tokenizer", None)
        if tokenizer is not None:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            tokens = len(tokenizer.encode(prompt))
            output = int(getattr(client, "max_new_tokens", self.owner.params.get("generator_max_tokens", 4096)))
            if tokens + output > BORA_CONTEXT_TOKENS:
                self.owner._record_decision({"stage": "bora_generation_failed", "reason": "context_limit",
                    "prompt_tokens": tokens, "output_tokens": output, "context_limit": BORA_CONTEXT_TOKENS,
                    "messages": copy.deepcopy(messages), "evaluation_slot_consumed": False})
                return ""
        error = None
        try:
            if hasattr(client, "complete_messages"):
                return client.complete_messages(messages, use_s1b_template=False)
            return client.complete(messages[0]["content"], json.dumps(messages[1:]))
        except torch.cuda.OutOfMemoryError as exc:
            error = str(exc)
        # Release the failed generation's traceback before clearing its cached allocations.
        torch.cuda.empty_cache()
        self.owner._record_decision({"stage": "bora_generation_failed", "reason": "cuda_out_of_memory",
            "error": error, "messages": copy.deepcopy(messages), "evaluation_slot_consumed": False})
        return ""

    def create(self, *, messages, n=1, **kwargs):
        choices = []
        transported = copy.deepcopy(messages)
        for index, message in enumerate(transported):
            if index and message["role"] == "system":
                # Upstream inserts generated draft comments as a mid-chat system message.
                # Qwen requires system at position zero; preserve these drafts as assistant text.
                message["role"] = "assistant"
        for _ in range(n):
            raw = self._complete(transported)
            self.owner.bora_last_raw = raw
            self.owner._record_decision({"stage": "bora_official_chat", "messages": copy.deepcopy(messages),
                                         "qwen_messages": transported, "raw_text": raw, "sample_count": n})
            choices.append(SimpleNamespace(message=SimpleNamespace(content=raw)))
        return SimpleNamespace(choices=choices, usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))


class MLAssistant(official.Assistant):
    def __init__(self, owner):
        self.owner = owner
        cs = owner.config_space
        parameters = [SimpleNamespace(name=hp.name, type=Type.continuous,
                                     get_bounds=lambda: (0, 1)) for hp in cs.get_hyperparameters()]
        bridge = QwenBridge(owner)
        experiment = SimpleNamespace(
            owner=owner, bridge=bridge, name=owner.task_context["task"], dim=len(parameters),
            description=json.dumps(owner.task_context), parameters=parameters, keys=list(cs.keys()),
            pbounds={hp.name: (0, 1) for hp in cs.get_hyperparameters()}, constraint=None,
            type=Type.continuous, default_precision=15, domain="machine learning hyperparameter tuning",
            target=SimpleNamespace(name="validation score (higher is better)"))
        super().__init__("local-qwen", experiment, np.zeros(len(parameters)),
                         str(Path(owner.traj_csv_path).with_suffix(".bora-comments.md")),
                         llm_model="local-qwen", random_seed=owner.seed)
        self._client = bridge

    def _init_log_path(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_model_info(self):
        return {"max_tokens": BORA_CONTEXT_TOKENS, "prompt_tokens_price_per_1k": 0,
                "completion_tokens_price_per_1k": 0}

    def _add_assistant_message(self, message):
        if message is None:
            self.owner._record_decision({"stage": "bora_missing_assistant_message",
                                        "reason": "upstream completion failed"})
            return
        super()._add_assistant_message(message)

    def _get_token_count(self, message):
        client = self.owner.llm_client
        if hasattr(client, "_ensure_loaded"):
            client._ensure_loaded()
            return len(client._tokenizer.encode(message))
        return len(message)

    def _add_parameters_to_prompt(self, prompt):
        return prompt.replace("[parameters_and_bounds]", json.dumps(self.owner.task_context["configspace"]))

    def _summarize_dataset(self, data):
        scores = pd.to_numeric(data["score"], errors="coerce")
        successful = np.isfinite(scores)
        numeric, categories = {}, []
        for hp in self.owner.config_space.get_hyperparameters():
            if hp.name not in data:
                continue
            if isinstance(hp, (IntegerHyperparameter, FloatHyperparameter)):
                numeric[hp.name] = pd.to_numeric(data.loc[successful, hp.name], errors="coerce")
            else:
                for value, group in data.groupby(hp.name, dropna=False, sort=False):
                    values = scores.loc[group.index]
                    measured = values[np.isfinite(values)]
                    categories.append({"parameter": hp.name, "value": value,
                        "attempts": len(group), "successful": len(measured),
                        "failed": len(group) - len(measured),
                        "mean_score": measured.mean(), "best_score": measured.max()})
        numeric["score"] = scores[successful]
        correlations = pd.DataFrame(numeric).corr().reset_index(names="parameter")
        summary = (f"Attempts: {len(data)}; successful: {int(successful.sum())}; "
                   f"failed: {int((~successful).sum())}. Failed scores are excluded.\n"
                   "Numeric correlations (NaN means insufficient variation/data):\n" +
                   correlations.to_string(index=False))
        if categories:
            summary += "\nCategorical measured-score summary:\n" + pd.DataFrame(categories).to_string(index=False)
        self.owner._record_decision({"stage": "bora_history_summary", "attempts": len(data),
                                    "successful": int(successful.sum()), "summary": summary})
        return SimpleNamespace(to_string=lambda **kwargs: summary)

    def _would_context_be_full(self, text):
        history = sum(self._get_token_count(m["content"]) + 8 for m in self._chat_history)
        output = int(self.owner.params.get("generator_max_tokens", 4096))
        return bool(history + self._get_token_count(text) + output > .9 * self._model_info["max_tokens"])

    def _generate_comment_prompt(self, data):
        return super()._generate_comment_prompt(data).replace(
            "The corresponding correlation matrix is:", "The observation summary is:")

    def _generate_comment_selection_prompt(self, data, suggestions):
        return super()._generate_comment_selection_prompt(data, suggestions).replace(
            "The corresponding correlation matrix is:", "The observation summary is:")

    def _generate_starter_prompt(self, n_hypotheses):
        text = (ROOT / "bora/prompts/starter_prompt.txt").read_text()
        defaults = list(dict(self.owner.config_space.get_default_configuration()).items())
        replacements = {"domain": self._experiment.domain, "target": self._experiment.target.name,
                        "n_hypotheses": str(n_hypotheses), "constraint_violation": ""}
        for i, (key, value) in enumerate(defaults[:2], 1):
            replacements.update({f"parameter_{i}": key, f"value_{i}": json.dumps(value)})
        for key, value in replacements.items():
            text = text.replace("[" + key + "]", value)
        return text

    def _constraint_violation(self, comment):
        # Each point has already passed the actual ConfigSpace, including conditions/forbiddens.
        return comment

    def comment_and_select_point(self, data, suggestions):
        # Upstream selection uses numeric np.around; compare actual mixed configurations instead.
        suggestions = SimpleNamespace(to_string=lambda **kwargs: json.dumps(
            [dict(c) for c in self.owner.selection_candidates], indent=2))
        self._add_user_message(self._generate_comment_selection_prompt(data, suggestions))
        allowed = {self.owner._key(c): dict(c) for c in self.owner.selection_candidates}
        self.owner._record_decision({"stage": "bora_selection_candidates", "configurations": list(allowed.values())})
        try:
            for _ in range(3):
                comment = MLComment(self._chat_completion(self._chat_history, n=3),
                                    len(data) + 1, self._experiment)
                if not comment.is_valid:
                    continue
                seen = set()
                for hypothesis in comment.hypotheses:
                    chosen = []
                    for point in hypothesis["points"]:
                        key = self.owner._key(point)
                        if key not in allowed:
                            matches = [k for k, original in allowed.items()
                                       if same_configuration(point, original, self.owner.config_space)]
                            if len(matches) == 1:
                                key = matches[0]
                                self.owner._record_decision({"stage": "bora_selection_float_roundtrip",
                                    "returned": point, "original": allowed[key]})
                        if key in allowed and key not in seen:
                            chosen.append(allowed[key])
                            seen.add(key)
                    hypothesis["points"] = chosen
                comment.hypotheses = [h for h in comment.hypotheses if h["points"]]
                if comment.is_valid:
                    self._comments.append(comment)
                    self._update_comment_log_file()
                    return comment
        finally:
            self._chat_history.pop()
        return None

    def snapshot(self):
        return {"chat_history": self._chat_history, "overview": self._experiment_overview,
                "total_tokens": self._total_tokens,
                "comments": [{"iteration": c.iteration, "comment": c.comment,
                              "hypotheses": c.hypotheses, "type": c.type.name} for c in self._comments]}

    def restore(self, state):
        self._chat_history = [m for m in state["chat_history"] if m.get("content") is not None]
        self._experiment_overview = state["overview"]
        self._total_tokens = state["total_tokens"]
        self._comments = []
        for saved in state["comments"]:
            comment = MLComment(None, saved["iteration"], self._experiment)
            comment._comment, comment._hypotheses = saved["comment"], saved["hypotheses"]
            comment.type = official.CommentType[saved["type"]]
            self._comments.append(comment)


class MLTargetSpace:
    """ConfigSpace codec around the released random plus L-BFGS-B EI optimizer."""
    constraint = None
    _allow_duplicate_points = False
    _discretization_steps = True

    def __init__(self, owner):
        self.owner = owner
        self.sampled_vectors = []
        self.bounds = np.array([(0, len(hp.choices) - 1) if hasattr(hp, "choices") else
                                (0, len(hp.sequence) - 1) if hasattr(hp, "sequence") else
                                (0, 0) if hasattr(hp, "value") else (0, 1)
                                for hp in owner.config_space.get_hyperparameters()], dtype=float)

    def __len__(self):
        return sum(np.isfinite(self.owner.perfs))

    @property
    def empty(self):
        return not len(self)

    def _target_max(self):
        return max(p for p in self.owner.perfs if np.isfinite(p))

    def decode(self, vector):
        vector = np.clip(np.array(vector), self.bounds[:, 0], self.bounds[:, 1])
        for i, hp in enumerate(self.owner.config_space.get_hyperparameters()):
            if hasattr(hp, "choices") or hasattr(hp, "sequence"):
                vector[i] = round(vector[i])
        config = Configuration(self.owner.config_space, vector=vector)
        result = deactivate_inactive_hyperparameters(dict(config), self.owner.config_space)
        self.owner.config_space.check_configuration(result)
        return result

    def discretize_point(self, vector):
        try:
            return self.owner._encode([self.decode(vector)])[0]
        except (ValueError, TypeError):
            return np.full(len(self.bounds), np.nan)

    def __contains__(self, vector):
        try:
            key = self.owner._key(self.decode(vector))
            return key in {self.owner._key(c) for c in self.owner.configs}
        except (ValueError, TypeError):
            return True

    def random_points(self, count):
        vectors = self.owner._encode(self.owner._pool(count))
        self.sampled_vectors.append(vectors)
        return vectors


def official_candidates(owner, count):
    acquisition = ExpectedImprovement(xi=0.01, random_state=owner.rng)
    space = MLTargetSpace(owner)
    vectors = acquisition.suggest(owner.gp, space, n_random=owner.n_pending,
                                  fit_gp=False, n=count)
    result = []
    for vector in vectors:
        candidate = space.decode(vector)
        if not any(same_configuration(candidate, c, owner.config_space) for c in result):
            result.append(candidate)
    if len(result) < count:
        # Projection/L-BFGS-B can collapse several optima onto the same ML configuration.
        # Fill from the already sampled locations, ranked by the same official EI.
        pool = np.concatenate(space.sampled_vectors)
        scores = acquisition._get_acq(owner.gp, constraint=None)(pool)
        for index in np.argsort(scores):
            candidate = space.decode(pool[index])
            if not any(same_configuration(candidate, c, owner.config_space) for c in result):
                result.append(candidate)
                if len(result) == count:
                    break
    return result
