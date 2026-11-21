"""Shared TaskSet CASH prompt text, chat rendering and LLM-reply parsing.

Every TaskSet LLM path (warm-start SFT data, warm-start preset generation, the
in-loop P5 prompt) builds its task/search-space text here so training and
inference prompts cannot drift apart.

Configurations use the exact flat ConfigSpace keys (``family:hp``, including the
``_inv`` / ``_is_zero`` / ``_is_one`` encodings), mirroring the ML prompts.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from mindware.domains.dlhpo.taskset_space import FAMILY_HP, TASKSET_FAMILIES

TASK_CONFIG_DIR = Path("/root/data/cys/google-research/task_set/tasks/configs")
PROTOCOL_TEXT = "1000 training steps, validation evaluated at step 1000, 1 replica"
METRIC_TEXT = (
    "normalized validation loss (min_normalized_valid_outer; 1.0 = validation loss "
    "at initialization), lower_is_better"
)

WARM_START_SYSTEM_PROMPT = (
    "You are an expert deep-learning optimizer warm-start assistant. Given a training "
    "task, metric, and a CASH search space over optimizer families and their "
    "hyperparameters, return one strong initial configuration as strict JSON."
)


# ---------------------------------------------------------------- task specs
def load_task_specs(config_dir: Path = TASK_CONFIG_DIR) -> Dict[str, Dict[str, Any]]:
    """task_id -> {"task_family", "spec"} from the TaskSet sampled-family configs."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(config_dir).glob("*.json")):
        for task_id, (spec, family) in json.loads(path.read_text()).items():
            out[str(task_id)] = {"task_family": str(family), "spec": spec}
    return out


def task_family_of(task_id: str) -> str:
    return str(task_id).rsplit("_seed", 1)[0]


def task_text(task_id: str, specs: Dict[str, Dict[str, Any]]) -> str:
    """Task description only: family and spec. No optimizer is named (CASH)."""
    record = specs.get(task_id)
    if record is None:
        raise KeyError(f"No TaskSet spec for {task_id}")
    return "\n".join([
        f"- task_family: {record['task_family']}",
        f"- task_spec: {json.dumps(record['spec'], sort_keys=True)}",
    ])


# ------------------------------------------------------------ search space
def ordered_hps(cs: ConfigurationSpace) -> List[Any]:
    hps = list(cs.get_hyperparameters())
    head = [hp for hp in hps if hp.name == FAMILY_HP]
    return head + sorted((hp for hp in hps if hp.name != FAMILY_HP), key=lambda hp: hp.name)


def hp_line(hp: Any) -> str:
    if isinstance(hp, CategoricalHyperparameter):
        return f"- {hp.name}: categorical, choices={list(hp.choices)}"
    details = ["float", f"range=[{hp.lower}, {hp.upper}]"]
    if getattr(hp, "log", False):
        details.append("log_scale=True")
    return f"- {hp.name}: " + ", ".join(details)


def search_space_lines(cs: ConfigurationSpace) -> List[str]:
    return [hp_line(hp) for hp in ordered_hps(cs)]


def families(cs: ConfigurationSpace) -> List[str]:
    return [str(c) for c in cs.get_hyperparameter(FAMILY_HP).choices]


def condition_lines(cs: ConfigurationSpace) -> List[str]:
    return [f"- {fam}:* parameters are active only when {FAMILY_HP} == {fam}" for fam in families(cs)]


PARAMETER_NOTES = [
    "- <name>_inv stores 1 - <name> (for example beta1 = 1 - beta1_inv).",
    "- <name>_is_zero=true sets <name> to 0; <name>_is_one=true sets <name> to 1.",
]


def family_params(cs: ConfigurationSpace, family: str) -> List[str]:
    prefix = f"{family}:"
    return sorted(hp.name for hp in cs.get_hyperparameters() if hp.name.startswith(prefix))


# ------------------------------------------------------- warm-start prompt
def warm_start_user_prompt(task_id: str, specs: Dict[str, Dict[str, Any]], cs: ConfigurationSpace) -> str:
    lines = [
        "Recommend one warm-start configuration for this deep-learning training task.",
        "",
        "TASK",
        task_text(task_id, specs),
        "",
        "OPTIMIZATION",
        f"- choose: {FAMILY_HP} and the hyperparameters of that family",
        f"- training: {PROTOCOL_TEXT}",
        f"- metric: {METRIC_TEXT}",
        "",
        "SEARCH_SPACE_PARAMETERS",
        *search_space_lines(cs),
        "",
        "SEARCH_SPACE_CONDITIONS",
        *condition_lines(cs),
        "",
        "PARAMETER_NOTES",
        *PARAMETER_NOTES,
        "",
        "OUTPUT_FORMAT",
        "Return only a JSON object containing the recommended configuration.",
        "Use exact parameter names from SEARCH_SPACE_PARAMETERS.",
        f"Include {FAMILY_HP} and every parameter of the chosen family, and no parameters of other families.",
        "",
        "Example shape:",
        "{",
        f'  "{FAMILY_HP}": "<family>",',
        '  "<family>:key_1": value_1,',
        "  ...",
        "}",
    ]
    return "\n".join(lines)


def warm_start_messages(task_id: str, specs: Dict[str, Dict[str, Any]], cs: ConfigurationSpace) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": WARM_START_SYSTEM_PROMPT},
        {"role": "user", "content": warm_start_user_prompt(task_id, specs, cs)},
    ]


def config_json(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True)


def display_config(config: Dict[str, Any]) -> str:
    """Prompt-history rendering: floats at 6 significant digits (targets keep full precision)."""
    shown = {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in config.items()}
    return json.dumps(shown, sort_keys=True)


# ------------------------------------------------------------ chat render
def render_nothink(messages: Iterable[Dict[str, str]]) -> str:
    """Exactly LlamaFactory's ``qwen3_5_nothink`` prompt (generation prompt, no <think>)."""
    parts = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
        elif role == "assistant":
            parts.append(f"{content}<|im_end|>\n")
        else:
            raise ValueError(f"Unsupported role {role}")
    return "".join(parts)


# ------------------------------------------------------------- reply parse
def extract_json_object(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object in LLM reply")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM reply JSON is not an object")
    return payload


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return {"true": True, "false": False, "1": True, "0": False}.get(text)


def coerce_config(raw: Dict[str, Any], cs: ConfigurationSpace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Map an LLM dict onto a valid CASH configuration, ML ``_dict_to_cfg`` style.

    The family comes from ``optimizer_family`` or, failing that, the key prefixes.
    Missing active parameters take their defaults, out-of-range floats are clipped
    and unknown keys are skipped; every such repair is reported in the notes.
    """
    if isinstance(raw.get("config"), dict):
        raw = raw["config"]
    choices = families(cs)
    family = raw.get(FAMILY_HP)
    if family is None:
        prefixes = sorted({k.split(":", 1)[0] for k in raw if ":" in k} & set(choices))
        if len(prefixes) != 1:
            raise ValueError(f"Cannot infer {FAMILY_HP} from keys {sorted(raw)}")
        family = prefixes[0]
    family = str(family)
    if family not in choices:
        raise ValueError(f"Invalid {FAMILY_HP} '{family}'. Choices: {choices}")
    names = family_params(cs, family)
    values: Dict[str, Any] = {FAMILY_HP: family}
    notes: Dict[str, Any] = {"defaulted": [], "clipped": [], "skipped": [], "other_family": []}
    for name in names:
        hp = cs.get_hyperparameter(name)
        key = name if name in raw else name.split(":", 1)[1] if name.split(":", 1)[1] in raw else None
        if key is None or raw[key] is None or (isinstance(raw[key], float) and math.isnan(raw[key])):
            values[name] = hp.default_value
            notes["defaulted"].append(name)
            continue
        value = raw[key]
        if isinstance(hp, CategoricalHyperparameter):
            match = next((c for c in hp.choices if c == value or str(c) == str(value)
                          or (isinstance(c, bool) and _as_bool(value) is c)), None)
            if match is None:
                values[name] = hp.default_value
                notes["defaulted"].append(name)
            else:
                values[name] = match
            continue
        assert isinstance(hp, UniformFloatHyperparameter)
        number = float(value)
        clipped = min(max(number, float(hp.lower)), float(hp.upper))
        if clipped != number:
            notes["clipped"].append(name)
        values[name] = clipped
    known = set(names) | {n.split(":", 1)[1] for n in names} | {FAMILY_HP}
    all_names = {hp.name for hp in cs.get_hyperparameters()}
    for key in raw:
        if key in known:
            continue
        (notes["other_family"] if key in all_names else notes["skipped"]).append(key)
    Configuration(cs, values=values).is_valid_configuration()
    return values, notes


def parse_reply(text: str, cs: ConfigurationSpace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    payload = extract_json_object(text)
    config, notes = coerce_config(payload, cs)
    return config, notes, payload


def signature(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


__all__ = [
    "FAMILY_HP", "TASKSET_FAMILIES", "TASK_CONFIG_DIR", "WARM_START_SYSTEM_PROMPT",
    "load_task_specs", "task_family_of", "task_text", "search_space_lines", "condition_lines",
    "PARAMETER_NOTES", "family_params", "families", "warm_start_user_prompt", "warm_start_messages",
    "config_json", "display_config", "render_nothink", "extract_json_object", "coerce_config", "parse_reply", "signature",
]
