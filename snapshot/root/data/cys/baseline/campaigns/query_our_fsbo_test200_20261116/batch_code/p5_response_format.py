"""The S1b response contract shared by preference data and offline RL."""

import json

RESPONSE_FORMAT = "rationale_json"
FORMAT_VERSION = "s1b_rationale_json_v1"
CHAT_TEMPLATE_VERSION = "llamafactory_qwen3_5_nothink_v1"
OUTPUT_INSTRUCTION = (
    "Return exactly this format:\n"
    "Rationale: <short reason for the next configuration>\n\n"
    'JSON:\n{"config": {"<hyperparameter_name>": "<value>"}}\n'
    "Return values for every tunable hyperparameter listed in Search space. "
    "Do not return a configuration that duplicates any Selected historical trial. "
    "Use the exact hyperparameter names shown in Search space."
)


def render_s1b_prompt(tokenizer, messages):
    """Match LlamaFactory qwen3_5_nothink at the answer boundary."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # Native Qwen's disabled-thinking mode still adds this empty thinking block.
    prompt = prompt.removesuffix("<think>\n\n</think>\n\n")
    if not prompt.endswith("<|im_start|>assistant\n"):
        raise ValueError("Tokenizer does not match the S1b no-thinking answer boundary")
    return prompt


def format_response(rationale, config):
    rationale = rationale.strip()
    if not rationale or "\nJSON:" in rationale or "<think>" in rationale:
        raise ValueError("Invalid rationale text")
    return "Rationale: " + rationale + "\n\nJSON:\n" + json.dumps(
        {"config": config}, ensure_ascii=True, allow_nan=False
    )


def parse_response(text):
    """Strict export validation; live inference retains its tolerant parser."""
    text = text.strip()
    if not text.startswith("Rationale: "):
        raise ValueError("Missing Rationale: header")
    rationale, separator, body = text[len("Rationale: "):].partition("\n\nJSON:\n")
    if not separator or not rationale.strip():
        raise ValueError("Missing rationale or JSON: separator")
    payload = json.loads(body)
    if not isinstance(payload, dict) or set(payload) != {"config"}:
        raise ValueError("Expected exactly one config field")
    if not isinstance(payload["config"], dict) or not payload["config"]:
        raise ValueError("Configuration must be a nonempty object")
    return rationale.strip(), payload["config"]


def rationale_conversations(conversations):
    """Convert the known T2 JSON-only prompt without changing its evidence."""
    system, user = conversations
    if system["from"] != "system" or user["from"] != "human":
        raise ValueError("Unexpected T2 conversation structure")
    old = "Return only one JSON object containing the configuration."
    if system["value"].count(old) != 1:
        raise ValueError("Unknown system output instruction")
    context, separator, suffix = user["value"].partition("\n\nReturn only this JSON object:\n")
    expected_suffix = (
        '{"config": {"<hyperparameter_name>": "<value>"}}\n'
        "Return values for every tunable hyperparameter listed in Search space. "
        "Do not return a configuration that duplicates any Selected historical trial. "
        "Use the exact hyperparameter names shown in Search space. No rationale."
    )
    if not separator or suffix != expected_suffix:
        raise ValueError("Unknown user output instruction")
    return [
        {"from": "system", "value": system["value"].replace(
            old, "First write a short rationale, then write one JSON object containing the configuration.")},
        {"from": "human", "value": context + "\n\n" + OUTPUT_INSTRUCTION},
    ], context


def validate_rollout_format(row, expected):
    if expected is None:
        return
    if row.get("response_format") != expected:
        raise ValueError("Rollout response format differs from the training recipe")
    if expected == RESPONSE_FORMAT and row.get("chat_template_contract") != CHAT_TEMPLATE_VERSION:
        raise ValueError("Rollout lacks the matching S1b chat-template contract")
    prompt = "\n".join(message.get("content", "") for message in row.get("messages", []))
    if expected == RESPONSE_FORMAT and (
        "Rationale:" not in prompt or "JSON:" not in prompt or "No rationale." in prompt
    ):
        raise ValueError("Rollout prompt contradicts the rationale-plus-JSON contract")
