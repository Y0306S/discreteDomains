from __future__ import annotations

import json
from typing import Any


def best_config_prompt(task_description: str, config_space_text: str) -> list[dict[str, str]]:
    user = f"""
You are an expert AutoML HPO assistant.
Task description:
{task_description}

Hyperparameter space:
{config_space_text}

Return exactly one JSON object:
{{
  "best_config": {{"param": value, "...": "..."}}
}}
"""
    return [
        {"role": "system", "content": "You output only valid JSON."},
        {"role": "user", "content": user.strip()},
    ]


def trajectory_smoothing_prompt(
    task_description: str,
    config_space_text: str,
    round_summaries: list[dict[str, Any]],
    trajectory_prefix: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user = {
        "task_description": task_description,
        "config_space": config_space_text,
        "round_summaries": round_summaries,
        "trajectory_prefix": trajectory_prefix,
        "instructions": [
            "Use only configs and losses that appear in the prefix.",
            "Return a trajectory that flows naturally from worse to better configurations.",
            "For every transition, include a rationale and per-parameter change rationales.",
            "Do not invent new losses, configs, or step ids.",
        ],
        "output_shape": {
            "trajectory": [
                {
                    "step": 0,
                    "config": {},
                    "loss": 0.0,
                    "rationale": "",
                    "change_rationales": [
                        {
                            "from_step": 0,
                            "to_step": 1,
                            "changed": {},
                            "reason": ""
                        }
                    ],
                }
            ]
        },
    }
    return [
        {"role": "system", "content": "You output only valid JSON."},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]
