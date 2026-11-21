"""Domain prompt packs (P5-style) and JSON action parsing."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ConfigSpace import ConfigurationSpace

from mindware.domains.base import DomainAdapter


def build_domain_messages(
    adapter: DomainAdapter,
    traj: Sequence[Dict[str, Any]],
    *,
    variant: str = "P5",
    top_k: int = 5,
    worst_k: int = 5,
    recent_k: int = 10,
) -> List[Dict[str, str]]:
    schema = adapter.prompt_schema()
    successes = [r for r in traj if r.get("feasible", 1)]
    ordered = sorted(successes, key=lambda r: float(r.get("objective", 1e10)))
    top = ordered[:top_k]
    worst = list(reversed(ordered[-worst_k:])) if worst_k else []
    recent = list(traj[-recent_k:]) if recent_k else []
    user = {
        "variant": variant,
        "schema": schema,
        "top": top,
        "worst": worst,
        "recent": recent,
        "run_summary": {
            "n": len(traj),
            "best": min((float(r.get("objective", 1e10)) for r in successes), default=None),
        },
        "instruction": schema.get("action"),
    }
    system = (
        f"You are an optimizer for the {adapter.domain_id} domain. "
        "Reply with a single JSON object. "
        + str(schema.get("action") or "")
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, default=str)},
    ]


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_semantic_action(text: str, cs: Optional[ConfigurationSpace] = None) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {}
    blob = str(text)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        m = _JSON_RE.search(blob)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    if not isinstance(obj, dict):
        return {}
    if "config" in obj and isinstance(obj["config"], dict):
        obj = obj["config"]
    names = {hp.name for hp in cs.get_hyperparameters()} if cs is not None else set(obj)
    return {k: v for k, v in obj.items() if k in names or not names}
"""Domain prompt packs (P5-style) and JSON action parsing."""
