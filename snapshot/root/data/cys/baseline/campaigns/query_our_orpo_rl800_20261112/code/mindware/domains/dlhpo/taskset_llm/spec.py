"""Exact ML Qwen no-think chat renderer, without TaskSet ConfigSpace dependencies."""
from typing import Iterable,Dict

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
