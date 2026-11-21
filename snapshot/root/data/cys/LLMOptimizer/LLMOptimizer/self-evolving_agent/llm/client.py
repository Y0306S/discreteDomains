from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMClientConfig:
    model: str = "gpt-5-mini"
    base_url: str = "https://api.openai-hk.com/v1"
    api_key: str | None = None
    timeout: float = 180.0


class LLMClient:
    def __init__(self, config: LLMClientConfig | None = None):
        self.config = config or LLMClientConfig()
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_HK_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or OPENAI_HK_API_KEY for LLM smoothing.")
        self.client = OpenAI(api_key=api_key, base_url=self.config.base_url, timeout=self.config.timeout)

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.7, n: int = 1) -> Any:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            n=n,
            response_format={"type": "json_object"},
            timeout=self.config.timeout,
        )
        return response
