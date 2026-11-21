"""In-process HuggingFace client for Qwen SFT models used by pt4opt_llm.

Supports:
  - merged full checkpoints (legacy SFT6 path)
  - base Qwen3.5-9B + LoRA adapter (P5 S1a / S1b, etc.)

Exposes a simple complete() API compatible with PT4OptLLMOptimizer's llm_client hook.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

_DEFAULT_MODEL_PATH = "/root/data/cys/blockdata_archive/model-weights/llamafactory_qwen35_9b_full_sft_6"
_DEFAULT_BASE_PATH = "/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B"

_CLIENTS: Dict[str, "HuggingFaceQwenClient"] = {}
_LOCK = threading.Lock()


class HuggingFaceQwenClient:
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        adapter_path: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 2048,
        top_p: float = 0.9,
        top_k: int = 50,
        enable_thinking: bool = False,
    ):
        # When adapter_path is set, model_path is the base model.
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.enable_thinking = bool(enable_thinking)
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForImageTextToText, AutoTokenizer

            tok_src = self.adapter_path or self.model_path
            load_src = self.model_path
            print(
                f"[HF Qwen] Loading base={load_src}"
                + (f" adapter={self.adapter_path}" if self.adapter_path else "")
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                tok_src,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForImageTextToText.from_pretrained(
                load_src,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            if self.adapter_path:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            self._model.eval()
            print("[HF Qwen] Model loaded")

    def _device(self):
        return next(self._model.parameters()).device

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3.5 chat template supports enable_thinking; SFT used nothink.
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **template_kwargs,
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device())
        # Stop at end-of-turn AND at the start of a leaked next turn.
        # Without this, SFT Qwen often continues as `\nuser\n{...}` (im_start stripped),
        # which broke naive first-{ to last-} JSON parsing.
        eos_ids = []
        for tok_name in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
            tid = self._tokenizer.convert_tokens_to_ids(tok_name)
            if isinstance(tid, int) and tid >= 0 and tid != getattr(self._tokenizer, "unk_token_id", None):
                eos_ids.append(tid)
        if self._tokenizer.eos_token_id is not None:
            eos_ids.append(int(self._tokenizer.eos_token_id))
        seen = set()
        eos_token_id = []
        for tid in eos_ids:
            if tid not in seen:
                seen.add(tid)
                eos_token_id.append(tid)

        generate_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": eos_token_id or self._tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            if self.top_p < 1.0:
                generate_kwargs["top_p"] = self.top_p
            if self.top_k > 0:
                generate_kwargs["top_k"] = self.top_k
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**generate_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for marker in ("\nuser\n", "\nUser\n", "\nassistant\n", "<|im_start|>"):
            cut = text.find(marker)
            if cut >= 0:
                text = text[:cut].rstrip()
                break
        return text


def get_hf_qwen_client(
    model_path: str = _DEFAULT_MODEL_PATH,
    adapter_path: Optional[str] = None,
    temperature: float = 0.2,
    max_new_tokens: int = 2048,
    enable_thinking: bool = False,
) -> HuggingFaceQwenClient:
    # If only adapter is given, default base to Qwen3.5-9B.
    if adapter_path and (
        not model_path
        or model_path == _DEFAULT_MODEL_PATH
        or "full_sft" in (model_path or "")
    ):
        # Prefer explicit base when using LoRA; fall back to adapter's recorded base.
        base = _DEFAULT_BASE_PATH
        try:
            import json
            from pathlib import Path

            cfg = Path(adapter_path) / "adapter_config.json"
            if cfg.is_file():
                recorded = json.loads(cfg.read_text()).get("base_model_name_or_path")
                if recorded:
                    base = recorded
        except Exception:
            pass
        if model_path == _DEFAULT_MODEL_PATH or "full_sft" in (model_path or ""):
            model_path = base

    key = f"{model_path}|{adapter_path or ''}|{temperature}|{max_new_tokens}|{enable_thinking}"
    with _LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = HuggingFaceQwenClient(
                model_path=model_path,
                adapter_path=adapter_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            _CLIENTS[key] = client
        return client


def ollama_chat_healthy(
    model: str = "qwen35-sft6:latest",
    base_url: str = "http://127.0.0.1:11434/v1",
    timeout: float = 30.0,
) -> bool:
    """Return True if Ollama can complete a trivial chat for the given model."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key="ollama", base_url=base_url, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=8,
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        return bool(content)
    except Exception as exc:
        print(f"[Ollama] health check failed: {exc}")
        return False
