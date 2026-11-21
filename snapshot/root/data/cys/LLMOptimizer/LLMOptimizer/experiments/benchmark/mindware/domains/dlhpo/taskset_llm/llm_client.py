"""Qwen3.5-9B + LoRA client for TaskSet prompts (warm-start presets and P5 turns).

Prompts are rendered with ``spec.render_nothink`` so inference matches the
LlamaFactory ``qwen3_5_nothink`` training template byte for byte (no ``<think>``).
Adapters are loaded with the verified key mapping used by the ML ckpt32 client:
LlamaFactory saves text-layout keys, while the HF Qwen3.5 base is multimodal.
"""

from __future__ import annotations

import hashlib
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mindware.domains.dlhpo.taskset_llm.spec import render_nothink

BASE_MODEL = "/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def load_verified_adapter(base: Any, adapter_path: str) -> tuple:
    import torch
    from peft import PeftConfig, PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    weights_path = Path(adapter_path) / "adapter_model.safetensors"
    saved = load_file(str(weights_path), device="cpu")
    source_prefix = "base_model.model.model.layers."
    target_prefix = "base_model.model.model.language_model.layers."
    text_layout = bool(saved) and all(key.startswith(source_prefix) for key in saved)
    multimodal_base = hasattr(getattr(base, "model", None), "language_model")
    mapping = {r"^model\.layers\.": "model.language_model.layers."} if text_layout and multimodal_base else {}
    expected = {target_prefix + key[len(source_prefix):] if mapping else key: value for key, value in saved.items()}
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*missing adapter keys.*")
        # Allocate and promote LoRA destinations BEFORE copying saved FP32 values.
        # from_pretrained can otherwise copy through BF16 and round irreversibly.
        config = PeftConfig.from_pretrained(adapter_path)
        config.inference_mode = True
        model = PeftModel(base, config, autocast_adapter_dtype=True)
        model.load_adapter(adapter_path, adapter_name="default", key_mapping=mapping,
                           autocast_adapter_dtype=True)
    actual = get_peft_model_state_dict(model)
    if not expected or set(expected) != set(actual):
        raise RuntimeError(f"Adapter key mismatch: saved={len(expected)}, loaded={len(actual)}")
    for key, value in expected.items():
        loaded = actual[key].detach().cpu()
        if (loaded.shape != value.shape or loaded.dtype != value.dtype
                or not torch.isfinite(loaded).all() or not torch.equal(loaded, value)):
            raise RuntimeError(f"Adapter tensor failed verification: {key}")
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    return model, {"path": str(weights_path), "sha256": digest, "verified_tensors": len(expected),
                   "key_mapping": mapping, "exact": True,
                   "saved_dtypes": sorted({str(value.dtype) for value in saved.values()})}


class QwenLoRAClient:
    def __init__(self, adapter_path: Optional[str], *, base_path: str = BASE_MODEL, temperature: float = 0.2,
                 top_p: float = 0.9, top_k: int = 50, max_new_tokens: int = 768,
                 disable_thinking: bool = False):
        self.base_path, self.adapter_path = base_path, adapter_path
        self.temperature, self.top_p, self.top_k = float(temperature), float(top_p), int(top_k)
        self.max_new_tokens = int(max_new_tokens)
        self.disable_thinking = bool(disable_thinking)
        self.adapter_verification: Optional[Dict[str, Any]] = None
        self._model = self._tokenizer = None
        self._lock = threading.Lock()

    def identity(self) -> Dict[str, Any]:
        return {"base": self.base_path, "adapter": self.adapter_path, "temperature": self.temperature,
                "top_p": self.top_p, "top_k": self.top_k, "max_new_tokens": self.max_new_tokens,
                "template": "qwen3_5_explicit_nothink" if self.disable_thinking else "qwen3_5_nothink",
                "adapter_verification": self.adapter_verification}

    def render_prompt(self, messages: Sequence[Dict[str, str]]) -> str:
        prompt = render_nothink(messages)
        if self.disable_thinking:
            # Qwen's native enable_thinking=False generation prefix. The stock
            # checkpoint otherwise starts reasoning even with the SFT prefix.
            prompt += "<think>\n\n</think>\n\n"
        return prompt

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForImageTextToText, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_path or self.base_path, trust_remote_code=True)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            model = AutoModelForImageTextToText.from_pretrained(
                self.base_path, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
            if self.adapter_path:
                model, self.adapter_verification = load_verified_adapter(model, self.adapter_path)
            model.eval()
            self._model = model

    def _stop_ids(self) -> List[int]:
        ids = []
        for name in STOP_TOKENS:
            tid = self._tokenizer.convert_tokens_to_ids(name)
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                ids.append(tid)
        return ids

    def generate(self, messages: Sequence[Dict[str, str]], *, n: int = 1,
                 temperature: Optional[float] = None, seed: Optional[int] = None) -> List[str]:
        import torch

        self._load()
        temperature = self.temperature if temperature is None else float(temperature)
        inputs = self._tokenizer(self.render_prompt(messages), return_tensors="pt").to(next(self._model.parameters()).device)
        kwargs: Dict[str, Any] = dict(inputs, max_new_tokens=self.max_new_tokens, num_return_sequences=int(n),
                                      eos_token_id=self._stop_ids(),
                                      pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=self.top_p, top_k=self.top_k)
        else:
            kwargs.update(do_sample=False)
        # A service serializes calls. Request seeds make sampling independent of
        # the order in which CPU workers reach that service.
        devices = sorted({p.device.index for p in self._model.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices, enabled=seed is not None), torch.no_grad():
            if seed is not None:
                torch.manual_seed(int(seed))
            output = self._model.generate(**kwargs)
        start = inputs["input_ids"].shape[-1]
        return [self._tokenizer.decode(row[start:], skip_special_tokens=True).strip() for row in output]

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate([{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}])[0]


class ScriptedClient:
    """Test double: returns queued replies (or a callable's output) and records prompts."""

    def __init__(self, replies: Any):
        self.replies = replies
        self.calls: List[Dict[str, str]] = []

    def identity(self) -> Dict[str, Any]:
        return {"client": "scripted"}

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        if callable(self.replies):
            return self.replies(system_prompt, user_prompt)
        return self.replies.pop(0)
