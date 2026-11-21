"""In-process HuggingFace client for Qwen SFT models used by pt4opt_llm.

Supports:
  - merged full checkpoints (legacy SFT6 path)
  - base Qwen3.5-9B + LoRA adapter (P5 S1a / S1b, etc.)

Exposes a simple complete() API compatible with PT4OptLLMOptimizer's llm_client hook.
"""

from __future__ import annotations

import threading
import hashlib
import os
from pathlib import Path
import warnings
from typing import Any, Dict, Optional

try:
    from .p5_response_format import render_s1b_prompt
except ImportError:
    from p5_response_format import render_s1b_prompt

_DEFAULT_MODEL_PATH = "/root/blockdata/model-weights/llamafactory_qwen35_9b_full_sft_6"
_DEFAULT_BASE_PATH = "/root/eb-public/huggingface-models/Qwen/Qwen3.5-9B"

_CLIENTS: Dict[str, "HuggingFaceQwenClient"] = {}
_LOCK = threading.Lock()


def load_verified_adapter(base, adapter_path, *, is_trainable=False, device_map=None):
    import torch
    from exact_adapter_load import install
    install()
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    weights_path = Path(adapter_path) / "adapter_model.safetensors"
    saved = load_file(str(weights_path), device="cpu")
    adapter_dtypes = {value.dtype for value in saved.values()}
    if adapter_dtypes not in ({torch.float32}, {torch.bfloat16}):
        raise RuntimeError(f"Unsupported saved adapter dtypes: {adapter_dtypes}")
    # SemanticOPT training saves BF16 LoRA weights; the ML adapters use FP32.
    # PEFT's default autocast would promote BF16 adapters and fail exact loading.
    autocast_adapter_dtype = adapter_dtypes == {torch.float32}
    source_prefix = "base_model.model.model.layers."
    target_prefix = "base_model.model.model.language_model.layers."
    text_layout = saved and all(key.startswith(source_prefix) for key in saved)
    multimodal_base = hasattr(getattr(base, "model", None), "language_model")
    mapping = {r"^model\.layers\.": "model.language_model.layers."} if text_layout and multimodal_base else {}
    expected = {target_prefix + key[len(source_prefix):] if mapping else key: value
                for key, value in saved.items()}
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*missing adapter keys.*")
        placement = {} if device_map is None else {'device_map': device_map}
        model = PeftModel.from_pretrained(base, adapter_path, key_mapping=mapping,
                                          is_trainable=is_trainable,
                                          autocast_adapter_dtype=autocast_adapter_dtype, **placement)
    actual = get_peft_model_state_dict(model)
    if not expected or set(expected) != set(actual):
        raise RuntimeError(f"Adapter key mismatch: saved={len(expected)}, loaded={len(actual)}")
    for key, value in expected.items():
        if actual[key].is_meta:
            raise RuntimeError(f'Adapter tensor is not materialized: {key}; use explicit GPU placement')
        loaded = actual[key].detach().cpu()
        if loaded.dtype != value.dtype or loaded.shape != value.shape or not torch.equal(loaded, value):
            raise RuntimeError(f"Adapter tensor failed verification: {key}")
    digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    verification = {"path": str(weights_path), "sha256": digest.hexdigest(),
                    "verified_tensors": len(expected), "key_mapping": mapping, "all_tensors_exact": True}
    print(f"[HF Qwen] Verified {len(expected)}/{len(expected)} adapter tensors; sha256={digest.hexdigest()}", flush=True)
    return model, verification


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
        self.adapter_verification = None
        self._load_lock = threading.Lock()
        self._remote = None
        self.last_generation = None

    def _shared_client(self):
        path = os.environ.get('ML_QWEN_POOL_CONFIG')
        if path and self._remote is None:
            try:
                from .qwen_inference_transport import SharedQwenClient
            except ImportError:
                from qwen_inference_transport import SharedQwenClient
            self._remote = SharedQwenClient(path, self.model_path, self.adapter_path)
        return self._remote

    def inference_state(self):
        client = self._shared_client()
        return dict(client.state) if client else None

    def restore_inference_state(self, state=None, legacy_cuda=None):
        client = self._shared_client()
        if client:
            client.import_state(state, legacy_cuda)
            return True
        return False

    def _ensure_loaded(self):
        remote = self._shared_client()
        if remote:
            remote.ensure_loaded()
            self.adapter_verification = remote.adapter_verification
            return
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

            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            placement = os.environ.get('HF_QWEN_DEVICE_MAP', 'auto')
            if placement != 'auto' and not (placement.startswith('cuda:') and placement[5:].isdigit()):
                raise ValueError('HF_QWEN_DEVICE_MAP must be auto or cuda:<logical index>')
            device_map = 'auto' if placement == 'auto' else {'': placement}
            model = AutoModelForImageTextToText.from_pretrained(
                load_src,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map=device_map,
                max_memory={0: "9GiB", 1: "12GiB"},
                attn_implementation="sdpa",
            )
            if placement != 'auto' and any(p.is_meta for p in model.parameters()):
                raise RuntimeError('Explicit base model loading left unmaterialized tensors')
            base_model = model
            if self.adapter_path:
                model, self.adapter_verification = load_verified_adapter(
                    model, self.adapter_path, device_map=None if placement == 'auto' else device_map)
            base_model.model.visual.cpu()
            torch.cuda.empty_cache()
            self._model = model
            self._model.eval()
            print("[HF Qwen] Model loaded")

    def _device(self):
        return self._model.get_input_embeddings().weight.device

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.complete_messages(messages)

    def complete_messages(self, messages, *, use_s1b_template=True) -> str:
        remote = self._shared_client()
        if remote:
            text = remote.complete_messages(messages, use_s1b_template=use_s1b_template,
                settings={key: getattr(self, key) for key in
                          ('temperature', 'max_new_tokens', 'top_p', 'top_k', 'enable_thinking')})
            self.last_generation = remote.last_response
            return text
        import torch

        self._ensure_loaded()
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

        if use_s1b_template and self.adapter_path and not self.enable_thinking:
            prompt = render_s1b_prompt(self._tokenizer, messages)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device())
        input_tokens = inputs['input_ids'].shape[-1]
        if input_tokens > getattr(self, 'max_input_tokens', float('inf')):
            raise RuntimeError('Full conversation exceeds the validated service context; no truncation performed')
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
            "logits_to_keep": 1,
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

        cache = getattr(self, 'prompt_kv_cache', None)
        cache_stats = None
        if cache is not None:
            # Adapter identity and exact rendered prompt both belong to the key.
            key = (self.adapter_path, hashlib.sha256(prompt.encode()).hexdigest())
            output_ids, cache_stats = cache.generate(self._model, key, generate_kwargs)
        else:
            with torch.no_grad():
                output_ids = self._model.generate(**generate_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        self.last_generation = {'input_tokens': int(input_tokens), 'output_tokens': len(new_tokens),
                                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        if cache_stats is not None:
            self.last_generation['prompt_cache'] = cache_stats
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
