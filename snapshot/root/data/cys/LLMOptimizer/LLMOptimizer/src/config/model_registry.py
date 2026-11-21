import os
import json
from urllib.error import URLError
from urllib.request import urlopen

OPENAI_HK_URL = "https://api.openai-hk.com/v1"

# New API endpoint (PKU lab server)
PKU_LAB_URL = "http://123.129.219.111:3000/v1"
PKU_LAB_API_KEY = os.environ.get("PKU_LAB_API_KEY", "")

# Ollama configuration - now the ONLY local model provider.
# Local setup lives in /root/data/cys/blockdata_archive/ollama/env.sh on this server.
LOCAL_OLLAMA_HOME = os.getenv("OLLAMA_HOME", "/root/data/cys/blockdata_archive/ollama")
LOCAL_OLLAMA_URL = "http://127.0.0.1:11434/v1"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", LOCAL_OLLAMA_URL).rstrip("/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# Approximate conversion used for DeepSeek RMB prices.
RMB_TO_USD = 0.14

# Ollama model mapping - all local models go through Ollama
OLLAMA_MODELS = {
    "llama-3.2-3b": "llama-3.2-3b:latest",
    "llama-3.2-vision": "llama3.2-vision-8k:latest",
    "qwen-3.5-4b": "qwen3.5:4b",
    "qwen-3.5-9b": "qwen3.5:9b",
}

def get_ollama_model_config(model_key):
    """Get configuration for an Ollama model."""
    model_id = OLLAMA_MODELS.get(model_key)
    if not model_id:
        return None
    return {
        "id": model_id,
        "base_url": OLLAMA_URL,
        "source": "ollama",
        "api_key_override": OLLAMA_API_KEY,
        "pricing": None,
    }


def is_ollama_model(model_key: str) -> bool:
    """Return True when a model key is served by the local Ollama server."""
    return model_key in OLLAMA_MODELS


def _ollama_api_root(base_url: str = OLLAMA_URL) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url[:-3].rstrip("/")
    return base_url


def require_ollama_server(model_keys, timeout: float = 3.0) -> None:
    """Fail early if selected Ollama models cannot be served locally."""
    requested_ids = [
        OLLAMA_MODELS[model_key]
        for model_key in model_keys
        if is_ollama_model(model_key)
    ]
    if not requested_ids:
        return

    api_root = _ollama_api_root()
    try:
        with urlopen(f"{api_root}/api/version", timeout=timeout) as response:
            response.read()
    except (OSError, URLError) as exc:
        raise RuntimeError(
            "Ollama model(s) requested but the local Ollama server is not reachable at "
            f"{OLLAMA_URL}. Start it with "
            f"`{LOCAL_OLLAMA_HOME}/start_ollama.sh` or run "
            f"`source {LOCAL_OLLAMA_HOME}/env.sh && ollama serve`."
        ) from exc

    try:
        with urlopen(f"{OLLAMA_URL}/models", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Ollama is reachable at {OLLAMA_URL}, but its model list could not be read."
        ) from exc

    available_ids = {item.get("id") for item in payload.get("data", [])}
    missing_ids = sorted(set(requested_ids) - available_ids)
    if missing_ids:
        raise RuntimeError(
            "Ollama is running, but these required model IDs are missing: "
            f"{missing_ids}. Import them with `{LOCAL_OLLAMA_HOME}/import_models.sh`."
        )

MODEL_CONFIGS = {
    # OpenAI
    "gpt-5.4": {
        "id": "gpt-5.4",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.0025, "output": 0.015},
    },
    "gpt-5-mini": {
        "id": "gpt-5-mini",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.00025, "output": 0.002},
    },
    "gpt-3.5": {
        "id": "gpt-3.5-turbo",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.0015, "output": 0.002},
    },
    # Google
    "gemini-3.1-pro": {
        "id": "gemini-3.1-pro-preview",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.00125, "output": 0.01},
    },
    "gemini-3-flash": {
        "id": "gemini-3-flash-preview",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.0005, "output": 0.003},
    },
    # Anthropic (OpenAI-HK)
    # Pricing: $3/$15 per 1M tokens for Sonnet, $5/$25 per 1M tokens for Opus
    "claude-opus-4-6": {
        "id": "claude-opus-4-6",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.005, "output": 0.025},  # $5/$25 per 1M
    },
    "claude-opus-4-6-thinking": {
        "id": "claude-opus-4-6-thinking",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.005, "output": 0.025},
    },
    "claude-sonnet-4-6": {
        "id": "claude-sonnet-4-6",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.003, "output": 0.015},  # $3/$15 per 1M
    },
    "claude-sonnet-4-6-thinking": {
        "id": "claude-sonnet-4-6-thinking",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.003, "output": 0.015},
    },
    # DeepSeek (RMB prices converted to USD)
    "deepseek-v3.2": {
        "id": "deepseek-v3.2",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.002 * RMB_TO_USD, "output": 0.008 * RMB_TO_USD},
    },
    "deepseek-r1": {
        "id": "deepseek-r1",
        "base_url": OPENAI_HK_URL,
        "source": "api",
        "pricing": {"input": 0.004 * RMB_TO_USD, "output": 0.016 * RMB_TO_USD},
    },
}

# Dynamically add Ollama models to MODEL_CONFIGS
for ollama_key in OLLAMA_MODELS:
    _ollama_config = get_ollama_model_config(ollama_key)
    if _ollama_config:
        MODEL_CONFIGS[ollama_key] = _ollama_config


def calculate_cost(model_key: str, prompt_tokens: int, completion_tokens: int):
    """Return USD cost for a run, or None for non-priced models."""
    pricing = MODEL_CONFIGS[model_key].get("pricing")
    if pricing is None:
        return None

    input_cost = (prompt_tokens / 1000.0) * pricing["input"]
    output_cost = (completion_tokens / 1000.0) * pricing["output"]
    return round(input_cost + output_cost, 6)
