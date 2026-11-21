class LLMCONFIGS:
    OPENAI_KEY = __import__("os").environ.get("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = "https://api.openai-hk.com/v1"

    kd_chat_engine = "gpt-4o-mini"
    opt_chat_engine = "gpt-4o-mini"
    
    HF_KEY = ""
    PWC_KEY = ""
    SEARCHAPI_API_KEY = ""
    TAVILY_API_KEY = ""

class CONFIGS:
    ORIGINAL_DATA_DIR = "/root/RAG/General-Transfer/_data/history_data"
    RAG_DATA_DIR = "/root/RAG/General-Transfer/_data/rag_data"
    HISTORY_RUN_SCOPE = None  # optional per-run history subdir (set by benchmark runner)


AVAILABLE_LLMs = {  
    'gpt-5-mini': {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gpt-5-mini"},
    'gpt-5-nano': {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gpt-5-nano"},
    'gpt-4.1-mini': {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gpt-4.1-mini"},
    "gpt-4.1": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gpt-4.1"},
    "gpt-4o": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gpt-4o"},
    "gpt-3.5": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gpt-3.5-turbo"},
    "deepseek-v3": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "deepseek-v3-0324", "temperature": 0.5},
    "gemini-2.5-flash": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gemini-2.5-flash-preview-05-20"},
    "gemini-2.0-flash": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gemini-2.0-flash"},
    "gpt-4o-mini": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "gpt-4o-mini"},

    "deepseek-ai/DeepSeek-V3.2": {"api_key": LLMCONFIGS.OPENAI_KEY, "base_url": LLMCONFIGS.OPENAI_BASE_URL, "model": "deepseek-ai/DeepSeek-V3.2"},
}

LLM_COST_DICT = {
    'gemini-2.5-flash': {
        'promt_cost_per_1k': 0.00213,
        'completion_cost_per_1k': 0.01775,
    },
    'gpt-4.1-mini': {
        'promt_cost_per_1k': 0.00284,
        'completion_cost_per_1k': 0.01136,
    },
    'deepseek-v3': {
        'promt_cost_per_1k': 0.0016,
        'completion_cost_per_1k': 0.0064,
    },
    'gemini-2.0-flash': {
        'promt_cost_per_1k': 0.00071,
        'completion_cost_per_1k': 0.00284,
    },
    'gpt-4o-mini':{
        'promt_cost_per_1k': 0.001065,
        'completion_cost_per_1k': 0.00426,
    },
    'deepseek-ai/DeepSeek-V3.2':{
        'promt_cost_per_1k': 0.002,
        'completion_cost_per_1k': 0.003,
    },
    'gpt-4o':{
        'promt_cost_per_1k': 0.0355,
        'completion_cost_per_1k': 0.1065,
    },
}


TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
}
