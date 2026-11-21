"""Allocate FP32 LoRA destinations before loading saved FP32 training weights."""
from functools import wraps
from pathlib import Path


def install():
    from peft import PeftConfig, PeftModel
    original = PeftModel.load_adapter

    @wraps(original)
    def load_adapter(self, model_id, adapter_name, is_trainable=False, **kwargs):
        if Path(model_id).is_dir() and kwargs.get('autocast_adapter_dtype', True):
            if adapter_name not in self.peft_config:
                config = PeftConfig.from_pretrained(model_id)
                config.inference_mode = not is_trainable
                self.add_adapter(adapter_name, config)
            self.base_model._cast_adapter_dtype(adapter_name=adapter_name, autocast_adapter_dtype=True)
        return original(self, model_id, adapter_name, is_trainable=is_trainable, **kwargs)

    PeftModel.load_adapter = load_adapter
