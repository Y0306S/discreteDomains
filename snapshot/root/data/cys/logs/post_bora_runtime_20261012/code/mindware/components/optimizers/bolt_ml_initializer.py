"""BOLT-ML learned initialization from the saved full-SFT model."""

from examples.benchmark.export_bolt_ml_top10 import build_messages
from mindware.components.optimizers.online_llm_optimizer import OnlineLLMOptimizer


MODEL = "/root/data/cys/baseline/artifacts/bolt_ml_top10_full_sft/model"


class BOLTMLInitializer(OnlineLLMOptimizer):
    optimizer_label = "bolt_ml_initialization"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_budget()
        if self.llm_client is None:
            from examples.benchmark.hf_qwen_client import HuggingFaceQwenClient
            self.llm_client = HuggingFaceQwenClient(model_path=MODEL, temperature=1.0,
                                                   max_new_tokens=512, enable_thinking=False)
        self.asset_identity.update(full_sft_model=MODEL, scope="learned initializer validation only")

    def _validate_budget(self):
        if self.evaluation_num_limit is None or self.evaluation_num_limit > 10:
            raise ValueError("Use BOLTMLOptimizer for the full 10-initialization/90-BO run")

    def _suggest(self):
        messages = build_messages(self.task_context["dataset"], self.fixed_algorithm,
                                  self.params["task_type"], self.config_space, {})[:-1]
        raw = self.llm_client.complete_messages(messages, use_s1b_template=False)
        config = dict(self.resolver.resolve(raw))
        self._record_decision(dict(self.resolver.last_record, messages=messages))
        return {"config": config, "note": "BOLT learned initialization"}
