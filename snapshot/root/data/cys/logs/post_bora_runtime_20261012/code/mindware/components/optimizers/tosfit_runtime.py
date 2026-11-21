"""Qwen-compatible ToSFiT numerical runtime; no cross-task model cache."""

import os
from pathlib import Path
import tempfile

import torch
import yaml

from mindware.components.optimizers.online_llm_optimizer import QWEN_BASE
from mindware.components.optimizers.online_upstream import UPSTREAM, tosfit_core, source_digest, load_module


RECIPE_PATH = UPSTREAM / "tosfit/experiments/tosfit/faq/large/tosfit/config.yaml"


def published_recipe():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())
    keys = ("batch_size", "temperature", "max_new_tokens", "bo_batch_size",
            "n_marginal_likelihood_warmup_steps", "ongoing_marginal_likelihood_maximization",
            "exploration_bonus", "nar", "fine_tune_steps/bo_step", "mini_batch_size",
            "learning_rate", "alpha", "momentum", "weight_decay")
    return {key: recipe[key] for key in keys}


class ToSFiTRuntime:
    def __init__(self, device, reference_device, *, model=None, reference=None,
                 tokenizer=None, recipe=None):
        self.device, self.reference_device = device, reference_device
        self.recipe = published_recipe() if recipe is None else dict(recipe)
        if any(self.recipe[key] != 1 for key in ("bo_batch_size", "mini_batch_size", "fine_tune_steps/bo_step")):
            raise ValueError("This adapter implements the released one-step, one-candidate recipe")
        self.model, self.reference, self.tokenizer = model, reference, tokenizer
        self.gp_module, self.misc, self.lite = tosfit_core()
        self.gp, self.prompt_ids = None, None
        self.policy_version, self.saved_version = 0, -1
        self.saved_weights = None
        self.ready = False
        self.weights_retired = False

    def _generate_batch(self):
        upstream = load_module("_ml_tosfit_language", UPSTREAM / "tosfit/src/language_model.py")
        model = self.model

        class LastTokenModel:
            # Qwen's large vocabulary otherwise materializes all prompt logits.
            device = next(model.parameters()).device

            def __call__(self, **kwargs):
                return model(**kwargs, logits_to_keep=1)

        return upstream.generate_tokens_exluding_backprop(
            self.prompt_ids, LastTokenModel(), self.tokenizer,
            temperature=self.recipe["temperature"], batch_size=self.recipe["batch_size"],
            max_new_tokens=self.recipe["max_new_tokens"])

    def _ensure_loaded(self):
        if self.weights_retired:
            raise RuntimeError("Completed task weights were retired; start a fresh task")
        if self.ready:
            return
        if self.model is None:
            from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
            config = AutoConfig.from_pretrained(QWEN_BASE, local_files_only=True)
            if config.model_type != "qwen3_5":
                raise RuntimeError("Expected the untouched Qwen3.5 base")
            self.tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                QWEN_BASE, dtype=torch.bfloat16, local_files_only=True,
                device_map={"": self.device})
            self.reference = AutoModelForImageTextToText.from_pretrained(
                QWEN_BASE, dtype=torch.bfloat16, local_files_only=True,
                device_map={"": self.reference_device})
        if self.reference is self.model or any(
                p.data_ptr() == q.data_ptr()
                for p, q in zip(self.model.parameters(), self.reference.parameters())):
            raise ValueError("ToSFiT requires independent policy and frozen reference parameters")
        self.model.requires_grad_(True)
        self.reference.requires_grad_(False)
        self.model.eval()
        self.reference.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        recipe = self.recipe
        if recipe["momentum"] != 0 or recipe["weight_decay"] != 0:
            raise ValueError("Checkpoint protocol requires the released stateless SGD recipe")
        # Released per-parameter SGD, with removable hooks for persistent slots.
        optimizers = {p: torch.optim.SGD([p], lr=recipe["learning_rate"], maximize=True,
            momentum=0, weight_decay=0, foreach=False)
            for p in self.model.parameters() if p.requires_grad}
        def update_parameter(parameter):
            optimizers[parameter].step()
            optimizers[parameter].zero_grad(set_to_none=True)
        self._gradient_hooks = [p.register_post_accumulate_grad_hook(update_parameter) for p in optimizers]
        self.ready = True

    def close(self):
        # Autograd hooks retain their optimizer's parameters across model unloads.
        for handle in getattr(self, "_gradient_hooks", []):
            handle.remove()
        self._gradient_hooks = []
        self.model = self.reference = self.tokenizer = None
        self.ready = False

    @staticmethod
    def _device(model):
        return next(model.parameters()).device

    def _set_prompt(self, prompt):
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].cpu()
        if self.prompt_ids is not None and not torch.equal(self.prompt_ids, tokens):
            raise RuntimeError("ToSFiT's prior prompt must stay fixed throughout a target run")
        self.prompt_ids = tokens

    def _score(self, model, tokens, feature=False):
        device = self._device(model)
        tokens = tokens.to(device)
        total = torch.cat((self.prompt_ids.to(device), tokens), dim=1)
        length = tokens.shape[1]
        outputs = model(input_ids=total, attention_mask=torch.ones_like(total),
                        use_cache=False, output_hidden_states=feature, return_dict=True,
                        logits_to_keep=length + 1)
        logits = outputs.logits[:, -(length + 1):-1, :]
        # Include the first EOS, exclude any padding after it, even when pad==EOS.
        eos = tokens.eq(self.tokenizer.eos_token_id)
        mask = (eos.cumsum(-1) - eos.long()).eq(0)
        score_mask = mask.clone()
        if length == self.recipe["max_new_tokens"]:
            score_mask[:, -1] = False  # Upstream's forced final EOS has probability one.
        logprob = torch.zeros((), dtype=torch.float64, device=device)
        for left in range(0, length, 32):
            right = min(left + 32, length)
            selected = -torch.nn.functional.cross_entropy(
                logits[:, left:right, :].float().reshape(-1, logits.shape[-1]) / self.recipe["temperature"],
                tokens[:, left:right].reshape(-1), reduction="none")
            logprob = logprob + (selected.double() * score_mask[:, left:right].reshape(-1)).sum()
        embedding = None
        if feature:
            hidden = outputs.hidden_states[-1][:, -length:, :].float()
            hidden = torch.nn.functional.normalize(hidden, dim=-1)
            embedding = (hidden * mask.unsqueeze(-1)).mean(1).double().cpu()
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            embedding = torch.cat((embedding, torch.ones((1, 1), dtype=torch.float64)), dim=1)[0]
        return logprob, embedding

    def propose(self, prompt, step):
        self._ensure_loaded()
        self._set_prompt(prompt)
        recipe = self.recipe
        tokens, ref_log_probs, features = [], [], []
        with torch.no_grad():
            generated, log_probs = self._generate_batch()
            for original in generated.split(1):
                if original.numel() == 0:
                    raise RuntimeError("Qwen generated no tokens")
                ref_log_prob, embedding = self._score(self.reference, original, feature=True)
                tokens.append(original.cpu())
                ref_log_probs.append(ref_log_prob.to(log_probs.device))
                features.append(embedding)
            features = torch.stack(features)
            if self.gp is None:
                self.gp = self.gp_module.LinearGaussianProcess(
                    features.shape[1], nar=recipe["nar"], dtype=torch.float64, device="cpu")
            means, stds = self.gp.posterior_mean(features), self.gp.posterior_var(features).sqrt()
            stds *= recipe["exploration_bonus"]
            advantage = self.lite.pom_objective_advantage_function(
                means.to(log_probs.device), stds.to(log_probs.device), log_probs,
                torch.stack(ref_log_probs), self.lite.inv_vapor_exp, recipe["alpha"])
            if not torch.isfinite(advantage).all():
                raise RuntimeError("Non-finite ToSFiT advantages")
        trained = step >= recipe["n_marginal_likelihood_warmup_steps"]
        if trained:
            self.model.train()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            try:
                for index, original in enumerate(tokens):
                    log_prob, _ = self._score(self.model, original)
                    objective = advantage[index].detach() * log_prob / recipe["batch_size"]
                    if not torch.isfinite(objective):
                        raise RuntimeError("Non-finite ToSFiT objective")
                    objective.backward()
            finally:
                self.model.gradient_checkpointing_disable()
                self.model.eval()
            self.policy_version += 1
        # Match upstream bo_batch_size=1: evaluate the first sample, not a reranked one.
        return {"raw_text": self.tokenizer.decode(tokens[0][0], skip_special_tokens=True),
                "tokens": tokens[0][0].tolist(), "features": features[0].tolist(),
                "policy_version": self.policy_version, "gradient_step": trained,
                "generation_batch_size": recipe["batch_size"],
                "generation_log_probs": log_probs.cpu().tolist(),
                "gp_means": means.cpu().tolist(), "gp_stds": stds.cpu().tolist()}

    def observe(self, features, reward):
        if not torch.isfinite(torch.tensor(reward)):
            return
        self.gp.add_observation(
            torch.tensor(features, dtype=torch.float64), torch.tensor(reward, dtype=torch.float64),
            perform_marginal_likelihood_maximization=True,
            min_obs=self.recipe["n_marginal_likelihood_warmup_steps"])

    def snapshot(self, directory):
        lock_path = os.environ.get("TOSFIT_CHECKPOINT_LOCK")
        if lock_path and self.ready and self.policy_version > 0 and self.saved_version != self.policy_version:
            import fcntl
            Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                return self._snapshot(directory)
        return self._snapshot(directory)

    def _snapshot(self, directory):
        directory = Path(directory) / "tosfit-policy"
        if self.ready and self.policy_version > 0 and self.saved_version != self.policy_version:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / ("weights-%06d.pt" % self.policy_version)
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = stream.name
                torch.save(self.model.state_dict(), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self.saved_weights, self.saved_version = str(path), self.policy_version
        return {"ready": self.ready, "weights": self.saved_weights, "version": self.policy_version,
                "gp": vars(self.gp) if self.gp is not None else None, "prompt_ids": self.prompt_ids,
                "recipe_sha256": source_digest(RECIPE_PATH), "weights_retired": self.weights_retired,
                "gpu_peak_reserved_gib": {str(i): torch.cuda.max_memory_reserved(i) / 2**30
                    for i in range(torch.cuda.device_count())} if torch.cuda.is_initialized() else {}}

    def restore(self, state):
        if state is None:
            return
        if state["recipe_sha256"] != source_digest(RECIPE_PATH):
            raise RuntimeError("Published ToSFiT recipe changed since checkpoint")
        self.weights_retired = state.get("weights_retired", False)
        if self.weights_retired:
            return
        if state["ready"]:
            self._ensure_loaded()
        if state["weights"]:
            weights = torch.load(state["weights"], map_location="cpu", weights_only=True)
            self.model.load_state_dict(weights, strict=True)
        self.policy_version = self.saved_version = state["version"]
        self.saved_weights, self.prompt_ids = state["weights"], state["prompt_ids"]
        if state["gp"] is not None:
            self.gp = self.gp_module.LinearGaussianProcess.__new__(self.gp_module.LinearGaussianProcess)
            vars(self.gp).update(state["gp"])
