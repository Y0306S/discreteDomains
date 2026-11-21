"""ML oracle adapter for target-only Thompson sampling through fine-tuning."""

import json
import math
import os

from mindware.components.optimizers.online_llm_optimizer import OnlineLLMOptimizer
from mindware.components.optimizers.online_upstream import UPSTREAM, source_digest
from mindware.components.optimizers.strict_recommendation import RecommendationError
from mindware.components.optimizers.tosfit_runtime import ToSFiTRuntime, RECIPE_PATH, published_recipe


class ToSFiTOptimizer(OnlineLLMOptimizer):
    optimizer_label = "tosfit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tosfit_runtime = self.params.get("tosfit_runtime")
        if self.tosfit_runtime is None and os.environ.get("TOSFIT_REMOTE_CONFIG"):
            from mindware.components.optimizers.tosfit_remote import RemoteToSFiTRuntime
            self.tosfit_runtime = RemoteToSFiTRuntime.from_environment(self.seed)
        if self.tosfit_runtime is None:
            device, reference_device = self.params.get("device"), self.params.get("reference_device")
            if device is None or reference_device is None:
                raise ValueError("Specify device and reference_device for ToSFiT before execution")
            self.tosfit_runtime = ToSFiTRuntime(device, reference_device)
        self.prompt = (json.dumps(self.task_context) +
                       '\nRecommend one configuration to maximize validation score. Return '
                       '{"rationale": "...", "config": {all active parameter names and values}}.')
        self.bo_step, self.solutions = 0, []
        self.asset_identity.update(
            recipe=published_recipe(), recipe_sha256=source_digest(RECIPE_PATH),
            numerical_sources={name: source_digest(UPSTREAM / "tosfit/src" / name)
                               for name in ("gaussian_process.py", "lite.py", "utils/misc.py")},
            feature_adapter="upstream token-normalized sequence mean of frozen Qwen plus bias",
            generation_sha256=source_digest(UPSTREAM / "tosfit/src/language_model.py"),
            invalid_reward="no finite reward; exclude from GP",
            incumbent_reevaluation="reuse measured incumbent score; no new evaluation or budget slot")

    def _suggest(self):
        sample = self.tosfit_runtime.propose(self.prompt, self.bo_step)
        try:
            config = dict(self.resolver.resolve(sample["raw_text"]))
            note = ("ConfigSpace default after unrecoverable ToSFiT proposal"
                    if self.resolver.last_record.get("fallback") else "ToSFiT proposal")
        except RecommendationError as exc:
            config, note = None, "rejected ToSFiT proposal: " + str(exc)
        finally:
            self._record_decision(dict(self.resolver.last_record, original_tokens=sample["tokens"],
                                       policy_version=sample["policy_version"],
                                       gradient_step=sample["gradient_step"]))
        return {"config": config, "note": note, "sample": sample}

    def _after_observation(self, proposal, perf):
        sample = proposal["sample"]
        if math.isfinite(perf):
            self.tosfit_runtime.observe(sample["features"], perf)
        self.solutions.append({"config": proposal["config"], "trial_id": self.trial_cnt - 1,
                               "sum": perf if math.isfinite(perf) else 0.0,
                               "count": int(math.isfinite(perf)), "mean": perf})
        self.bo_step += 1
        best = max(self.solutions, key=lambda item: item["mean"])
        if math.isfinite(best["mean"]) and perf < best["mean"]:
            self._record_decision({"stage": "incumbent_cached", "after_trial": self.trial_cnt - 1,
                                   "source_trial": best["trial_id"], "config": best["config"],
                                   "score": best["mean"], "oracle_called": False,
                                   "evaluation_slot_consumed": False})

    def _checkpoint_state(self, directory):
        return self.tosfit_runtime.snapshot(directory)

    def _checkpoint_restore(self, extra):
        super()._checkpoint_restore(extra)
        if self.asset_identity.get("incumbent_reevaluation") != (
                "reuse measured incumbent score; no new evaluation or budget slot"):
            raise RuntimeError("ToSFiT incumbent policy changed; use a fresh task session")
        if extra and extra.get("weights_retired") and self.trial_cnt < self.evaluation_num_limit:
            raise RuntimeError("Cannot extend a completed task after retiring its weights")
        self.tosfit_runtime.restore(extra)

    def _checkpoint_committed(self, extra):
        if extra and extra.get("weights"):
            from pathlib import Path
            current = Path(extra["weights"])
            for path in current.parent.glob("weights-*.pt"):
                if path != current:
                    path.unlink()

    def _finalize_completed_task(self):
        from pathlib import Path
        from mindware.components.optimizers import ml_checkpoint
        session = getattr(self, "_ml_session", None)
        if session is None or session.pending is not None:
            return
        summary = ml_checkpoint.progress(self.traj_rows, int(self.evaluation_num_limit))
        if not summary["budget_exhausted"] or not all(r["oracle_called"] for r in self.traj_rows):
            return
        session.export_trajectory()
        ml_checkpoint.atomic_json(session.directory / "verified_completion.json", summary)
        # Commit a terminal checkpoint without task weights before unlinking those weights.
        self.tosfit_runtime.saved_weights = None
        self.tosfit_runtime.saved_version = self.tosfit_runtime.policy_version
        self.tosfit_runtime.weights_retired = True
        session.save()
        for weights in (session.directory / "tosfit-policy").glob("weights-*.pt"):
            weights.unlink()
