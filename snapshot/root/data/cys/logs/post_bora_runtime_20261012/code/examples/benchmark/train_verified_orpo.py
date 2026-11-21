"""Run the approved ORPO recipe with frozen inputs and checked SFT inheritance."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import traceback
import warnings


PARENT_SHA256 = "b81ee340fb24a3d4638574eeb447376447a6dcc095575a8599894234c7d6ad80"
DATA_SHA256 = "c68e9be131489f0a766e6af3e74e72bcdb6c7fcce759eca23f41093f4c9429d0"
DATASET = "p5_attr_t2_rationale"
PAIRS = 960
STEPS = 60


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_recipe(source, config):
    allowed = {"dataset_dir", "dataset", "output_dir", "overwrite_output_dir", "deepspeed"}
    differences = {key: [source.get(key), config.get(key)]
                   for key in source.keys() | config.keys() if source.get(key) != config.get(key)}
    unexpected = set(differences) - allowed
    if unexpected:
        raise ValueError(f"Unapproved recipe changes: {sorted(unexpected)}")
    if config.get("stage") != "dpo" or config.get("pref_loss") != "orpo":
        raise ValueError("This launch requires the LlamaFactory ORPO objective")
    if config.get("dataset") != DATASET or config.get("overwrite_output_dir"):
        raise ValueError("Wrong dataset or destructive output policy")
    if config.get("resume_from_checkpoint") is not None:
        raise ValueError("This is a fresh ORPO run from the verified SFT parent")
    return differences


def check_adapter_states(saved, actual):
    import torch

    if not saved or set(saved) != set(actual):
        raise ValueError(f"Adapter key mismatch: saved={len(saved)}, actual={len(actual)}")
    for key, value in saved.items():
        current = actual[key].detach().cpu()
        if not torch.isfinite(current).all():
            raise ValueError(f"Nonfinite adapter: {key}")
        if current.shape != value.shape or not torch.equal(current, value.to(current.dtype)):
            raise ValueError(f"Adapter tensor mismatch: {key}")
    return len(saved)


def freeze_inputs(config, source_path, config_path, differences):
    output = Path(config["output_dir"])
    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=False)
    source_data = Path(config["dataset_dir"])
    source_parent = Path(config["adapter_name_or_path"])
    shutil.copy2(source_path, provenance / "source_recipe.yaml")
    shutil.copy2(config_path, provenance / "launch_recipe.yaml")
    shutil.copy2(__file__, provenance / "train_verified_orpo.py")
    shutil.copy2(config["deepspeed"], provenance / "deepspeed.json")
    data = provenance / "dataset"
    data.mkdir()
    for name in ("dataset_info.json", "p5_attr_t2_rationale_train.jsonl",
                 "manifest.json", "validation.json", "quality_review.json",
                 "train_manifest.jsonl", "TRAINING_FORMAT.md"):
        shutil.copy2(source_data / name, data / name)
    parent = provenance / "parent_adapter"
    parent.mkdir()
    for path in source_parent.iterdir():
        if path.is_file() and (path.suffix in {".json", ".jinja", ".safetensors"}):
            shutil.copy2(path, parent / path.name)
    if sha256(parent / "adapter_model.safetensors") != PARENT_SHA256:
        raise ValueError("Frozen parent checksum mismatch")
    if sha256(data / "p5_attr_t2_rationale_train.jsonl") != DATA_SHA256:
        raise ValueError("Frozen dataset checksum mismatch")

    import llamafactory
    import transformers
    import peft
    import trl

    code_files = {}
    with tarfile.open(provenance / "training_code.tar.gz", "w:gz") as archive:
        lf_root = Path(llamafactory.__file__).parent
        for path in sorted(lf_root.rglob("*.py")):
            relative = "llamafactory/" + str(path.relative_to(lf_root))
            archive.add(path, arcname=relative)
            code_files[relative] = sha256(path)
        for package, relative in ((transformers, "trainer.py"),
                                  (peft, "peft_model.py"),
                                  (peft, "utils/save_and_load.py"),
                                  (trl, "trainer/dpo_trainer.py")):
            path = Path(package.__file__).parent / relative
            name = package.__name__ + "/" + relative
            archive.add(path, arcname=name)
            code_files[name] = sha256(path)

    base = Path(config["model_name_or_path"])
    base_hashes = {path.name: sha256(path) for path in sorted(base.iterdir())
                   if path.is_file() and path.suffix in {".json", ".jinja", ".safetensors"}}
    input_hashes = {str(path.relative_to(provenance)): sha256(path)
                    for path in sorted(provenance.rglob("*")) if path.is_file()}
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "peft", "trl", "accelerate", "deepspeed", "datasets")}
    write_json(provenance / "manifest.json", {
        "created_utc": now(), "source_recipe": str(source_path),
        "approved_config_differences": differences, "original_config": config,
        "parent_sha256": PARENT_SHA256, "training_sha256": DATA_SHA256,
        "base_files_sha256": base_hashes, "input_files_sha256": input_hashes,
        "training_code_sha256": code_files, "versions": versions,
        "world_size": 2, "effective_batch_size": 16,
        "expected_pairs": PAIRS, "expected_optimizer_steps": STEPS,
        "response_format": "rationale_json", "rl_launched": False,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-recipe", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.sched_setaffinity(0, set(range(local_rank * 4, local_rank * 4 + 4)))

    import yaml
    config = yaml.safe_load(args.config.read_text())
    source = yaml.safe_load(args.source_recipe.read_text())
    differences = validate_recipe(source, config)
    parent = Path(config["adapter_name_or_path"])
    data = Path(config["dataset_dir"])
    if sha256(parent / "adapter_model.safetensors") != PARENT_SHA256:
        raise ValueError("Source SFT parent checksum mismatch")
    if sha256(data / "p5_attr_t2_rationale_train.jsonl") != DATA_SHA256:
        raise ValueError("Source training data checksum mismatch")
    rows = [json.loads(line) for line in (data / "p5_attr_t2_rationale_train.jsonl").read_text().splitlines()]
    if len(rows) != PAIRS or any(set(row) != {"conversations", "chosen", "rejected"} for row in rows):
        raise ValueError("Unexpected pair count or preference metadata in training view")
    validation = json.loads((data / "validation.json").read_text())
    if not validation["passed"] or validation["training_file_sha256"] != DATA_SHA256:
        raise ValueError("Dataset validation does not match training inputs")
    if args.check_only:
        print(json.dumps({"passed": True, "parent_sha256": PARENT_SHA256,
                          "pairs": PAIRS, "differences": differences}, indent=2))
        return

    import torch
    import torch.distributed as dist
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise ValueError("This approved run requires two GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    output = Path(config["output_dir"])
    if rank == 0:
        freeze_inputs(config, args.source_recipe, args.config, differences)
    dist.barrier()
    provenance = output / "provenance"
    config["dataset_dir"] = str(provenance / "dataset")
    config["adapter_name_or_path"] = str(provenance / "parent_adapter")
    config["deepspeed"] = str(provenance / "deepspeed.json")
    if rank == 0:
        (provenance / "effective_recipe.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    saved = load_file(str(provenance / "parent_adapter/adapter_model.safetensors"))

    class VerificationCallback(TrainerCallback):
        def record(self, event, **details):
            value = {"event": event, "time_utc": now(), "rank": rank, **details}
            with (output / f"verification_rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
                stream.flush()
            print("ORPO_VERIFY " + json.dumps(value, allow_nan=False), flush=True)
            if rank == 0:
                write_json(output / "run_status.json", value)

        def adapter_state(self, model):
            import deepspeed
            parameters = [value for name, value in model.named_parameters() if "lora_" in name]
            partitioned = [value for value in parameters if hasattr(value, "ds_id")]
            context = deepspeed.zero.GatheredParameters(partitioned, modifier_rank=None) if partitioned else nullcontext()
            with context:
                return {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}

        def verify(self, event, model):
            current = self.adapter_state(model)
            count = check_adapter_states(saved, current)
            if any("lora_" not in name for name, value in model.named_parameters() if value.requires_grad):
                raise ValueError("Non-adapter parameters are trainable")
            self.record(event, verified_tensors=count, parent_sha256=PARENT_SHA256,
                        cpu_affinity=sorted(os.sched_getaffinity(0)))

        def on_init_end(self, args, state, control, model=None, **kwargs):
            self.verify("parent_verified_after_load", model)

        def on_train_begin(self, args, state, control, model=None, train_dataloader=None, **kwargs):
            self.verify("parent_verified_after_deepspeed", model)
            batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * dist.get_world_size()
            if batch != 16 or state.max_steps != STEPS or len(train_dataloader.dataset) != PAIRS:
                raise ValueError("Effective training budget does not match the approved recipe")
            self.record("training_started", step=0, total_steps=state.max_steps,
                        pairs=len(train_dataloader.dataset), effective_batch=batch)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if state.global_step in (1, 2):
                current = self.adapter_state(model)
                if not all(torch.isfinite(value).all() for value in current.values()):
                    raise ValueError("Training produced nonfinite adapter weights")
                changed = sum(not torch.equal(value, saved[key].to(value.dtype)) for key, value in current.items())
                if state.global_step == 2 and not changed:
                    raise ValueError("The adapter did not update after the first nonzero-LR step")
                self.record("initial_update_checked", step=state.global_step,
                            total_steps=state.max_steps, finite=True, changed_tensors=changed)
                control.should_log = True
            else:
                self.record("step_completed", step=state.global_step, total_steps=state.max_steps)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and any(not math.isfinite(value) for value in logs.values() if isinstance(value, (float, int))):
                raise ValueError(f"Nonfinite training metrics: {logs}")
            if logs:
                self.record("metrics", step=state.global_step, total_steps=state.max_steps, metrics=logs)

        def on_save(self, args, state, control, **kwargs):
            self.record("checkpoint_saved", step=state.global_step, total_steps=state.max_steps)

    callback = VerificationCallback()
    try:
        warnings.filterwarnings("error", message=".*missing adapter keys.*")
        run_exp(args=config, callbacks=[callback])
        if rank == 0:
            final = load_file(str(output / "adapter_model.safetensors"))
            state = json.loads((output / "trainer_state.json").read_text())
            if state["global_step"] != STEPS or set(final) != set(saved):
                raise ValueError("Final checkpoint has an unexpected step count or tensor set")
            if not all(torch.isfinite(value).all() for value in final.values()):
                raise ValueError("Final checkpoint contains nonfinite tensors")
            if not any(not torch.equal(value, saved[key].to(value.dtype)) for key, value in final.items()):
                raise ValueError("Final checkpoint did not change from SFT parent")
            callback.record("complete", step=STEPS, total_steps=STEPS,
                            adapter_sha256=sha256(output / "adapter_model.safetensors"))
    except BaseException as error:
        callback.record("failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
