#!/usr/bin/env python3
"""Optimized task_set evaluation for DL warm-start predictions.

Runs multiple replicas in parallel, uses reduced training steps for speed,
and extracts median test loss across replicas.
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
TASK_SET_ROOT = Path("/root/data/cys/blockdata_archive/LLMOptimizer/google-research")
PREDICTIONS_PATH = BASE_DIR / "eval_runs/dl_qwen_full_sft_4_dl_predictions.jsonl"
TASK_CONFIGS_PATH = TASK_SET_ROOT / "task_set/tasks/configs/char_rnn_language_model.json"
OPTIMIZER_CONFIGS_PATH = TASK_SET_ROOT / "task_set/optimizers/configs"
OUTPUT_DIR = Path("/tmp/task_set_dl_eval")
CONDA_ACTIVATE = "/root/miniconda3/bin/activate"
CONDA_ENV = "taskset_eval"


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def find_task_seed(target_task: dict, task_configs: dict) -> str:
    """Find the task_set task name matching the dl_test task config."""
    for key, cfg_list in task_configs.items():
        cfg = cfg_list[0]
        if (cfg.get("embed_dim") == target_task.get("embed_dim") and
            cfg.get("vocab_size") == target_task.get("vocab_size") and
            cfg.get("core", [None])[0] == target_task.get("core", [None])[0] and
            cfg.get("core", [None, {}])[1].get("core_dim") == target_task.get("core", [None, {}])[1].get("core_dim") and
            cfg.get("trainable_init") == target_task.get("trainable_init")):
            return key
    return None


def create_optimizer_config(optimizer_family: str, hyperparameters: dict, seed: int, out_dir: Path) -> str:
    """Create a task_set optimizer config and return its name."""
    config_name = f"custom_{optimizer_family}_seed{seed}"
    config_path = out_dir / f"{config_name}.json"
    
    # task_set expects config as [hyperparameters_dict, getter_name]
    payload = [hyperparameters, optimizer_family]
    
    with config_path.open("w") as f:
        json.dump({config_name: payload}, f)
    
    return config_name, str(config_path)


def run_single_replica(args: tuple) -> dict:
    """Run one replica of train_inner. Called by multiprocessing."""
    task_name, optimizer_name, output_dir, training_steps, eval_every_n, replica = args
    
    replica_dir = Path(output_dir) / task_name / optimizer_name / str(training_steps) / str(replica)
    replica_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = (
        f"source {CONDA_ACTIVATE} {CONDA_ENV} && "
        f"cd {TASK_SET_ROOT} && "
        f"python -m task_set.train_inner "
        f"--optimizer_name={optimizer_name} "
        f"--task_name={task_name} "
        f"--output_directory={output_dir} "
        f"--training_steps={training_steps} "
        f"--eval_every_n={eval_every_n} "
        f"--replica={replica}"
    )
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        # Read result file
        result_file = replica_dir / "result"
        if result_file.exists():
            with result_file.open() as f:
                curves = json.load(f)
            # Get final test loss (last eval step, 4th value = test)
            last_step = max(int(k) for k in curves.keys())
            test_loss = curves[str(last_step)][3]
            return {"replica": replica, "test_loss": test_loss, "status": "ok", "last_step": last_step}
        else:
            return {"replica": replica, "status": "failed", "error": "No result file", "stderr": result.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"replica": replica, "status": "timeout"}
    except Exception as e:
        return {"replica": replica, "status": "error", "error": str(e)}


def evaluate_one_sample(sample_idx: int, prediction: dict, gold_record: dict,
                        task_configs: dict, custom_config_dir: Path,
                        training_steps: int, eval_every_n: int,
                        num_replicas: int) -> dict:
    """Evaluate one sample with N replicas in parallel."""
    
    # Extract predicted config
    pred_config = prediction.get("config", {})
    if not pred_config:
        return {"index": sample_idx, "status": "no_prediction"}
    
    # Handle nested config
    if "warm_start_config" in pred_config:
        pred_config = pred_config["warm_start_config"]
    if "config" in pred_config and isinstance(pred_config["config"], dict):
        pred_config = pred_config["config"]
    
    optimizer_family = pred_config.get("optimizer_family")
    hyperparameters = pred_config.get("hyperparameters", {})
    
    if not optimizer_family or not hyperparameters:
        return {"index": sample_idx, "status": "invalid_config"}
    
    # Find matching task
    user_msg = json.loads(gold_record["messages"][1]["content"])
    target_task = user_msg["task"]
    task_name = find_task_seed(target_task, task_configs)
    
    if not task_name:
        return {"index": sample_idx, "status": "task_not_found"}
    
    # Create custom optimizer config
    opt_name, opt_path = create_optimizer_config(optimizer_family, hyperparameters, sample_idx, custom_config_dir)
    
    # Temporarily add custom config to task_set optimizer configs
    # task_set loads ALL .json files directly from the configs directory
    import shutil
    target_opt_file = OPTIMIZER_CONFIGS_PATH / f"custom_{sample_idx}_{opt_name}.json"
    shutil.copy(opt_path, target_opt_file)
    
    # Build args for parallel replicas
    output_dir = OUTPUT_DIR / f"sample_{sample_idx}"
    replica_args = [
        (task_name, opt_name, str(output_dir), training_steps, eval_every_n, r)
        for r in range(num_replicas)
    ]
    
    print(f"[Sample {sample_idx}] Task: {task_name}, Opt: {opt_name}, Replicas: {num_replicas}")
    
    # Run replicas in parallel
    with multiprocessing.Pool(processes=num_replicas) as pool:
        results = pool.map(run_single_replica, replica_args)
    
    # Compute median test loss
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        test_losses = sorted(r["test_loss"] for r in ok_results)
        median_loss = test_losses[len(test_losses) // 2]
        mean_loss = sum(test_losses) / len(test_losses)
    else:
        median_loss = None
        mean_loss = None
    
    return {
        "index": sample_idx,
        "task_name": task_name,
        "optimizer_family": optimizer_family,
        "status": "ok" if ok_results else "all_failed",
        "replicas_ok": len(ok_results),
        "replicas_total": num_replicas,
        "median_test_loss": median_loss,
        "mean_test_loss": mean_loss,
        "all_test_losses": [r.get("test_loss") for r in results],
        "replica_details": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--test-data", type=Path, default=Path("/jizhicfs/pkuhetu/bcxu/LLMOptimizer/bolt_qwen9b_finetune/data/dl_test.jsonl"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--eval-every-n", type=int, default=200)
    parser.add_argument("--num-replicas", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=Path("/tmp/task_set_dl_eval_results.json"))
    args = parser.parse_args()
    
    # Load data
    predictions = load_jsonl(args.predictions)
    gold_records = load_jsonl(args.test_data)
    
    with TASK_CONFIGS_PATH.open() as f:
        task_configs = json.load(f)
    
    # Limit to N samples
    num_samples = min(args.num_samples, len(predictions))
    
    # Directory for custom optimizer configs
    custom_config_dir = OUTPUT_DIR / "custom_optimizers"
    custom_config_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating {num_samples} samples with {args.num_replicas} replicas each")
    print(f"Training steps: {args.training_steps}, Eval every: {args.eval_every_n}")
    print(f"Parallel replicas per sample: {args.num_replicas}")
    print()
    
    all_results = []
    for i in range(num_samples):
        result = evaluate_one_sample(
            sample_idx=i,
            prediction=predictions[i],
            gold_record=gold_records[i],
            task_configs=task_configs,
            custom_config_dir=custom_config_dir,
            training_steps=args.training_steps,
            eval_every_n=args.eval_every_n,
            num_replicas=args.num_replicas,
        )
        all_results.append(result)
        print(f"[Sample {i}] Median test loss: {result.get('median_test_loss')}")
        print()
    
    # Save results
    with args.output_json.open("w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to: {args.output_json}")
    
    # Summary
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    if ok_results:
        medians = [r["median_test_loss"] for r in ok_results]
        print(f"\n=== SUMMARY ===")
        print(f"Samples evaluated: {len(ok_results)}/{num_samples}")
        print(f"Mean of median test losses: {sum(medians)/len(medians):.4f}")
        print(f"Best (lowest) median: {min(medians):.4f}")
        print(f"Worst (highest) median: {max(medians):.4f}")


if __name__ == "__main__":
    main()
