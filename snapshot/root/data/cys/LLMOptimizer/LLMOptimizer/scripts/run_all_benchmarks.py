#!/usr/bin/env python3
"""
run_all_benchmarks.py
---------------------
Unified script to test and run all benchmarks:
- ML benchmarks (requires hpobench conda env)
- NASBench-201 (local, any env)
- NASBench-101 + Tabular (requires hpobench_101 conda env)

Usage:
    python scripts/run_all_benchmarks.py --test           # Test all connections
    python scripts/run_all_benchmarks.py --mini           # Mini tests (1 iter each)
    python scripts/run_all_benchmarks.py --full           # Full benchmarks
    python scripts/run_all_benchmarks.py --test --mini    # Test then mini
    python scripts/run_all_benchmarks.py --models gpt-5-mini,llama-3.2-3b  # Filter models
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"
ROOT_STR = str(ROOT)


# Configuration
CONDA_BASE = "/jizhicfs/johnnyslin/anaconda3"
RESULTS_DIR = str(EXPERIMENT_RUNS_DIR / "results")

API_MODELS = [
    "gpt-5.4", "gpt-5-mini", "gpt-3.5",
    "gemini-3.1-pro", "gemini-3-flash",
    "claude-opus-4-6", "claude-opus-4-6-thinking",
    "claude-sonnet-4-6", "claude-sonnet-4-6-thinking",
    "deepseek-v3.2", "deepseek-r1",
]

OLLAMA_MODELS = ["llama-3.2-3b", "llama-3.2-vision", "qwen-3.5-4b", "qwen-3.5-9b"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_in_conda(env_name, command, cwd=None):
    """Run a command in a specific conda environment."""
    full_cmd = f"source {CONDA_BASE}/etc/profile.d/conda.sh && conda activate {env_name} && {command}"
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, cwd=cwd, executable="/bin/bash")
    return result


def embedded_repo_path() -> str:
    return repr(ROOT_STR)


def test_api_models(models):
    """Test API model connectivity."""
    log("Testing API models...")
    test_script = """
import os
import sys
sys.path.insert(0, """ + embedded_repo_path() + """)
api_key = os.getenv("OPENAI_HK_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_HK_API_KEY is not set")

from src.config.model_registry import MODEL_CONFIGS
from openai import OpenAI

models = [""" + ",".join(f'"{m}"' for m in models) + """]

for model in models:
    if model not in MODEL_CONFIGS:
        print(f"SKIP: {model} not configured")
        continue
    config = MODEL_CONFIGS[model]
    try:
        client = OpenAI(
            base_url=config["base_url"],
            api_key=config.get("api_key_override") or api_key
        )
        resp = client.chat.completions.create(
            model=config["id"],
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
            timeout=30
        )
        content = resp.choices[0].message.content.strip()
        print(f"OK: {model} -> {content}")
    except Exception as e:
        print(f"FAIL: {model} -> {str(e)[:60]}")
"""
    result = subprocess.run([sys.executable, "-c", test_script], 
	                          capture_output=True, text=True, cwd=ROOT_STR)
    for line in result.stdout.strip().split("\n"):
        if line:
            log(f"  {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[:5]:
            if line:
                log(f"  ERR: {line}")


def test_ollama_models(models):
    """Test Ollama model connectivity."""
    log("Testing Ollama models...")
    test_script = """
import sys
sys.path.insert(0, """ + embedded_repo_path() + """)
from openai import OpenAI
from src.config.model_registry import MODEL_CONFIGS, require_ollama_server

models = [""" + ",".join(f'"{m}"' for m in models) + """]
models = [model for model in models if model in MODEL_CONFIGS]
require_ollama_server(models)

for model in models:
    config = MODEL_CONFIGS[model]
    client = OpenAI(
        base_url=config["base_url"],
        api_key=config.get("api_key_override", "ollama")
    )
    try:
        resp = client.chat.completions.create(
            model=config["id"],
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
            timeout=30
        )
        content = resp.choices[0].message.content.strip()
        print(f"OK: {model} -> {content}")
    except Exception as e:
        print(f"FAIL: {model} -> {str(e)[:60]}")
"""
    result = subprocess.run([sys.executable, "-c", test_script],
	                          capture_output=True, text=True, cwd=ROOT_STR)
    for line in result.stdout.strip().split("\n"):
        if line:
            log(f"  {line}")


def run_ml_benchmarks(models, resume=True):
    """Run ML benchmarks in hpobench environment."""
    log("Running ML benchmarks (hpobench env)...")
    
    model_arg = f"--models {','.join(models)}" if models else ""
    resume_arg = "--resume" if resume else ""
    
    cmd = f"cd {ROOT_STR} && python scripts/run_batch_ml.py {model_arg} {resume_arg}"
    
    result = run_in_conda("hpobench", cmd)
    log(f"  stdout: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
    if result.returncode != 0:
        log(f"  ERROR: return code {result.returncode}")
        log(f"  stderr: {result.stderr[-300:] if len(result.stderr) > 300 else result.stderr}")


def run_nas201_benchmarks(models, resume=True):
    """Run NASBench-201 benchmarks (local env)."""
    log("Running NASBench-201 benchmarks...")
    
    model_arg = f"--models {','.join(models)}" if models else ""
    resume_arg = "--resume" if resume else ""
    
    cmd = f"python scripts/run_batch_nas_201.py {model_arg} {resume_arg}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
	                          cwd=ROOT_STR)
    log(f"  stdout: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
    if result.returncode != 0:
        log(f"  ERROR: return code {result.returncode}")


def run_nas_container_benchmarks(models, resume=True):
    """Run NASBench-101 + Tabular benchmarks in hpobench_101 environment."""
    log("Running NASBench-101 + Tabular benchmarks (hpobench_101 env)...")
    
    model_arg = f"--models {','.join(models)}" if models else ""
    resume_arg = "--resume" if resume else ""
    
    cmd = f"cd {ROOT_STR} && python scripts/run_batch_nas_container.py {model_arg} {resume_arg}"
    
    result = run_in_conda("hpobench_101", cmd)
    log(f"  stdout: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
    if result.returncode != 0:
        log(f"  ERROR: return code {result.returncode}")


def run_mini_tests(api_model="gpt-5-mini", ollama_model="llama-3.2-3b"):
    """Run mini validation tests (1 iteration each)."""
    log("Running mini validation tests...")
    
    # Test API model on ML (hpobench env)
    log(f"  Testing {api_model} on ML (1 iter)...")
    test_script = f"""
import os
import sys
sys.path.insert(0, """ + embedded_repo_path() + """)
api_key = os.getenv("OPENAI_HK_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_HK_API_KEY is not set")
from src.benchmarks.benchmark_runner import run_benchmark
result = run_benchmark("lr", 359955, "{api_model}", api_key, max_iterations=1, time_budget=60)
print(f"Score: {{result.get('best_score', 'N/A')}}")
"""
    result = run_in_conda("hpobench", f"cd {ROOT_STR} && python3 -c {shlex.quote(test_script)}")
    log(f"    {result.stdout.strip()[-100:] if result.stdout else 'No output'}")
    
    # Test Ollama model on ML (hpobench env)
    log(f"  Testing {ollama_model} on ML (1 iter)...")
    test_script = f"""
import os
import sys
sys.path.insert(0, """ + embedded_repo_path() + """)
os.environ["OPENAI_HK_API_KEY"] = "ollama"
from src.benchmarks.benchmark_runner import run_benchmark
result = run_benchmark("lr", 359955, "{ollama_model}", "ollama", max_iterations=1, time_budget=60)
print(f"Score: {{result.get('best_score', 'N/A')}}")
"""
    result = run_in_conda("hpobench", f"cd {ROOT_STR} && python3 -c {shlex.quote(test_script)}")
    log(f"    {result.stdout.strip()[-100:] if result.stdout else 'No output'}")
    
    # Test API model on NAS (hpobench_101 env)
    log(f"  Testing {api_model} on NAS-101 (1 iter)...")
    test_script = f"""
import os
import sys
sys.path.insert(0, """ + embedded_repo_path() + """)
api_key = os.getenv("OPENAI_HK_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_HK_API_KEY is not set")
from src.benchmarks.nas_runner import run_nas_benchmark
result = run_nas_benchmark("NASCifar10A", "{api_model}", api_key, max_iterations=1, time_budget=60)
print(f"Error: {{result.get('best_val_error', 'N/A')}}")
"""
    result = run_in_conda("hpobench_101", f"cd {ROOT_STR} && python3 -c {shlex.quote(test_script)}")
    log(f"    {result.stdout.strip()[-100:] if result.stdout else 'No output'}")
    
    # Test Ollama model on NAS (hpobench_101 env)
    log(f"  Testing {ollama_model} on NAS-101 (1 iter)...")
    test_script = f"""
import os
import sys
sys.path.insert(0, """ + embedded_repo_path() + """)
os.environ["OPENAI_HK_API_KEY"] = "ollama"
from src.benchmarks.nas_runner import run_nas_benchmark
result = run_nas_benchmark("NASCifar10A", "{ollama_model}", "ollama", max_iterations=1, time_budget=60)
print(f"Error: {{result.get('best_val_error', 'N/A')}}")
"""
    result = run_in_conda("hpobench_101", f"cd {ROOT_STR} && python3 -c {shlex.quote(test_script)}")
    log(f"    {result.stdout.strip()[-100:] if result.stdout else 'No output'}")


def main():
    parser = argparse.ArgumentParser(description="Run all benchmarks with unified control")
    parser.add_argument("--test", action="store_true", help="Test all model connections")
    parser.add_argument("--mini", action="store_true", help="Run mini validation tests (1 iteration)")
    parser.add_argument("--full", action="store_true", help="Run full benchmarks")
    parser.add_argument("--ml", action="store_true", help="Run ML benchmarks only")
    parser.add_argument("--nas201", action="store_true", help="Run NAS-201 only")
    parser.add_argument("--nas101", action="store_true", help="Run NAS-101 + Tabular only")
    parser.add_argument("--models", type=str, help="Comma-separated model list")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous runs")
    args = parser.parse_args()
    
    # Default to all if no specific benchmark selected
    if not (args.ml or args.nas201 or args.nas101):
        args.ml = args.nas201 = args.nas101 = True
    
    # Parse model list
    models = args.models.split(",") if args.models else None
    
    log("=" * 60)
    log("LLMOptimizer Unified Benchmark Runner")
    log(f"Models: {models if models else 'ALL'}")
    log(f"ML: {args.ml}, NAS-201: {args.nas201}, NAS-101: {args.nas101}")
    log("=" * 60)
    
    # Test phase
    if args.test:
        test_api_models(models if models else API_MODELS)
        test_ollama_models(models if models else OLLAMA_MODELS)
    
    # Mini tests
    if args.mini:
        run_mini_tests()
    
    # Full benchmarks
    if args.full:
        if args.ml:
            run_ml_benchmarks(models, resume=not args.no_resume)
        if args.nas201:
            run_nas201_benchmarks(models, resume=not args.no_resume)
        if args.nas101:
            run_nas_container_benchmarks(models, resume=not args.no_resume)
    
    log("=" * 60)
    log("Done!")


if __name__ == "__main__":
    main()
