import sys
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"


# Add HPOBench to path
sys.path.insert(0, '/jizhicfs/pkuhetu/bcxu/HPOBench')

os.environ['OPENAI_API_KEY'] = 'hpobench'

from src.benchmarks.benchmark_runner import run_benchmark

model = sys.argv[1]
print(f"Testing model: {model}")

try:
    result = run_benchmark(
        model_name='LR',
        task_id=359955,
        api_key='hpobench',
        llm_model=model,
        base_url='http://123.129.219.111:3000/v1',
        model_key=model,
    )
    print(f"SUCCESS: {model}")
    print(f"  Recommendations: {len(result.get('recommendations', []))}")
except Exception as e:
    print(f"FAILED: {model}")
    print(f"  Error: {str(e)[:100]}")
