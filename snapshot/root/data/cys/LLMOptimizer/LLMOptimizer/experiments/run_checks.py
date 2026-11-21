#!/usr/bin/env python3
"""Run source checks and the CPU tests that do not require external assets."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
subprocess.run([sys.executable, str(ROOT / "verify_bundle.py")], check=True, env=environment)
environment["PYTHONPATH"] = os.pathsep.join([
    str(ROOT / "vendor/peptide_native"),
    str(ROOT / "benchmark/tools"),
])
subprocess.run([
    sys.executable, "-m", "unittest", "discover", "-s",
    str(ROOT / "peptide_fsbo/code"), "-p", "test_*.py", "-v",
], check=True, env=environment)
environment["PYTHONPATH"] = os.pathsep.join([
    str(ROOT / "vendor/query_native"),
    str(ROOT / "benchmark/tools"),
])
subprocess.run([
    sys.executable, "-m", "unittest", "-v",
    "test_context_routing", "test_query_resources", "test_replica_isolation",
    "test_retraining", "test_service_recovery",
], cwd=ROOT / "query_fsbo/batch_code", check=True, env=environment)
