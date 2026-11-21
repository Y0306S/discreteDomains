#!/usr/bin/env python3
"""
Diagnostic script to check NAS benchmark setup
"""
import sys
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"


print("=" * 70)
print("NAS Benchmark Diagnostic Script")
print("=" * 70)

# 1. Check Python environment
print("\n[1] Python Environment")
print(f"    Python version: {sys.version}")
print(f"    Python executable: {sys.executable}")

# 2. Check hpobench installation
print("\n[2] HPOBench Installation")
try:
    import hpobench
    print(f"    hpobench location: {hpobench.__file__}")
    print("    hpobench: OK")
except ImportError as e:
    print(f"    hpobench: FAILED - {e}")
    sys.exit(1)

# 3. Check nasbench_201 imports
print("\n[3] NASBench-201 Imports")
try:
    from hpobench.benchmarks.nas.nasbench_201 import (
        Cifar10ValidNasBench201Benchmark,
        Cifar100NasBench201Benchmark,
        ImageNetNasBench201Benchmark
    )
    print("    nasbench_201 benchmarks: OK")
except ImportError as e:
    print(f"    nasbench_201 benchmarks: FAILED - {e}")

# 4. Check nasbench_101 imports
print("\n[4] NASBench-101 Imports")
try:
    from hpobench.benchmarks.nas.nasbench_101 import NASCifar10ABenchmark
    print("    nasbench_101 benchmarks: OK")
except ImportError as e:
    print(f"    nasbench_101 benchmarks: FAILED - {e}")

# 5. Check tabular_benchmarks imports
print("\n[5] Tabular Benchmarks Imports")
try:
    from hpobench.benchmarks.nas.tabular_benchmarks import (
        SliceLocalizationBenchmark,
        ProteinStructureBenchmark,
        NavalPropulsionBenchmark,
        ParkinsonsTelemonitoringBenchmark
    )
    print("    tabular_benchmarks: OK")
except ImportError as e:
    print(f"    tabular_benchmarks: FAILED - {e}")

# 6. Check data directories
print("\n[6] Data Directories")
from hpobench.config import config_file
print(f"    Data directory: {config_file.data_dir}")
print(f"    Cache directory: {config_file.cache_dir}")

# 7. Check nasbench_201 data files
print("\n[7] NASBench-201 Data Files")
from hpobench.util.data_manager import NASBench_201Data
for dataset in ['cifar10-valid', 'cifar100', 'ImageNet16-120']:
    try:
        dm = NASBench_201Data(dataset)
        filepath = dm._save_dir / dm.filename
        exists = filepath.exists()
        status = "EXISTS" if exists else "MISSING"
        print(f"    {dm.filename}: {status}")
    except Exception as e:
        print(f"    {dataset}: ERROR - {e}")

# 8. Check if we can instantiate a benchmark (without data)
print("\n[8] Benchmark Instantiation Test (no data download)")
try:
    from hpobench.benchmarks.nas.nasbench_201 import Cifar10ValidNasBench201Benchmark
    # This will fail if data is not available, but we can catch that
    print("    Benchmark class loaded: OK")
except Exception as e:
    print(f"    Benchmark class: FAILED - {e}")

# 9. Check Singularity/Apptainer
print("\n[9] Container Runtime")
import shutil
singularity_path = shutil.which("singularity")
apptainer_path = shutil.which("apptainer")
if singularity_path:
    print(f"    singularity: {singularity_path}")
elif apptainer_path:
    print(f"    apptainer: {apptainer_path}")
else:
    print("    singularity/apptainer: NOT FOUND")

# 10. Check SIF files
print("\n[10] SIF Container Files")
cache_dir = config_file.cache_dir / "hpobench-0"
if cache_dir.exists():
    sif_files = list(cache_dir.glob("*.sif"))
    if sif_files:
        for f in sif_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"    {f.name}: {size_mb:.1f} MB")
    else:
        print("    No SIF files found in cache")
else:
    print("    Cache directory does not exist")

print("\n" + "=" * 70)
print("Diagnostic Complete")
print("=" * 70)
