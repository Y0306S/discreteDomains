import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"

import urllib.request
import tarfile

# Use the standard per-user hpobench cache directory.
target_dir = os.path.expanduser("~/.local/share/hpobench")
os.makedirs(target_dir, exist_ok=True)

# The official download link from the AutoML research group
url = "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
tar_path = os.path.join(target_dir, "fcnet_tabular_benchmarks.tar.gz")

print("Downloading the FCNet database (approx. 700MB). This might take a few minutes...")
urllib.request.urlretrieve(url, tar_path)

print("Download complete. Extracting files...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=target_dir)

print("Cleaning up the .tar.gz file...")
os.remove(tar_path)

print("Done! You can now run run_tabular.py successfully.")