#!/usr/bin/env python3
"""
Download FCNet tabular benchmark data files.
"""

import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"

import urllib.request
import zipfile

# Data directory
DATA_DIR = "/root/.local/share/hpobench/fcnet_tabular_benchmarks"
os.makedirs(DATA_DIR, exist_ok=True)

# FCNet benchmark files from Figshare
# Based on https://github.com/automl/nas_benchmarks and HPOBench documentation
FCNET_FILES = {
    "fcnet_slice_localization_data.hdf5": "https://ndownloader.figshare.com/files/11412750",
    "fcnet_protein_structure_data.hdf5": "https://ndownloader.figshare.com/files/11412747",
    "fcnet_naval_propulsion_data.hdf5": "https://ndownloader.figshare.com/files/11412744",
    "fcnet_parkinsons_telemonitoring_data.hdf5": "https://ndownloader.figshare.com/files/11412741",
}

def download_file(filename, url):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists ({os.path.getsize(filepath) / 1024 / 1024:.1f} MB)")
        return True
    
    print(f"\nDownloading {filename}...")
    print(f"  URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  ✓ Downloaded ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        # Clean up partial download
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def main():
    print("=" * 70)
    print("FCNet Tabular Benchmark Data Downloader")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print()
    
    success_count = 0
    for filename, url in FCNET_FILES.items():
        if download_file(filename, url):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"Downloaded {success_count}/{len(FCNET_FILES)} files")
    print("=" * 70)
    
    # List files
    print("\nFiles in data directory:")
    for f in sorted(os.listdir(DATA_DIR)):
        filepath = os.path.join(DATA_DIR, f)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
