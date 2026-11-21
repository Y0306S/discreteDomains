#!/usr/bin/env python3
"""Download SUN397.tar.gz + Partitions.zip for TFDS sun397/tfds builder.

Source: https://huggingface.co/zhengli97/SUN397_Dataset

Download order:
  1. hf download + HF_XET_HIGH_PERFORMANCE=1 (direct huggingface.co + hf-xet)
  2. hfd.sh + HF_ENDPOINT=https://hf-mirror.com (per repo README)
  3. huggingface_hub + hf-mirror fallback
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("/root/data/cys/tensorflow_datasets")
HF_REPO = "zhengli97/SUN397_Dataset"
# Uploaded as a model repo on HF (not datasets/), despite the name.
HF_REPO_TYPE = "model"
HF_MIRROR = "https://hf-mirror.com"
DEFAULT_MANUAL_DIR = DATA_DIR / "downloads" / "manual" / "sun397"
HFD_SCRIPT = Path(__file__).resolve().parent / "hfd.sh"

REQUIRED_FILES = ("SUN397.tar.gz", "Partitions.zip")
# SUN397.tar.gz is ~39.1 GB on HF; Partitions.zip ~18 MB
MIN_BYTES = {
    "SUN397.tar.gz": 30_000_000_000,
    "Partitions.zip": 1_000_000,
}


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env or os.environ.copy())


def _mirror_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_ENDPOINT"] = HF_MIRROR
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    return env


def _direct_xet_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("HF_ENDPOINT", None)
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    return env


def _xet_available() -> bool:
    try:
        subprocess.run(
            ["/root/miniconda3/bin/python", "-c", "import hf_xet"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        return bool(shutil.which("hf") or shutil.which("huggingface-cli"))
    except (subprocess.CalledProcessError, OSError):
        return False


def ensure_hfd_script() -> Path:
    if HFD_SCRIPT.exists() and HFD_SCRIPT.stat().st_size > 1000:
        return HFD_SCRIPT
    print(f"  Installing hfd.sh -> {HFD_SCRIPT}")
    subprocess.run(
        ["wget", "-q", "-O", str(HFD_SCRIPT), f"{HF_MIRROR}/hfd/hfd.sh"],
        check=True,
    )
    HFD_SCRIPT.chmod(0o755)
    return HFD_SCRIPT


def _file_ready(path: Path) -> bool:
    if not path.exists():
        return False
    min_b = MIN_BYTES.get(path.name, 1_000_000)
    return path.stat().st_size >= min_b


def _find_downloaded(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            return direct
        for hit in root.rglob(name):
            if hit.is_file():
                return hit
    return None


def _install_to_dest(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    if dest.exists():
        dest.unlink()
    shutil.copy2(src, dest)
    print(f"    installed {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def download_via_hf_cli(filename: str, staging: Path, *, use_mirror: bool) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf_bin:
        raise RuntimeError("hf CLI not found on PATH")
    env = _mirror_env() if use_mirror else _direct_xet_env()
    label = "mirror" if use_mirror else "direct+xet"
    print(f"  hf download ({label}): {filename}")
    _run(
        [hf_bin, "download", HF_REPO, filename, "--repo-type", HF_REPO_TYPE, "--local-dir", str(staging)],
        env=env,
    )
    found = _find_downloaded(filename, [staging])
    if not found:
        raise FileNotFoundError(f"{filename} not found under {staging}")
    return found


def download_via_hfd(staging: Path) -> Path:
    hfd = ensure_hfd_script()
    repo_dir = staging / HF_REPO.split("/")[-1]
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    print(f"  hfd.sh + {HF_MIRROR}")
    _run(
        [str(hfd), HF_REPO, "--local-dir", str(staging), "--tool", "wget"],
        env=_mirror_env(),
    )
    if not repo_dir.exists():
        # hfd may flatten to staging directly
        repo_dir = staging
    return repo_dir


def download_via_hub(filename: str, staging: Path) -> Path:
    from huggingface_hub import hf_hub_download

    staging.mkdir(parents=True, exist_ok=True)
    old_endpoint = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = HF_MIRROR
    try:
        cached = hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            repo_type=HF_REPO_TYPE,
            local_dir=str(staging),
        )
    finally:
        if old_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = old_endpoint
    return Path(cached)


def ensure_file(filename: str, dest_dir: Path, *, high_performance: bool = True) -> Path:
    dest = dest_dir / filename
    if _file_ready(dest):
        print(f"  skip {filename}: already at {dest}")
        return dest

    staging = dest_dir / "_staging"
    errors: list[str] = []

    if high_performance and _xet_available():
        try:
            src = download_via_hf_cli(filename, staging / "hf_xet", use_mirror=False)
            _install_to_dest(src, dest)
            return dest
        except Exception as exc:
            errors.append(f"hf+xet: {exc}")

    try:
        repo_dir = download_via_hfd(staging / "hfd")
        src = _find_downloaded(filename, [repo_dir, staging / "hfd"])
        if not src:
            raise FileNotFoundError(f"{filename} not found after hfd download")
        _install_to_dest(src, dest)
        return dest
    except Exception as exc:
        errors.append(f"hfd+mirror: {exc}")

    try:
        src = download_via_hf_cli(filename, staging / "hf_mirror", use_mirror=True)
        _install_to_dest(src, dest)
        return dest
    except Exception as exc:
        errors.append(f"hf+mirror: {exc}")

    try:
        src = download_via_hub(filename, staging / "hub_mirror")
        _install_to_dest(src, dest)
        return dest
    except Exception as exc:
        errors.append(f"hub+mirror: {exc}")

    raise RuntimeError(f"All download methods failed for {filename}:\n  " + "\n  ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SUN397 for TFDS manual_dir")
    parser.add_argument("--dest", type=Path, default=DEFAULT_MANUAL_DIR)
    parser.add_argument(
        "--no-xet-high-perf",
        action="store_true",
        help="Skip HF_XET_HIGH_PERFORMANCE direct download",
    )
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    high_perf = not args.no_xet_high_perf
    if high_perf and _xet_available():
        print("Using HF_XET_HIGH_PERFORMANCE=1 + hfd.sh fallback from", HF_REPO)
    else:
        print("Downloading from", HF_REPO, "(hfd.sh + hf-mirror)")

    for filename in REQUIRED_FILES:
        ensure_file(filename, args.dest, high_performance=high_perf)

    # Promote any complete staging files (e.g. if hfd wrote directly under _staging/hfd).
    try:
        from sun397_manual import promote_staging_files

        promote_staging_files(args.dest)
    except ImportError:
        pass

    tar = args.dest / "SUN397.tar.gz"
    parts = args.dest / "Partitions.zip"
    if not _file_ready(tar):
        print(f"ERROR: missing or incomplete {tar}")
        sys.exit(1)
    if not _file_ready(parts):
        print(f"WARN: missing {parts} — TFDS may fail on partition splits")

    print(f"\nManual files ready under {args.dest}")
    print("Run precache with sun397_32x32 after files are in place.")


if __name__ == "__main__":
    main()
