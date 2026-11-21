#!/usr/bin/env python3
"""Download Amazon US Reviews PDS TSV files from HuggingFace for TFDS precache.

Fast path (direct huggingface.co + hf-xet):
  HF_XET_HIGH_PERFORMANCE=1 — replaces deprecated HF_HUB_ENABLE_HF_TRANSFER (no-op).

Fallback order when huggingface.co is unreachable:
  1. huggingface_hub with HF_ENDPOINT=https://hf-mirror.com
  2. hfd.sh (HF mirror + wget/aria2) full or partial repo download
  3. hf download CLI with HF_ENDPOINT mirror
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HF_REPO = "shansoround/amazon_reviews"
HF_MIRROR = "https://hf-mirror.com"
DEFAULT_MANUAL_DIR = Path("/root/data/cys/tensorflow_datasets/downloads/manual/amazon_us_reviews")
HFD_SCRIPT = Path(__file__).resolve().parent / "hfd.sh"

REQUIRED_CONFIGS = [
    "Books_v1_02",
    "Camera_v1_00",
    "Home_v1_00",
    "Video_v1_00",
]


def tfds_tsv_name(config: str) -> str:
    return f"amazon_reviews_us_{config}.tsv.gz"


def _config_pattern(config: str) -> re.Pattern[str]:
    return re.compile(re.escape(config), re.IGNORECASE)


def find_repo_file_for_config(repo_files: list[str], config: str) -> str | None:
    pat = _config_pattern(config)
    candidates = []
    for path in repo_files:
        name = Path(path).name
        if not pat.search(name):
            continue
        if name.endswith((".tsv.gz", ".tsv", ".tsv.zip", ".csv.gz", ".csv", ".parquet")):
            candidates.append(path)
    if not candidates:
        for path in repo_files:
            if pat.search(Path(path).name):
                candidates.append(path)
    if not candidates:
        return None

    def score(p: str) -> tuple[int, int, str]:
        name = Path(p).name.lower()
        s = 0
        if name.endswith(".tsv.gz"):
            s += 100
        elif name.endswith(".tsv.zip"):
            s += 95
        elif name.endswith(".tsv"):
            s += 80
        if "amazon_reviews_us" in name:
            s += 50
        if name == tfds_tsv_name(config).lower():
            s += 200
        return (-s, len(p), p)

    return sorted(candidates, key=score)[0]


def _mirror_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_ENDPOINT"] = HF_MIRROR
    env.pop("HF_XET_HIGH_PERFORMANCE", None)
    return env


def _direct_hub_env(*, high_performance: bool = True) -> dict[str, str]:
    """Env for direct huggingface.co downloads via hf-xet."""
    env = os.environ.copy()
    env.pop("HF_ENDPOINT", None)
    if high_performance:
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
    else:
        env.pop("HF_XET_HIGH_PERFORMANCE", None)
    return env


@contextlib.contextmanager
def _temporary_environ(updates: dict[str, str | None]):
    """Apply env overrides for the current process, then restore."""
    saved: dict[str, str | None] = {}
    for key, value in updates.items():
        saved[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _xet_available() -> bool:
    try:
        import hf_xet  # noqa: F401
        return True
    except ImportError:
        pass
    for py in ("/root/miniconda3/bin/python",):
        try:
            subprocess.run(
                [py, "-c", "import hf_xet"],
                check=True,
                capture_output=True,
                timeout=10,
            )
            return bool(shutil.which("hf") or shutil.which("huggingface-cli"))
        except (subprocess.CalledProcessError, OSError):
            continue
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


def list_hf_repo_files(repo: str = HF_REPO, use_mirror: bool = True) -> list[str]:
    from huggingface_hub import HfApi

    endpoint = HF_MIRROR if use_mirror else "https://huggingface.co"
    api = HfApi(endpoint=endpoint)
    return api.list_repo_files(repo_id=repo, repo_type="dataset")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env or os.environ.copy())


def download_via_hub(
    repo: str,
    filename: str,
    dest_dir: Path,
    *,
    use_mirror: bool,
    high_performance: bool = True,
) -> Path:
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    if use_mirror:
        env_updates = {"HF_ENDPOINT": HF_MIRROR, "HF_XET_HIGH_PERFORMANCE": None}
    else:
        env_updates = {
            "HF_ENDPOINT": None,
            "HF_XET_HIGH_PERFORMANCE": "1" if high_performance and _xet_available() else None,
        }
    with _temporary_environ(env_updates):
        cached = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest_dir),
        )
    return Path(cached)


def download_via_hfd(repo: str, dest_dir: Path, include_pattern: str | None = None) -> Path:
    hfd = ensure_hfd_script()
    repo_dir = dest_dir / repo.split("/")[-1]
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    cmd = [
        str(hfd),
        repo,
        "--dataset",
        "--local-dir",
        str(dest_dir),
        "--tool",
        "wget",
    ]
    if include_pattern:
        cmd.extend(["--include", include_pattern])
    _run(cmd, env=_mirror_env())
    return repo_dir


def download_via_hf_cli(
    repo: str,
    filename: str,
    dest_dir: Path,
    *,
    use_mirror: bool,
    high_performance: bool = True,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / Path(filename).name
    hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
    if not hf_bin:
        raise RuntimeError("hf CLI not found on PATH")
    env = _mirror_env() if use_mirror else _direct_hub_env(high_performance=high_performance)
    _run(
        [
            hf_bin,
            "download",
            repo,
            filename,
            "--repo-type",
            "dataset",
            "--local-dir",
            str(dest_dir),
        ],
        env=env,
    )
    candidate = dest_dir / filename
    return candidate if candidate.exists() else out


def _find_local_match(search_roots: list[Path], config: str) -> Path | None:
    pat = _config_pattern(config)
    hits: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if pat.search(path.name) and path.suffix in (".gz", ".tsv", ".csv", ".parquet", ""):
                hits.append(path)
    if not hits:
        return None

    def score(p: Path) -> tuple[int, int]:
        name = p.name.lower()
        s = 0
        if name.endswith(".tsv.gz"):
            s += 100
        if "amazon_reviews_us" in name:
            s += 50
        return (-s, len(str(p)))

    return sorted(hits, key=score)[0]


def download_hf_file(
    repo: str,
    filename: str,
    dest_dir: Path,
    *,
    high_performance: bool = True,
) -> Path:
    """Try direct hub+xet first, then mirror / hfd fallbacks."""
    errors: list[str] = []
    xet_note = " (HF_XET_HIGH_PERFORMANCE=1)" if high_performance and _xet_available() else ""

    for label, fn in [
        (f"huggingface_hub+direct+xet{xet_note}", lambda: download_via_hub(
            repo, filename, dest_dir, use_mirror=False, high_performance=high_performance)),
        (f"hf-cli+direct+xet{xet_note}", lambda: download_via_hf_cli(
            repo, filename, dest_dir / "_hf_cli_direct", use_mirror=False,
            high_performance=high_performance)),
        ("huggingface_hub+hf-mirror", lambda: download_via_hub(
            repo, filename, dest_dir, use_mirror=True, high_performance=False)),
    ]:
        try:
            return fn()
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    include = Path(filename).name
    config_key = re.sub(r"\.(tsv\.gz|tsv\.zip|tsv)$", "", filename.replace("amazon_reviews_us_", ""))
    try:
        repo_dir = download_via_hfd(repo, dest_dir / "_hfd", include_pattern=f"*{config_key}*")
        match = _find_local_match([repo_dir], config_key)
        if match:
            return match
        errors.append(f"hfd partial: no match for {config_key} under {repo_dir}")
    except Exception as exc:
        errors.append(f"hfd partial: {exc}")

    try:
        return download_via_hf_cli(
            repo, filename, dest_dir / "_hf_cli", use_mirror=True, high_performance=False)
    except Exception as exc:
        errors.append(f"hf download+mirror: {exc}")

    try:
        repo_dir = download_via_hfd(repo, dest_dir / "_hfd_full")
        config_guess = re.sub(r"\.(tsv\.gz|tsv\.zip|tsv)$", "", filename.replace("amazon_reviews_us_", ""))
        match = _find_local_match([repo_dir], config_guess)
        if match:
            return match
        errors.append(f"hfd full: no file matching {config_guess} under {repo_dir}")
    except Exception as exc:
        errors.append(f"hfd full: {exc}")

    raise RuntimeError("All HF download methods failed:\n  " + "\n  ".join(errors))


def _convert_to_tsv_gz(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(src.suffixes).lower()
    if suffix == ".tsv.gz" or src.name == dest.name:
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return
    if suffix == ".tsv.zip":
        with zipfile.ZipFile(src) as zf:
            members = [m for m in zf.namelist() if m.endswith((".tsv", ".csv"))]
            if not members:
                raise RuntimeError(f"No .tsv inside zip: {src}")
            with zf.open(members[0]) as fin, gzip.open(dest, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        return
    if suffix in (".tsv", ".csv"):
        with src.open("rb") as fin, gzip.open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return
    if suffix == ".csv.gz":
        with gzip.open(src, "rb") as fin, gzip.open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        return
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                f"Need pandas to convert {src} parquet -> tsv.gz: pip install pandas pyarrow"
            ) from exc
        df = pd.read_parquet(src)
        df.to_csv(dest, sep="\t", index=False, compression="gzip")
        return
    raise RuntimeError(f"Unsupported Amazon reviews file type: {src}")


def ensure_amazon_tsv(
    config: str,
    manual_dir: Path = DEFAULT_MANUAL_DIR,
    repo: str = HF_REPO,
    *,
    high_performance: bool = True,
) -> Path:
    dest = manual_dir / tfds_tsv_name(config)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest

    xet = high_performance and _xet_available()
    print(
        f"  Fetching {config} from datasets/{repo} "
        f"(xet_high_perf={'on' if xet else 'off'}, mirror fallback: {HF_MIRROR}) ..."
    )
    repo_files = list_hf_repo_files(repo, use_mirror=True)
    match = find_repo_file_for_config(repo_files, config)
    if not match:
        # Still try hfd full download — repo listing may fail but files exist
        print(f"  WARN: could not map {config} in repo listing; trying hfd full download")
        staging = manual_dir / "_staging"
        repo_dir = download_via_hfd(repo, staging)
        local = _find_local_match([repo_dir], config)
        if not local:
            raise FileNotFoundError(
                f"No file for {config} in {repo}. Repo files: {repo_files[:20]}"
            )
        _convert_to_tsv_gz(local, dest)
        print(f"    wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"    matched repo file: {match}")
    staging = manual_dir / "_staging"
    downloaded = download_hf_file(repo, match, staging, high_performance=high_performance)
    _convert_to_tsv_gz(downloaded, dest)
    print(f"    wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def ensure_all_required(
    manual_dir: Path = DEFAULT_MANUAL_DIR,
    repo: str = HF_REPO,
    *,
    high_performance: bool = True,
) -> dict[str, Path]:
    return {
        config: ensure_amazon_tsv(
            config, manual_dir=manual_dir, repo=repo, high_performance=high_performance)
        for config in REQUIRED_CONFIGS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Amazon US Reviews from HuggingFace")
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--dest", type=Path, default=DEFAULT_MANUAL_DIR)
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--list-files", action="store_true")
    parser.add_argument(
        "--no-xet-high-perf",
        action="store_true",
        help="Disable HF_XET_HIGH_PERFORMANCE (use if low RAM; needs ~64GB for best results)",
    )
    args = parser.parse_args()
    high_performance = not args.no_xet_high_perf
    if high_performance and _xet_available():
        print("Using HF_XET_HIGH_PERFORMANCE=1 for direct huggingface.co downloads (hf-xet)")
    elif high_performance:
        print("WARN: hf-xet not installed; install with: pip install -U huggingface_hub hf-xet")

    if args.list_files:
        for use_mirror in (True, False):
            label = "hf-mirror" if use_mirror else "direct"
            try:
                files = list_hf_repo_files(args.repo, use_mirror=use_mirror)
                print(f"\n=== {label} ({len(files)} files) ===")
                for f in files:
                    print(f)
                for config in REQUIRED_CONFIGS:
                    match = find_repo_file_for_config(files, config)
                    print(f"  {config}: {match or 'NOT FOUND'}")
            except Exception as exc:
                print(f"\n=== {label}: FAILED ({exc}) ===")
        return

    configs = args.config or REQUIRED_CONFIGS
    for config in configs:
        ensure_amazon_tsv(
            config, manual_dir=args.dest, repo=args.repo, high_performance=high_performance)
    print(f"\nReady under {args.dest}")


if __name__ == "__main__":
    main()
