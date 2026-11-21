"""CPU parallelism helpers for TFDS precache (Beam + TensorFlow)."""

from __future__ import annotations

import multiprocessing
import os
from contextlib import contextmanager

_BEAM_WORKERS: int | None = None
_TF_THREADS: int | None = None


def logical_cpu_count() -> int:
    return multiprocessing.cpu_count() or os.cpu_count() or 8


def default_beam_workers() -> int:
    """Beam DirectRunner workers — ~25% of logical CPUs, capped."""
    cpus = logical_cpu_count()
    return max(8, min(64, cpus // 4))


def default_tf_threads() -> int:
    """TensorFlow / OpenMP threads for image-heavy builds."""
    cpus = logical_cpu_count()
    return max(8, min(32, cpus // 8))


def configure_parallelism(
    *,
    beam_workers: int | None = None,
    tf_threads: int | None = None,
) -> tuple[int, int]:
    """Store global parallelism settings (0 or None = auto)."""
    global _BEAM_WORKERS, _TF_THREADS
    if beam_workers and beam_workers > 0:
        _BEAM_WORKERS = beam_workers
    else:
        _BEAM_WORKERS = default_beam_workers()
    if tf_threads and tf_threads > 0:
        _TF_THREADS = tf_threads
    else:
        _TF_THREADS = default_tf_threads()
    return _BEAM_WORKERS, _TF_THREADS


def beam_worker_count() -> int:
    return _BEAM_WORKERS or default_beam_workers()


def tf_thread_count() -> int:
    return _TF_THREADS or default_tf_threads()


def apply_tf_thread_env(threads: int | None = None) -> int:
    """Set thread env vars consumed by TF, OpenCV, and MKL."""
    count = threads or tf_thread_count()
    os.environ["OMP_NUM_THREADS"] = str(count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(count)
    os.environ["MKL_NUM_THREADS"] = str(count)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(count)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(max(2, count // 4))
    return count


@contextmanager
def tf_parallel_context(threads: int | None = None):
    """Raise TF/OpenCV thread pools for CPU-heavy dataset builds."""
    count = apply_tf_thread_env(threads)
    print(f"  TF/OpenCV thread pools: {count} (cpus={logical_cpu_count()})")
    yield count


def wikipedia_download_config(tfds):
    """DownloadConfig for TFDS wikipedia BeamBasedBuilder parents."""
    from apache_beam.options.pipeline_options import PipelineOptions
    from apache_beam.runners import DirectRunner

    workers = beam_worker_count()
    print(f"  Wikipedia Beam DirectRunner: {workers} workers (multi_processing)")
    beam_options = PipelineOptions(
        direct_num_workers=workers,
        direct_running_mode="multi_processing",
    )
    _patch_gcs_download_skip_diffs()
    return tfds.download.DownloadConfig(
        beam_runner=DirectRunner(),
        beam_options=beam_options,
    )


_GCS_PATCHED = False


def _patch_gcs_download_skip_diffs() -> None:
    """Skip TFDS GCS `diffs/` artifacts that overwrite real files by basename."""
    global _GCS_PATCHED
    if _GCS_PATCHED:
        return
    from tensorflow_datasets.core.utils import gcs_utils

    if getattr(gcs_utils.download_gcs_dataset, "_skip_diffs_patch", False):
        _GCS_PATCHED = True
        return

    import concurrent.futures
    import posixpath

    orig_download = gcs_utils.download_gcs_dataset

    def download_gcs_dataset(dataset_name, local_dataset_dir, max_simultaneous_downloads=50):
        gcs_paths = gcs_utils.gcs_files(
            posixpath.join(gcs_utils.GCS_DATASETS_DIR, dataset_name)
        )
        gcs_paths = [p for p in gcs_paths if "/diffs/" not in p]
        with gcs_utils.utils.async_tqdm(
            total=len(gcs_paths), desc="Dl Completed...", unit=" file"
        ) as pbar:
            def _copy_from_gcs(gcs_path):
                local_path = posixpath.join(
                    local_dataset_dir, posixpath.basename(gcs_path)
                )
                gcs_utils.download_gcs_file(gcs_path, local_path)
                pbar.update(1)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_simultaneous_downloads
            ) as executor:
                futures = [executor.submit(_copy_from_gcs, path) for path in gcs_paths]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

    download_gcs_dataset._skip_diffs_patch = True  # type: ignore[attr-defined]
    gcs_utils.download_gcs_dataset = download_gcs_dataset
    _GCS_PATCHED = True
