#!/usr/bin/env python3
"""Pack completed TaskSet jobs, verify archives, or recover to a new directory."""

import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mindware.domains.taskset_archive import pack_job, restore_archive, verify_archive
from run_taskset_cpu_campaign import inspect_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "verify", "restore"))
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--job", help="Omit to process all completed jobs.")
    parser.add_argument("--cleanup", action="store_true", help="Remove originals only after full verification under the job lock.")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        manifest = verify_archive(args.archive)
        print(json.dumps({"verified": True, "job": manifest["job"], "entries": len(manifest["entries"])}))
        return
    if args.action == "restore":
        print(restore_archive(args.archive, args.destination))
        return
    from mindware.domains.taskset_archive import archive_path, job_lock
    from run_taskset_cpu_campaign import history_path

    stop_requested = False

    def stop_after_job(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop_after_job)
    signal.signal(signal.SIGINT, stop_after_job)

    plan = json.loads((args.campaign / "distributed/plan.json").read_text())
    jobs = [job for job in plan["jobs"] if args.job is None or job["id"] == args.job]
    if not jobs:
        parser.error("Unknown job")
    failed = 0
    for job in jobs:
        if stop_requested:
            print(json.dumps({"status": "stopped_after_job"}), flush=True)
            break
        try:
            with job_lock(args.campaign, job["id"]) as lock:
                if inspect_history(job, validate=True)["n_observations"] != 100:
                    print(json.dumps({"job": job["id"], "status": "incomplete"}), flush=True)
                    continue
                result_path = args.campaign / "results" / (job["id"] + ".json")
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                if (result.get("archive", {}).get("cleanup_complete")
                        and archive_path(history_path(job)).exists()):
                    print(json.dumps({"job": job["id"], "status": "already_archived"}), flush=True)
                    continue
                report = pack_job(args.campaign, job, cleanup=args.cleanup, lock=lock)
            print(json.dumps({"job": job["id"], "status": "archived", **report}), flush=True)
        except BlockingIOError:
            print(json.dumps({"job": job["id"], "status": "locked"}), flush=True)
        except Exception as exc:
            failed += 1
            print(json.dumps({"job": job["id"], "status": "error", "error": str(exc)}), flush=True)
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
