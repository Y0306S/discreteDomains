import fcntl
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mindware.domains.history import history_exists, iter_history_paths, load_openbox_history
from mindware.domains.taskset_archive import (
    ArchiveError, archive_path, file_digest, pack_job, restore_archive, verify_archive,
)


class TaskSetArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.campaign = self.root / "campaign"
        self.job = {"id": "test.random.task1", "task_id": "task1", "method": "random",
                    "side": "test", "output_root": str(self.campaign / "test")}
        self.run = self.campaign / "test/task1/random/seed1"
        self.run.mkdir(parents=True)
        self.runtime = self.campaign / "runtimes" / self.job["id"]
        self.runtime.mkdir(parents=True)
        self.shared = self.root / "shared"
        self.shared.mkdir()
        (self.shared / "keep.txt").write_text("external evidence")
        (self.runtime / "shared").symlink_to(self.shared, target_is_directory=True)
        (self.runtime / "raw.txt").write_text("raw evidence")
        for folder in ("results", "job_logs", "locks"):
            (self.campaign / folder).mkdir()
        (self.campaign / "job_logs" / (self.job["id"] + ".log")).write_text("evaluation output")
        self.history = {"task_id": "task1", "meta_info": {"domain": "taskset", "optimizer": "random", "seed": 1},
                        "observations": [{"trial_state": 0, "config": {"x": i}, "objectives": [float(i)]} for i in range(100)]}
        (self.run / "history.json").write_text(json.dumps(self.history))
        (self.run / "history.prev.json").write_text(json.dumps({**self.history, "observations": self.history["observations"][:99]}))
        self.result_path = self.campaign / "results" / (self.job["id"] + ".json")
        self.result = {"job": self.job, "returncode": 0, "status": "complete", "n_observations": 100, "n_successful": 100}
        self.result_path.write_text(json.dumps(self.result))
        self.archive = archive_path(self.run / "history.json")

    def tearDown(self):
        self.temporary.cleanup()

    def test_pack_cleanup_discover_read_and_restore(self):
        lock = self.campaign / "locks" / (self.job["id"] + ".lock")
        lock.touch()
        inode = lock.stat().st_ino
        report = pack_job(self.campaign, self.job, cleanup=True)
        self.assertTrue(report["cleanup_complete"])
        self.assertEqual(report["sha256"], file_digest(self.archive))
        self.assertFalse(self.run.exists())
        self.assertFalse(self.runtime.exists())
        self.assertEqual(lock.stat().st_ino, inode)
        self.assertTrue((self.shared / "keep.txt").exists())
        self.assertTrue(history_exists(self.run / "history.json"))
        self.assertEqual(load_openbox_history(self.run / "history.json"), self.history)
        self.assertEqual(load_openbox_history(self.archive), self.history)
        self.assertEqual(list(iter_history_paths(self.campaign / "test")), [self.run / "history.json"])
        restored = restore_archive(self.archive, self.root / "restored")
        self.assertEqual(json.loads((restored / "run/history.json").read_text()), self.history)
        self.assertEqual((restored / "runtime/raw.txt").read_text(), "raw evidence")
        self.assertTrue((restored / "runtime/shared").is_symlink())
        with self.assertRaises(FileExistsError):
            restore_archive(self.archive, restored)

    def test_resume_after_partial_cleanup(self):
        pack_job(self.campaign, self.job)
        (self.runtime / "raw.txt").unlink()
        (self.run / "history.json").unlink()
        self.assertEqual(load_openbox_history(self.run / "history.json"), self.history)
        pack_job(self.campaign, self.job, cleanup=True)
        self.assertFalse(self.run.exists())
        self.assertEqual(pack_job(self.campaign, self.job, cleanup=True)["removed_entries_this_call"], 0)

    def test_corrupt_archive_stops_cleanup_and_reader(self):
        pack_job(self.campaign, self.job)
        self.archive.write_bytes(self.archive.read_bytes()[:100])
        with self.assertRaises(ArchiveError):
            pack_job(self.campaign, self.job, cleanup=True)
        with self.assertRaises(ArchiveError):
            load_openbox_history(self.run / "history.json")
        self.assertTrue((self.runtime / "raw.txt").exists())

    def test_new_source_prevents_any_cleanup(self):
        pack_job(self.campaign, self.job)
        (self.runtime / "new.txt").write_text("new evidence")
        with self.assertRaises(ArchiveError):
            pack_job(self.campaign, self.job, cleanup=True)
        self.assertTrue((self.run / "history.json").exists())

    def test_modified_source_prevents_cleanup(self):
        pack_job(self.campaign, self.job)
        (self.runtime / "raw.txt").write_text("changed")
        with self.assertRaises(ArchiveError):
            pack_job(self.campaign, self.job, cleanup=True)
        self.assertTrue((self.run / "history.json").exists())

    def test_incomplete_run_is_preserved(self):
        self.result_path.write_text(json.dumps({**self.result, "status": "incomplete"}))
        with self.assertRaises(ArchiveError):
            pack_job(self.campaign, self.job, cleanup=True)
        self.assertFalse(self.archive.exists())
        self.assertTrue(self.run.exists())

    def test_legacy_completion_without_status_is_supported(self):
        self.result.pop("status")
        self.result_path.write_text(json.dumps(self.result))
        pack_job(self.campaign, self.job, cleanup=True)
        self.assertEqual(load_openbox_history(self.archive), self.history)

    def test_lock_excludes_competing_packer(self):
        with (self.campaign / "locks" / (self.job["id"] + ".lock")).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                pack_job(self.campaign, self.job, cleanup=True)
            pack_job(self.campaign, self.job, cleanup=True, lock=lock)

    def test_failed_evaluations_are_preserved(self):
        self.history["observations"][1]["trial_state"] = 1
        (self.run / "history.json").write_text(json.dumps(self.history))
        self.result_path.write_text(json.dumps({**self.result, "status": "complete_with_failures", "n_successful": 99}))
        pack_job(self.campaign, self.job, cleanup=True)
        self.assertEqual(load_openbox_history(self.archive), self.history)

    def test_published_index_checksum_is_enforced(self):
        pack_job(self.campaign, self.job)
        result = json.loads(self.result_path.read_text())
        result["archive"]["sha256"] = "wrong"
        self.result_path.write_text(json.dumps(result))
        with self.assertRaises(ArchiveError):
            pack_job(self.campaign, self.job, cleanup=True)
        self.assertTrue(self.run.exists())

    def test_previous_checkpoint_still_recovers_active_runs(self):
        (self.run / "history.json").write_text("broken JSON")
        self.assertEqual(len(load_openbox_history(self.run / "history.json")["observations"]), 99)

    def test_concurrent_publication_during_history_read(self):
        pack_job(self.campaign, self.job)
        packed = self.archive.read_bytes()
        self.archive.unlink()
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if path == self.run / "history.json":
                self.archive.write_bytes(packed)
                path.unlink()
                (self.run / "history.prev.json").unlink()
                raise FileNotFoundError(path)
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            self.assertEqual(load_openbox_history(self.run / "history.json"), self.history)

    def test_unpublished_partial_is_rebuilt(self):
        self.archive.with_name(self.archive.name + ".partial").write_text("incomplete")
        pack_job(self.campaign, self.job, cleanup=True)
        self.assertEqual(verify_archive(self.archive)["observations"], 100)

    def test_symlink_cleanup_root_is_refused(self):
        self.runtime.rename(self.root / "moved-runtime")
        self.runtime.symlink_to(self.root / "moved-runtime")
        with self.assertRaises(ArchiveError):
            pack_job(self.campaign, self.job, cleanup=True)
        self.assertTrue((self.root / "moved-runtime/raw.txt").exists())


if __name__ == "__main__":
    unittest.main()
