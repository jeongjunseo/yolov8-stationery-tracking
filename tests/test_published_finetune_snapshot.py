"""Exercise preservation against real, isolated local Git fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import preserve_published_finetune as snapshot


class PublishedSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.checkout = self.workspace / "new_run_20261120_example"
        self.checkout.mkdir()
        self.git("init", "--quiet")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.symlinks", "true")
        self.files = {
            ".gitignore": b"runs/\n",
            "models/best.pt": b"baseline model bytes",
            "outputs/test_metrics.json": b'{"baseline":true}\n',
            "scripts/train.py": b"# baseline training entry point\n",
            "data/stationery.yaml": b"train: train/images\n",
            "models/video_finetune/best.pt": b"previous published adaptation",
            "outputs/video_finetune/training_run.json": b'{"epochs_completed":100}\n',
            "outputs/video_finetune/test_metrics.json": b'{"old_metrics":true}\n',
            "report/assets/video_finetune/train/results.csv": b"epoch,value\n1,0.8\n",
            "report/assets/video_finetune/test/confusion_matrix.png": b"old test image",
            "report/assets/video_finetune/tracking/preview.jpg": b"old tracking preview",
            snapshot.REUSABLE_INPUT: b"shared reviewed annotation preview",
        }
        for relative, content in self.files.items():
            destination = self.checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        self.commit()
        self.destination = self.checkout.with_name(self.checkout.name + "_published_snapshot")

    def git(self, *arguments):
        return subprocess.check_output(["git", *arguments], cwd=self.checkout, text=True,
                                       stderr=subprocess.STDOUT).strip()

    def commit(self):
        self.git("add", "--all")
        self.git("-c", "user.name=Test Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "--quiet", "-m", "isolated fixture")

    def test_preserves_all_published_trees_and_only_restores_shared_annotation(self):
        before = snapshot.inventory(self.checkout)
        commit = self.git("rev-parse", "HEAD")
        record = snapshot.preserve_published_artifacts(self.checkout)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["source_commit"], commit)
        self.assertEqual(record["files"], before)
        self.assertEqual(snapshot.inventory(self.destination), before)
        self.assertFalse((self.checkout / "models/video_finetune").exists())
        self.assertFalse((self.checkout / "outputs/video_finetune").exists())
        self.assertFalse((self.checkout / "report/assets/video_finetune/train").exists())
        self.assertFalse((self.checkout / "report/assets/video_finetune/test").exists())
        self.assertFalse((self.checkout / "report/assets/video_finetune/tracking").exists())
        for relative in snapshot.BASELINE_FILES:
            self.assertEqual((self.checkout / relative).read_bytes(), self.files[relative])
        self.assertEqual((self.checkout / snapshot.REUSABLE_INPUT).read_bytes(),
                         (self.destination / snapshot.REUSABLE_INPUT).read_bytes())
        stored = json.loads((self.destination / "snapshot_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["baseline_sha256_before"], stored["baseline_sha256_after"])
        self.assertEqual(stored["snapshot_files_verified"], len(before))

    def test_existing_snapshot_is_never_overwritten_or_merged(self):
        before = snapshot.inventory(self.checkout)
        self.destination.mkdir()
        sentinel = self.destination / "existing.txt"
        sentinel.write_bytes(b"keep existing snapshot")
        with self.assertRaises(FileExistsError):
            snapshot.preserve_published_artifacts(self.checkout)
        self.assertEqual(snapshot.inventory(self.checkout), before)
        self.assertEqual(sentinel.read_bytes(), b"keep existing snapshot")

    def test_cli_operates_on_a_new_local_clone_and_leaves_source_repository_unchanged(self):
        original = snapshot.inventory(self.checkout)
        cloned = self.workspace / "another_new_run"
        subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(self.checkout), str(cloned)],
                       check=True, capture_output=True, text=True)
        # A clone may apply the machine's Git newline policy. Preservation
        # compares that checkout's real bytes before/after, not another tree's.
        cloned_before = snapshot.inventory(cloned)
        process = subprocess.run([sys.executable, str(Path(snapshot.__file__).resolve()), str(cloned)],
                                 cwd=cloned, check=True, capture_output=True, text=True, encoding="utf-8")
        record = json.loads(process.stdout)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(Path(record["source_checkout"]), cloned)
        self.assertEqual(Path(record["snapshot_directory"]), self.workspace / "another_new_run_published_snapshot")
        self.assertEqual(record["files"], cloned_before)
        self.assertEqual(snapshot.inventory(self.checkout), original)
        self.assertEqual(self.git("status", "--porcelain", "--ignored=matching"), "")

    def test_ignored_runtime_files_block_the_operation(self):
        runtime = self.checkout / "runs/active/session.txt"
        runtime.parent.mkdir(parents=True)
        runtime.write_bytes(b"active run state")
        with self.assertRaisesRegex(RuntimeError, "fresh clone"):
            snapshot.preserve_published_artifacts(self.checkout)
        self.assertFalse(self.destination.exists())
        self.assertEqual(runtime.read_bytes(), b"active run state")

    def test_dirty_tracked_baseline_blocks_the_operation_without_ignored_files(self):
        (self.checkout / "models/best.pt").write_bytes(b"user baseline edit")
        with self.assertRaisesRegex(RuntimeError, "fresh clone"):
            snapshot.preserve_published_artifacts(self.checkout)
        self.assertFalse(self.destination.exists())
        self.assertEqual((self.checkout / "models/best.pt").read_bytes(), b"user baseline edit")
        self.assertTrue((self.checkout / "models/video_finetune/best.pt").is_file())

    def test_nested_checkout_and_path_escape_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkout root"):
            snapshot.preserve_published_artifacts(self.checkout / "outputs")
        with self.assertRaisesRegex(ValueError, "escapes"):
            snapshot.checked_path(self.checkout, "../outside")
        with self.assertRaisesRegex(ValueError, "escapes"):
            snapshot.checked_path(self.checkout, str(self.workspace / "outside"))
        self.assertFalse(self.destination.exists())

    def test_tracked_symlink_is_rejected_without_moving_or_touching_target(self):
        outside = self.workspace / "outside.txt"
        outside.write_bytes(b"external file unchanged")
        linked = self.checkout / "outputs/video_finetune/external.txt"
        try:
            linked.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"OS does not permit fixture symlinks: {error}")
        self.commit()
        with self.assertRaisesRegex(ValueError, "Linked artifact"):
            snapshot.preserve_published_artifacts(self.checkout)
        self.assertFalse(self.destination.exists())
        self.assertEqual(outside.read_bytes(), b"external file unchanged")
        self.assertTrue((self.checkout / "models/video_finetune/best.pt").exists())

    def test_changed_bytes_during_move_fail_verification_and_remain_preserved(self):
        original_rename = Path.rename
        injected = False

        def changing_rename(source, target):
            nonlocal injected
            if not injected and source == self.checkout / "outputs/video_finetune":
                (source / "test_metrics.json").write_bytes(b"concurrent change")
                injected = True
            return original_rename(source, target)

        with patch.object(Path, "rename", changing_rename):
            with self.assertRaisesRegex(RuntimeError, "inventory/hash mismatch"):
                snapshot.preserve_published_artifacts(self.checkout)
        self.assertTrue((self.destination / "planned_snapshot.json").is_file())
        self.assertFalse((self.destination / "snapshot_manifest.json").exists())
        self.assertEqual((self.destination / "outputs/video_finetune/test_metrics.json").read_bytes(),
                         b"concurrent change")
        self.assertEqual((self.checkout / "models/best.pt").read_bytes(), self.files["models/best.pt"])


if __name__ == "__main__":
    unittest.main()
