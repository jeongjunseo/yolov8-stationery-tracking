"""Archive isolation and original-setting checks; no real model training."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from scripts import finetune_video as ft


class VideoAdditionTests(unittest.TestCase):
    def archive(self, root, *, extra=None, split="train", label=None):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 48), "gray").save(buffer, format="JPEG")
        contents = {
            "data/train/images/video_f000000.jpg": buffer.getvalue(),
            "data/train/labels/video_f000000.txt": (label or
                "0 0.5 0.2 0.4 0.1\n1 0.5 0.8 0.4 0.1\n2 0.5 0.5 0.4 0.1\n").encode(),
        }
        if extra:
            contents.update(extra)
        manifest = {"split": split, "classes": ft.CLASS_NAMES,
                    "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                              for name, data in contents.items()]}
        archive = root / "addition.zip"
        with zipfile.ZipFile(archive, "x") as zipped:
            for name, data in contents.items():
                zipped.writestr(name, data)
            zipped.writestr("video_dataset_manifest.json", json.dumps(manifest))
        return archive, ft.digest_file(archive)

    def test_verified_train_only_archive_and_differing_file_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, sha = self.archive(root)
            target = root / "separate"
            result = ft.extract_addition(archive, target, sha)
            self.assertEqual(result["manifest_files"], 2)
            self.assertEqual(ft.validate_addition(target)["boxes_per_class"],
                             {"highlighter": 1, "correction_tape": 1, "eraser": 1})
            ft.extract_addition(archive, target, sha)  # Identical files may be read again.
            label = target / "data/train/labels/video_f000000.txt"
            label.write_text("preserve this user edit", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ft.extract_addition(archive, target, sha)
            self.assertEqual(label.read_text(), "preserve this user edit")

    def test_traversal_baseline_and_test_members_rejected_before_extraction(self):
        for name in ("../escape.txt", "models/best.pt", "data/test/images/forbidden.jpg"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive, sha = self.archive(root, extra={name: b"unchanged"})
                target = root / "separate"
                with self.assertRaises(ValueError):
                    ft.extract_addition(archive, target, sha)
                self.assertFalse(target.exists())

    def test_checksum_or_nontrain_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, sha = self.archive(root, split="test")
            with self.assertRaisesRegex(ValueError, "train split only"):
                ft.extract_addition(archive, root / "addition", sha)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ft.extract_addition(archive, root / "addition", "0" * 64)

    def test_missing_class_cannot_be_used_as_reviewed_three_object_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, sha = self.archive(root, label="0 0.5 0.5 0.4 0.1\n")
            target = root / "addition"
            ft.extract_addition(archive, target, sha)
            with self.assertRaisesRegex(ValueError, "three reviewed objects"):
                ft.validate_addition(target)

    def test_training_keeps_requested_settings_and_uses_separate_run_path(self):
        options = ft.train_options(Path("mixed.yaml"), 16)
        self.assertEqual({key: options[key] for key in
            ("epochs", "imgsz", "batch", "workers", "seed", "deterministic", "device")},
            {"epochs": 100, "imgsz": 640, "batch": 16, "workers": 4,
             "seed": 42, "deterministic": True, "device": "0"})
        self.assertEqual(Path(options["project"]).name, "train_video_finetune")
        with self.assertRaises(ValueError):
            ft.train_options(Path("mixed.yaml"), 2)


if __name__ == "__main__":
    unittest.main()
