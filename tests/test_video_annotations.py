"""Regression for normalized boxes touching a real video image boundary."""

from pathlib import Path
import tempfile
import unittest

from scripts.check_dataset import validate_label
from scripts.prepare_video_finetune_data import yolo_label


class VideoAnnotationBoundsTests(unittest.TestCase):
    def test_serialized_edge_boxes_pass_the_unchanged_strict_checker(self):
        # Frame 234's right-touching box overflowed after eight-place rounding.
        # Also exercise the top/right corner in frame 216 and the full canvas.
        boxes = {
            "0": [16, 484, 1080, 636],
            "1": [158, 0, 1080, 78],
            "2": [0, 0, 1080, 1920],
        }
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "bounds.txt"
            label.write_bytes(yolo_label(boxes, 1080, 1920))
            classes, errors = validate_label(label)
        self.assertEqual(classes, {0, 1, 2})
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
