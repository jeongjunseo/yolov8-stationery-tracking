"""Synthetic pipeline regression tests; these do not measure detection accuracy.

Run from the repository root after installing requirements:
    python -m unittest discover -s tests -v

All videos, CSVs, and placeholder weights live in temporary directories. YOLO
inference is mocked; actual Ultralytics Results and OpenCV encode/decode are used.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts import evaluate, track_video  # noqa: E402


NAMES = {0: "highlighter", 1: "correction_tape", 2: "eraser"}
TRACKS = ((101, 0), (202, 1), (303, 2))


def center_for(track_id: int, frame_index: int) -> tuple[int, int]:
    return {
        101: (30 + frame_index, 60),
        202: (265 - frame_index, 120),
        303: (30 + frame_index, 190),
    }[track_id]


class SyntheticDetector:
    """Return known CPU tensors, without loading weights or running detection."""

    names = NAMES

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def track(self, frame, **kwargs):
        frame_index = len(self.calls)
        self.calls.append(kwargs)
        boxes = []
        for track_id, class_id in TRACKS:
            x, y = center_for(track_id, frame_index)
            # Tracked Results format: xyxy, track_id, confidence, class_id.
            boxes.append([x - 10, y - 12, x + 10, y + 12, track_id, 0.9, class_id])
        return [
            Results(
                orig_img=frame,
                path=f"synthetic_frame_{frame_index}.jpg",
                names=self.names,
                boxes=torch.tensor(boxes, dtype=torch.float32),
            )
        ]


class TrackingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="stationery_synthetic_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "placeholder.pt"
        self.source = self.root / "source.mp4"
        self.output = self.root / "result.mp4"
        self.trajectory = self.root / "trajectory.csv"
        self.model.write_bytes(b"Synthetic placeholder; never load as a model.")

    def command(self) -> list[str]:
        return [
            "track_video.py",
            "--model", str(self.model),
            "--source", str(self.source),
            "--output", str(self.output),
            "--trajectory", str(self.trajectory),
            "--device", "0",
        ]

    def make_source(self) -> None:
        writer = cv2.VideoWriter(
            str(self.source), cv2.VideoWriter_fourcc(*"mp4v"), 29.97, (320, 240)
        )
        self.assertTrue(writer.isOpened(), "OpenCV must support MP4 encoding for this test")
        try:
            for index in range(160):
                frame = np.full((240, 320, 3), 35 + index % 10, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def test_fractional_source_fps_produces_five_seconds_and_matching_csv(self) -> None:
        self.make_source()
        source_digest = hashlib.sha256(self.source.read_bytes()).digest()
        model_bytes = self.model.read_bytes()
        source_capture = cv2.VideoCapture(str(self.source))
        try:
            self.assertTrue(source_capture.isOpened())
            source_fps = source_capture.get(cv2.CAP_PROP_FPS)
            self.assertAlmostEqual(source_fps, 29.97, places=2)
            self.assertGreater(source_capture.get(cv2.CAP_PROP_FRAME_COUNT) / source_fps, 5)
        finally:
            source_capture.release()

        detector = SyntheticDetector()
        drawn_segments = set()
        original_line = cv2.line

        def record_line(image, start, end, color, thickness, *args, **kwargs):
            if tuple(color) == (0, 255, 255) and thickness == 3:
                drawn_segments.add((tuple(start), tuple(end)))
            return original_line(image, start, end, color, thickness, *args, **kwargs)

        with (
            patch.object(sys, "argv", self.command()),
            patch.object(track_video, "YOLO", return_value=detector) as model_factory,
            patch.object(track_video.cv2, "line", new=record_line),
            redirect_stdout(io.StringIO()),
        ):
            track_video.main()

        model_factory.assert_called_once_with(str(self.model.resolve()))
        self.assertEqual(len(detector.calls), 150)
        self.assertTrue(all(call["device"] == "0" for call in detector.calls))
        self.assertTrue(all(call["persist"] is True for call in detector.calls))
        self.assertTrue(all(call["tracker"] == "bytetrack.yaml" for call in detector.calls))

        capture = cv2.VideoCapture(str(self.output))
        decoded_frames = 0
        last_frame = None
        try:
            self.assertTrue(capture.isOpened())
            output_fps = capture.get(cv2.CAP_PROP_FPS)
            self.assertAlmostEqual(output_fps, 30.0, places=6)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 150)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                decoded_frames += 1
                last_frame = frame
        finally:
            capture.release()
        self.assertEqual(decoded_frames, 150)
        self.assertAlmostEqual(decoded_frames / output_fps, 5.0, places=6)
        self.assertIsNotNone(last_frame)
        # An encoded yellow trail behind track 303, clear of its final box/label.
        blue, green, red = map(int, last_frame[190, 135])
        self.assertLess(blue, 80)
        self.assertGreater(green, 180)
        self.assertGreater(red, 180)

        with self.trajectory.open(newline="", encoding="utf-8-sig") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 150 * len(TRACKS))
        rows_by_key = {(int(row["frame"]), int(row["track_id"])): row for row in rows}
        self.assertEqual(len(rows_by_key), len(rows))
        for index in range(150):
            for track_id, class_id in TRACKS:
                row = rows_by_key[index, track_id]
                self.assertEqual(int(row["class_id"]), class_id)
                self.assertEqual(row["class_name"], NAMES[class_id])
                self.assertEqual(
                    (int(row["center_x"]), int(row["center_y"])),
                    center_for(track_id, index),
                )
                self.assertAlmostEqual(float(row["confidence"]), 0.9, places=6)
                self.assertAlmostEqual(float(row["time_sec"]), index / 30, delta=0.00005)
        self.assertEqual(float(rows[-1]["time_sec"]), round(149 / 30, 4))
        expected_segments = {
            (center_for(track_id, index), center_for(track_id, index + 1))
            for track_id, _ in TRACKS
            for index in range(149)
        }
        self.assertEqual(drawn_segments, expected_segments)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).digest(), source_digest)
        self.assertEqual(self.model.read_bytes(), model_bytes)

    def assert_existing_output_preserved(self, destination: Path) -> None:
        self.source.write_bytes(b"Synthetic source; must not be opened in this test.")
        destination.write_bytes(b"Existing artifact to preserve.")
        before = {
            path: path.read_bytes() for path in (self.model, self.source, destination)
        }
        with (
            patch.object(sys, "argv", self.command()),
            patch.object(track_video, "YOLO") as model_factory,
            patch.object(track_video.cv2, "VideoCapture") as capture_factory,
        ):
            with self.assertRaises(FileExistsError):
                track_video.main()
        model_factory.assert_not_called()
        capture_factory.assert_not_called()
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        other_output = self.trajectory if destination == self.output else self.output
        self.assertFalse(other_output.exists())

    def test_existing_video_is_preserved(self) -> None:
        self.assert_existing_output_preserved(self.output)

    def test_existing_csv_is_preserved(self) -> None:
        self.assert_existing_output_preserved(self.trajectory)


class PerClassMetricsTests(unittest.TestCase):
    def test_reordered_classes_and_absent_class(self) -> None:
        rows = [(0.91, 0.82, 0.73, 0.64), (0.55, 0.46, 0.37, 0.28)]
        metrics = SimpleNamespace(
            box=SimpleNamespace(ap_class_index=[2, 0], class_result=rows.__getitem__),
            nt_per_class=[12, 0, 27],
        )
        result = evaluate.per_class_metrics(metrics, NAMES)
        self.assertEqual(result["eraser"], {
            "class_id": 2, "precision": 0.91, "recall": 0.82,
            "mAP50": 0.73, "mAP50-95": 0.64, "target_count": 27,
        })
        self.assertEqual(result["highlighter"], {
            "class_id": 0, "precision": 0.55, "recall": 0.46,
            "mAP50": 0.37, "mAP50-95": 0.28, "target_count": 12,
        })
        self.assertEqual(result["correction_tape"], {
            "class_id": 1, "precision": None, "recall": None,
            "mAP50": None, "mAP50-95": None, "target_count": 0,
        })
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertEqual(evaluate.per_class_metrics(metrics, list(NAMES.values())), result)

    def test_unavailable_counts_and_empty_metrics_are_null(self) -> None:
        metrics = SimpleNamespace(box=SimpleNamespace(ap_class_index=[]))
        result = evaluate.per_class_metrics(metrics, NAMES)
        self.assertEqual(set(result), set(NAMES.values()))
        for values in result.values():
            for key in ("precision", "recall", "mAP50", "mAP50-95", "target_count"):
                self.assertIsNone(values[key])


if __name__ == "__main__":
    unittest.main()
