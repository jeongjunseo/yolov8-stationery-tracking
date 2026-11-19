"""Inference-rotation geometry and CSV integration, with no real model inference."""
from __future__ import annotations

import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.engine.results import Results

from scripts import track_video

NAMES = {0: "highlighter", 1: "correction_tape", 2: "eraser"}


class OrientationGeometryTests(unittest.TestCase):
    def test_all_rotations_restore_source_box_and_preserve_tracking_columns(self):
        frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
        source_bytes = frame.tobytes()
        rotated_boxes = {
            0: [20, 10, 70, 40], 90: [60, 20, 90, 70],
            180: [130, 60, 180, 90], 270: [10, 130, 40, 180],
        }
        for angle, coordinates in rotated_boxes.items():
            with self.subTest(angle=angle):
                inference_frame = track_video.rotate_inference_frame(frame, angle)
                expected_shape = (200, 100, 3) if angle in (90, 270) else frame.shape
                self.assertEqual(inference_frame.shape, expected_shape)
                # Follow a distinctive pixel, not only the array dimensions.
                location = {0: (7, 13), 90: (13, 92), 180: (92, 186), 270: (186, 7)}[angle]
                np.testing.assert_array_equal(inference_frame[location], frame[7, 13])
                data = torch.tensor([coordinates + [17, 0.73, 2], coordinates + [28, 0.91, 0]])
                original_data = data.clone()
                result = Results(orig_img=inference_frame, path="source.jpg", names=NAMES, boxes=data)
                restored = track_video.restore_result_orientation(result, frame, angle)
                torch.testing.assert_close(restored.boxes.xyxy, torch.tensor([[20., 10., 70., 40.]] * 2))
                torch.testing.assert_close(restored.boxes.data[:, 4:], original_data[:, 4:])
                torch.testing.assert_close(result.boxes.data, original_data)
                self.assertNotEqual(restored.boxes.data.data_ptr(), result.boxes.data.data_ptr())
                self.assertEqual(restored.orig_shape, (100, 200))
                self.assertEqual(restored.names, NAMES)
                self.assertEqual(restored.path, "source.jpg")
                self.assertTrue(restored.boxes.is_track)
                self.assertEqual(frame.tobytes(), source_bytes)

    def test_clipping_untracked_empty_and_missing_boxes(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        inference_frame = track_video.rotate_inference_frame(frame, 90)
        untracked = Results(orig_img=inference_frame, path="source.jpg", names=NAMES,
                            boxes=torch.tensor([[-5., -10., 110., 220., 0.8, 1.]]))
        restored = track_video.restore_result_orientation(untracked, frame, 90)
        torch.testing.assert_close(restored.boxes.xyxy, torch.tensor([[0., 0., 200., 100.]]))
        self.assertIsNone(restored.boxes.id)
        torch.testing.assert_close(restored.boxes.conf, untracked.boxes.conf)
        for boxes in (None, torch.empty((0, 6)), torch.empty((0, 7))):
            with self.subTest(boxes_none=boxes is None, shape=None if boxes is None else tuple(boxes.shape)):
                result = Results(orig_img=inference_frame, path="source.jpg", names=NAMES, boxes=boxes)
                restored = track_video.restore_result_orientation(result, frame, 90)
                self.assertEqual(restored.orig_shape, frame.shape[:2])
                if boxes is None:
                    self.assertIsNone(restored.boxes)
                else:
                    self.assertEqual(tuple(restored.boxes.data.shape), tuple(boxes.shape))

    def test_rotation_pipeline_writes_original_coordinate_csv_and_frames(self):
        frame = np.full((100, 200, 3), 70, dtype=np.uint8)
        frame[5:15, 10:30] = (10, 50, 200)
        original_frame = frame.copy()
        calls, written, segments = [], [], []

        class Capture:
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FPS: 3, cv2.CAP_PROP_FRAME_WIDTH: 200,
                        cv2.CAP_PROP_FRAME_HEIGHT: 100, cv2.CAP_PROP_FRAME_COUNT: 3}[prop]
            def read(self): return True, frame
            def release(self): pass

        class Writer:
            def isOpened(self): return True
            def write(self, value): written.append(value.copy())
            def release(self): pass

        class Detector:
            names = NAMES
            def track(self, image, **kwargs):
                calls.append(kwargs)
                assert image.shape == (200, 100, 3)
                # Rotated source boxes move right one source pixel each frame.
                offset = len(calls) - 1
                box = torch.tensor([[60., 20. + offset, 90., 70. + offset, 17., 0.8, 2.]])
                return [Results(orig_img=image, path="source.jpg", names=NAMES, boxes=box)]

        real_line = cv2.line
        def record_line(image, start, end, color, thickness, *args, **kwargs):
            if tuple(color) == (0, 255, 255) and thickness == 3:
                segments.append((start, end))
            return real_line(image, start, end, color, thickness, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="stationery_orientation_") as temporary:
            root = Path(temporary)
            model, source, trajectory = root / "model.pt", root / "source.mp4", root / "trajectory.csv"
            model.write_bytes(b"mock model")
            source.write_bytes(b"mock source")
            argv = ["track_video.py", "--model", str(model), "--source", str(source),
                    "--output", str(root / "render.mp4"), "--trajectory", str(trajectory),
                    "--duration", "1", "--imgsz", "1280", "--rotate-inference", "90", "--device", "cpu", "--agnostic-nms"]
            with (patch.object(sys, "argv", argv), patch.object(track_video, "YOLO", return_value=Detector()),
                  patch.object(track_video.cv2, "VideoCapture", return_value=Capture()),
                  patch.object(track_video.cv2, "VideoWriter", return_value=Writer()),
                  patch.object(track_video.cv2, "line", new=record_line), redirect_stdout(io.StringIO())):
                track_video.main()
            with trajectory.open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            for index, row in enumerate(rows):
                self.assertEqual((int(row["center_x"]), int(row["center_y"])), (45 + index, 25))
                self.assertEqual((row["track_id"], row["class_id"], row["class_name"]), ("17", "2", "eraser"))
                self.assertAlmostEqual(float(row["time_sec"]), index / 3, delta=0.00005)
            self.assertTrue(all(call["imgsz"] == 1280 and call["device"] == "cpu" and call["persist"] for call in calls))
            self.assertTrue(all(call["agnostic_nms"] is True for call in calls))
            self.assertEqual([value.shape for value in written], [frame.shape] * 3)
            self.assertIn(((45, 25), (46, 25)), segments)
            self.assertIn(((46, 25), (47, 25)), segments)
            np.testing.assert_array_equal(frame, original_frame)
            self.assertEqual(model.read_bytes(), b"mock model")
            self.assertEqual(source.read_bytes(), b"mock source")


if __name__ == "__main__":
    unittest.main()
