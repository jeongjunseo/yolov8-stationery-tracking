"""Pure diagnostic matching tests; no model loading, prediction, or data mutation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analyze_failures import box_iou, load_targets, match_detections, select_examples, summarize


NAMES = {0: "highlighter", 1: "correction_tape", 2: "eraser"}


def box(class_id, coordinates):
    return {"class_id": class_id, "xyxy": coordinates}


class FailureMatchingTests(unittest.TestCase):
    def test_best_overlap_wins_and_duplicates_are_false_positives(self):
        targets = [box(0, [0, 0, 10, 10])]
        predictions = [box(0, [0, 0, 8, 10]), box(0, [0, 0, 10, 10])]
        matched = match_detections(targets, predictions)
        self.assertEqual(matched["true_positive_matches"], [
            {"target_index": 0, "prediction_index": 1, "iou": 1.0}
        ])
        self.assertEqual(matched["false_positive_prediction_indices"], [0])
        self.assertEqual(matched["false_negative_target_indices"], [])

    def test_wrong_class_is_one_fp_and_one_fn_not_a_tp(self):
        targets = [box(0, [0, 0, 10, 10])]
        predictions = [box(2, [0, 0, 10, 10])]
        matched = match_detections(targets, predictions)
        self.assertEqual(matched["true_positive_matches"], [])
        self.assertEqual(matched["false_positive_prediction_indices"], [0])
        self.assertEqual(matched["false_negative_target_indices"], [0])
        self.assertEqual(len(matched["wrong_class_overlaps"]), 1)
        record = {"targets": targets, "predictions": predictions, "error_count": 2, **matched}
        summary = summarize([record], NAMES)
        self.assertEqual(summary["overall"]["TP"], 0)
        self.assertEqual(summary["overall"]["FP"], 1)
        self.assertEqual(summary["overall"]["FN"], 1)
        self.assertEqual(summary["per_class"]["highlighter"]["FN"], 1)
        self.assertEqual(summary["per_class"]["eraser"]["FP"], 1)

    def test_same_class_matches_take_priority_over_wrong_class_diagnostics(self):
        targets = [box(0, [0, 0, 10, 10]), box(1, [0, 0, 10, 10])]
        predictions = [box(1, [0, 0, 10, 10])]
        matched = match_detections(targets, predictions)
        self.assertEqual(matched["true_positive_matches"][0]["target_index"], 1)
        self.assertEqual(matched["false_negative_target_indices"], [0])
        self.assertEqual(matched["wrong_class_overlaps"], [])

    def test_empty_targets_predictions_and_iou_boundary(self):
        self.assertEqual(match_detections([], [box(0, [0, 0, 10, 10])])[
            "false_positive_prediction_indices"], [0])
        self.assertEqual(match_detections([box(0, [0, 0, 10, 10])], [])[
            "false_negative_target_indices"], [0])
        self.assertEqual(match_detections([], [])["true_positive_matches"], [])
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 5, 10]), 0.5)
        boundary = match_detections([box(0, [0, 0, 10, 10])], [box(0, [0, 0, 5, 10])])
        self.assertEqual(len(boundary["true_positive_matches"]), 1)
        self.assertEqual(box_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)

    def test_normalized_labels_scale_to_original_image_not_network_size(self):
        with tempfile.TemporaryDirectory(prefix="stationery_label_test_") as temporary:
            label_path = Path(temporary) / "example.txt"
            label_path.write_text("0 0.5 0.25 0.5 0.25\n", encoding="utf-8")
            targets = load_targets(label_path, width=1200, height=800, names=NAMES)
            self.assertEqual(targets[0]["xyxy"], [300, 100, 900, 300])
            label_path.write_text("", encoding="utf-8")
            self.assertEqual(load_targets(label_path, 1200, 800, NAMES), [])
            with self.assertRaises(FileNotFoundError):
                load_targets(Path(temporary) / "missing.txt", 1200, 800, NAMES)

    def test_class_coverage_and_stable_example_selection(self):
        records = []
        for filename, class_id in (("a.jpg", 0), ("b.jpg", 1), ("c.jpg", 2)):
            targets = [box(class_id, [0, 0, 10, 10])]
            records.append({"image": filename, "targets": targets, "predictions": [],
                            "error_count": 1, **match_detections(targets, [])})
        selected = select_examples(list(reversed(records)), NAMES, 3)
        self.assertEqual([item["image"] for item in selected], ["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual(select_examples(records, NAMES, 0), [])


if __name__ == "__main__":
    unittest.main()
