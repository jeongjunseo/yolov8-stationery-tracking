"""Diagnose test errors at fixed thresholds; this is not the official AP evaluator.

The thresholds are fixed before reading test predictions. Same-class candidates
are greedily matched in descending IoU order, one prediction and one target at a
time. Wrong-class overlaps are diagnostic pairs among the remaining FP/FN boxes;
they stay counted as one FP and one FN, rather than becoming true positives.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE = 0.25
NMS_IOU = 0.5
MATCH_IOU = 0.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def box_iou(first: list[float], second: list[float]) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_pairs(targets, predictions, target_ids, prediction_ids, same_class, threshold):
    candidates = []
    for target_id in target_ids:
        for prediction_id in prediction_ids:
            classes_match = targets[target_id]["class_id"] == predictions[prediction_id]["class_id"]
            if classes_match != same_class:
                continue
            overlap = box_iou(targets[target_id]["xyxy"], predictions[prediction_id]["xyxy"])
            if overlap >= threshold:
                candidates.append((-overlap, target_id, prediction_id))
    used_targets, used_predictions, matches = set(), set(), []
    # Index order breaks equal-IoU ties deterministically, independent of hash order.
    for negative_iou, target_id, prediction_id in sorted(candidates):
        if target_id in used_targets or prediction_id in used_predictions:
            continue
        used_targets.add(target_id)
        used_predictions.add(prediction_id)
        matches.append({
            "target_index": target_id,
            "prediction_index": prediction_id,
            "iou": -negative_iou,
        })
    return matches


def match_detections(targets, predictions, threshold: float = MATCH_IOU) -> dict:
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Matching IoU must be finite and in (0, 1].")
    matches = greedy_pairs(
        targets, predictions, range(len(targets)), range(len(predictions)), True, threshold
    )
    used_targets = {pair["target_index"] for pair in matches}
    used_predictions = {pair["prediction_index"] for pair in matches}
    false_negatives = sorted(set(range(len(targets))) - used_targets)
    false_positives = sorted(set(range(len(predictions))) - used_predictions)
    wrong_class = greedy_pairs(
        targets, predictions, false_negatives, false_positives, False, threshold
    )
    return {
        "true_positive_matches": matches,
        "false_negative_target_indices": false_negatives,
        "false_positive_prediction_indices": false_positives,
        "wrong_class_overlaps": wrong_class,
    }


def load_targets(label_path: Path, width: int, height: int, names: dict[int, str]) -> list[dict]:
    """Convert YOLO normalized xywh labels to the original-image pixel coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    targets = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected 5 label values: {label_path}:{line_number}")
        class_id = int(fields[0])
        x, y, box_width, box_height = map(float, fields[1:])
        if class_id not in names:
            raise ValueError(f"Unknown class {class_id}: {label_path}:{line_number}")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x, y, box_width, box_height)):
            raise ValueError(f"Invalid normalized coordinates: {label_path}:{line_number}")
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"Nonpositive box size: {label_path}:{line_number}")
        normalized = [x - box_width / 2, y - box_height / 2, x + box_width / 2, y + box_height / 2]
        if any(value < -1e-6 or value > 1 + 1e-6 for value in normalized):
            raise ValueError(f"Box outside image: {label_path}:{line_number}")
        targets.append({
            "class_id": class_id,
            "class_name": names[class_id],
            "xyxy": [normalized[0] * width, normalized[1] * height,
                     normalized[2] * width, normalized[3] * height],
        })
    return targets


def summarize(records: list[dict], names: dict[int, str]) -> dict:
    per_class = {
        name: {"class_id": class_id, "targets": 0, "predictions": 0, "TP": 0, "FP": 0, "FN": 0}
        for class_id, name in names.items()
    }
    for record in records:
        targets, predictions = record["targets"], record["predictions"]
        for item in targets:
            per_class[names[item["class_id"]]]["targets"] += 1
        for item in predictions:
            per_class[names[item["class_id"]]]["predictions"] += 1
        for pair in record["true_positive_matches"]:
            per_class[names[targets[pair["target_index"]]["class_id"]]]["TP"] += 1
        for index in record["false_negative_target_indices"]:
            per_class[names[targets[index]["class_id"]]]["FN"] += 1
        for index in record["false_positive_prediction_indices"]:
            per_class[names[predictions[index]["class_id"]]]["FP"] += 1
    totals = {key: sum(values[key] for values in per_class.values())
              for key in ("targets", "predictions", "TP", "FP", "FN")}
    for values in [totals, *per_class.values()]:
        values["precision"] = values["TP"] / values["predictions"] if values["predictions"] else None
        values["recall"] = values["TP"] / values["targets"] if values["targets"] else None
    totals["images"] = len(records)
    totals["images_with_errors"] = sum(record["error_count"] > 0 for record in records)
    totals["wrong_class_overlap_pairs"] = sum(len(record["wrong_class_overlaps"]) for record in records)
    return {"overall": totals, "per_class": per_class}


def select_examples(records: list[dict], names: dict[int, str], limit: int) -> list[dict]:
    ranked = sorted(
        (record for record in records if record["error_count"]),
        key=lambda record: (-record["error_count"], -len(record["wrong_class_overlaps"]), record["image"]),
    )
    selected = []
    for class_id in sorted(names):
        for record in ranked:
            error_classes = {
                record["targets"][index]["class_id"] for index in record["false_negative_target_indices"]
            } | {
                record["predictions"][index]["class_id"] for index in record["false_positive_prediction_indices"]
            }
            if class_id in error_classes:
                if record not in selected:
                    selected.append(record)
                break
    selected.extend(record for record in ranked if record not in selected)
    return selected[:limit]


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def save_example(record: dict, source: Path, destination: Path, names: dict[int, str]) -> None:
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot decode image: {source}")
    height, width = frame.shape[:2]
    scale = min(640 / width, 480 / height)
    scaled_width, scaled_height = max(1, round(width * scale)), max(1, round(height * scale))
    preview = cv2.resize(frame, (scaled_width, scaled_height))
    canvas = np.full((scaled_height + 100, 1280, 3), 245, dtype=np.uint8)
    offset_x = (640 - scaled_width) // 2
    for panel, items in enumerate((record["targets"], record["predictions"])):
        image_offset_x = panel * 640 + offset_x
        canvas[55:55 + scaled_height, image_offset_x:image_offset_x + scaled_width] = preview
        failures = set(record["false_negative_target_indices"] if panel == 0
                       else record["false_positive_prediction_indices"])
        for index, item in enumerate(items):
            is_error = index in failures
            color = (0, 0, 220) if is_error else (0, 150, 0)
            box = item["xyxy"]
            x1, y1, x2, y2 = [round(value * scale) for value in box]
            x1, x2 = x1 + image_offset_x, x2 + image_offset_x
            y1, y2 = y1 + 55, y2 + 55
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            status = ("FN" if panel == 0 else "FP") if is_error else "TP"
            label = f"{status} {names[item['class_id']]}"
            if panel == 1:
                label += f" {item['confidence']:.2f}"
            text_y = max(70, min(scaled_height + 45, y1 - 5))
            text_x = max(panel * 640 + 2, min((panel + 1) * 640 - 260, x1))
            cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    titles = ("GROUND TRUTH: red = missed target", "PREDICTIONS: red = false positive")
    for panel, title in enumerate(titles):
        cv2.putText(canvas, title, (panel * 640 + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1)
    footer = (f"conf={CONFIDENCE}, match IoU={MATCH_IOU}; "
              f"TP={len(record['true_positive_matches'])}, "
              f"FP={len(record['false_positive_prediction_indices'])}, "
              f"FN={len(record['false_negative_target_indices'])}; "
              f"wrong-class pairs={len(record['wrong_class_overlaps'])}")
    cv2.putText(canvas, footer, (10, scaled_height + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not success:
        raise RuntimeError(f"Cannot encode example: {destination}")
    with destination.open("xb") as file:
        file.write(encoded.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="고정 임계값 test 오류 분석 (공식 AP 평가와 별도)")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "stationery.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "failure_analysis.json")
    parser.add_argument("--assets", type=Path, default=PROJECT_ROOT / "report" / "assets" / "failures")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-examples", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_examples < 0 or args.imgsz <= 0:
        raise ValueError("max-examples must be nonnegative and imgsz must be positive.")
    model_path, data_path = args.model.resolve(), args.data.resolve()
    output_path, assets_path = args.output.resolve(), args.assets.resolve()
    for required in (model_path, data_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_path.exists() or (assets_path.exists() and any(assets_path.iterdir())):
        raise FileExistsError("Preserving existing analysis output; specify new --output and --assets paths.")

    import yaml
    from ultralytics import YOLO

    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_path.parent / dataset_root
    if not isinstance(config.get("test"), str):
        raise ValueError("This analysis expects a single test image directory in the dataset YAML.")
    image_dir = (dataset_root / config["test"]).resolve()
    label_dir = image_dir.parent / "labels"
    configured_names = config["names"]
    names = (dict(enumerate(configured_names)) if isinstance(configured_names, list)
             else {int(key): value for key, value in configured_names.items()})
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No test images: {image_dir}")
    model = YOLO(str(model_path))
    if model.names != names:
        raise ValueError(f"Model names {model.names} do not match dataset names {names}.")
    records = []
    for number, image_path in enumerate(images, 1):
        result = model.predict(
            str(image_path), device=args.device, imgsz=args.imgsz, conf=CONFIDENCE,
            iou=NMS_IOU, agnostic_nms=False, max_det=300, rect=False,
            save=False, verbose=False,
        )[0]
        height, width = result.orig_shape
        targets = load_targets(label_dir / f"{image_path.stem}.txt", width, height, names)
        predictions = []
        if result.boxes is not None:
            for coordinates, class_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.cls.int().cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                predictions.append({
                    "class_id": class_id, "class_name": names[class_id],
                    "confidence": confidence, "xyxy": coordinates,
                })
        record = {"image": portable_path(image_path), "width": width, "height": height,
                  "targets": targets, "predictions": predictions,
                  **match_detections(targets, predictions)}
        record["error_count"] = (len(record["false_positive_prediction_indices"])
                                 + len(record["false_negative_target_indices"]))
        records.append(record)
        print(f"[{number}/{len(images)}] {image_path.name}: FP+FN={record['error_count']}", flush=True)

    examples = []
    selected = select_examples(records, names, args.max_examples)
    if selected:
        assets_path.mkdir(parents=True, exist_ok=True)
    for number, record in enumerate(selected, 1):
        destination = assets_path / f"failure_{number:02d}.jpg"
        source = Path(record["image"])
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        save_example(record, source, destination, names)
        examples.append({"image": record["image"], "annotated_image": portable_path(destination),
                         "error_count": record["error_count"]})
    summary = {
        "purpose": "Fixed-threshold diagnostic; not official test AP metrics or threshold tuning.",
        "split": "test", "model": portable_path(model_path), "data": portable_path(data_path),
        "settings": {"confidence": CONFIDENCE, "nms_iou": NMS_IOU, "matching_iou": MATCH_IOU,
                     "imgsz": args.imgsz, "rect": False, "agnostic_nms": False,
                     "max_det": 300, "device": args.device},
        "matching": "Same-class candidate pairs at IoU >= 0.5, greedy descending IoU, one-to-one. Equal IoU ties use target then prediction index.",
        "aggregation": "Overall precision and recall are micro averages from total TP/FP/FN at the fixed thresholds; they are not the class-mean confidence-sweep metrics from evaluate.py.",
        "wrong_class_overlap_definition": "Greedy one-to-one different-class pairs among remaining FP/FN boxes at IoU >= 0.5. These remain FP and FN in all counts.",
        "example_selection": "Worst error-count image for each class with errors, then remaining images ranked by FP+FN, wrong-class pairs, and filename; no image repeated.",
        "limitations": [
            "These fixed-threshold counts differ from Ultralytics AP/confidence-sweep evaluation in test_metrics.json.",
            "Errors are relative to supplied labels; missing or inaccurate annotations can appear as model errors.",
            "Images must be visually inspected before attributing errors to lighting, angle, occlusion, or background.",
        ],
        **summarize(records, names), "examples": examples, "images": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"Failure analysis saved: {output_path}")


if __name__ == "__main__":
    main()
