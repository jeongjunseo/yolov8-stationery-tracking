"""Extract reviewed video annotations into a separate, train-only addition.

This consumes explicit, visually reviewed boxes. It never uses model predictions
as ground truth, changes baseline data, or assigns same-video frames to val/test.
All large outputs default to the ignored runs directory (D: on this workstation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

if __package__:
    from .check_dataset import validate_label
else:
    from check_dataset import validate_label


CLASSES = {"0": "highlighter", "1": "correction_tape", "2": "eraser"}
COLORS = {"0": (0, 180, 0), "1": (0, 100, 255), "2": (255, 0, 0)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def validate_annotations(annotations: dict) -> None:
    if annotations.get("classes") != CLASSES or annotations.get("split") != "train":
        raise ValueError("Expected the unchanged three classes and train-only split")
    if annotations.get("reviewed") is not True:
        raise ValueError("Unreviewed annotation proposals must not become training labels")
    width, height = annotations["width"], annotations["height"]
    seen = set()
    for item in annotations["frames"]:
        frame = item["frame"]
        if not isinstance(frame, int) or frame < 0 or frame >= annotations["source_frame_count"] or frame in seen:
            raise ValueError(f"Invalid or duplicate frame: {frame}")
        seen.add(frame)
        if set(item["boxes"]) != set(CLASSES):
            raise ValueError(f"Expected all three reviewed objects in frame {frame}")
        for class_id, box in item["boxes"].items():
            if len(box) != 4 or not all(isinstance(v, int) for v in box):
                raise ValueError(f"Expected integer xyxy box: {frame}/{class_id}")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Out-of-bounds box: {frame}/{class_id}/{box}")
    if not seen:
        raise ValueError("No reviewed video frames")


def yolo_label(boxes: dict, width: int, height: int) -> bytes:
    rows = []
    for class_id in sorted(boxes, key=int):
        x1, y1, x2, y2 = boxes[class_id]
        values = [float(f"{v:.12f}") for v in ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)]
        # An edge-touching box can round to cx+w/2=1.000000005 at eight
        # decimals. Keep the recorded integer box intact, and move a rounded
        # serialized edge inward by only trillionths if float bounds require it.
        for center_index, extent_index in ((0, 2), (1, 3)):
            center, extent = values[center_index], values[extent_index]
            while center - extent / 2 < 0 or center + extent / 2 > 1:
                extent = float(f"{extent - 0.000000000002:.12f}")
            values[extent_index] = extent
        rows.append(f"{class_id} " + " ".join(f"{v:.12f}" for v in values))
    return ("\n".join(rows) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runs/tracking_20261120_1000030242/source_sdr.mp4"))
    parser.add_argument("--original-video", type=Path, default=Path("../1000030242.mp4"))
    parser.add_argument("--annotations", type=Path, default=Path("outputs/video_finetune_annotations.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/video_finetune_20261120"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/video_finetune_dataset.json"))
    parser.add_argument("--preview", type=Path, default=Path("report/assets/video_finetune/annotation_preview.jpg"))
    parser.add_argument("--archive-name", default="video_training_addition.zip")
    args = parser.parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    validate_annotations(annotations)
    for path, expected in [(args.source, annotations["decoded_video_sha256"]), (args.original_video, annotations["source_video_sha256"])]:
        if sha256(path) != expected:
            raise ValueError(f"Source hash mismatch: {path}")
    root = args.output_root
    if Path(args.archive_name).name != args.archive_name or not args.archive_name.endswith(".zip"):
        raise ValueError("Archive name must be a plain ZIP filename")
    targets = [root / "data", root / "video_dataset_manifest.json", root / args.archive_name, args.summary, args.preview]
    if any(path.exists() for path in targets):
        raise FileExistsError("Output exists; choose new output, summary and preview paths. Nothing overwritten.")
    width, height, fps = annotations["width"], annotations["height"], annotations["fps"]
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise ValueError("Cannot open source video")
    if round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != annotations["source_frame_count"] or abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 1e-6:
        cap.release()
        raise ValueError("Decoded source frame count/FPS changed")
    files, frames, previews = [], [], []
    try:
        for item in sorted(annotations["frames"], key=lambda r: r["frame"]):
            index = item["frame"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame.shape[:2] != (height, width):
                raise ValueError(f"Failed frame or unexpected geometry: {index}")
            stem = f"video1000030242_f{index:06d}"
            image_path = Path("data/train/images") / f"{stem}.jpg"
            label_path = Path("data/train/labels") / f"{stem}.txt"
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            write_new(root / image_path, encoded.tobytes())
            write_new(root / label_path, yolo_label(item["boxes"], width, height))
            present, label_errors = validate_label(root / label_path)
            if label_errors or present != {0, 1, 2}:
                raise ValueError(f"Strict label validation failed: {label_path}: {label_errors}")
            check = cv2.imread(str(root / image_path))
            if check is None or check.shape != frame.shape:
                raise ValueError(f"JPEG readback failed: {image_path}")
            for path in (image_path, label_path):
                files.append({"path": path.as_posix(), "bytes": (root / path).stat().st_size, "sha256": sha256(root / path)})
            frames.append({**item, "image_path": image_path.as_posix(), "label_path": label_path.as_posix()})
            if index in {0, 54, 114, 162, 216, 282}:
                shown = frame.copy()
                for class_id, box in item["boxes"].items():
                    cv2.rectangle(shown, tuple(box[:2]), tuple(box[2:]), COLORS[class_id], 4)
                    cv2.putText(shown, f"GT {class_id}: {CLASSES[class_id]}", (box[0], max(28, box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .8, COLORS[class_id], 2)
                preview = cv2.resize(shown, (360, 640))
                header = np.full((40, 360, 3), 245, np.uint8)
                cv2.putText(header, f"Training annotation | frame {index}", (7, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
                previews.append(np.vstack([header, preview]))
    finally:
        cap.release()
    manifest = {k: value for k, value in annotations.items() if k != "frames"}
    manifest.update({"files": files, "frames": frames, "image_count": len(frames), "label_count": len(frames), "box_count": len(frames) * 3, "jpeg_quality": 95, "geometry_changed": False, "opencv_version": cv2.__version__, "annotation_file_sha256": sha256(args.annotations), "label_serialization": "12 decimal places; rounded size moves inward by 2e-12 only if required for strict floating-point bounds"})
    write_new(root / "video_dataset_manifest.json", json_bytes(manifest))
    if len(previews) != 6:
        raise ValueError("Expected six documented preview samples")
    preview = np.vstack([np.hstack(previews[:3]), np.hstack(previews[3:])])
    ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("Preview encoding failed")
    write_new(args.preview, encoded.tobytes())
    archive = root / args.archive_name
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for record in files:
            output.write(root / record["path"], arcname=record["path"])
        output.write(root / "video_dataset_manifest.json", arcname="video_dataset_manifest.json")
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None or len(check.namelist()) != len(files) + 1:
            raise ValueError("Archive integrity check failed")
        for item in files:
            blob = check.read(item["path"])
            if len(blob) != item["bytes"] or hashlib.sha256(blob).hexdigest() != item["sha256"]:
                raise ValueError(f"Archive content mismatch: {item['path']}")
    if archive.stat().st_size >= 100 * 1024 * 1024:
        raise ValueError("Addition exceeds upload size limit")
    summary = {
        "status": "reviewed_train_only_addition_prepared",
        "source_video": annotations["source_video"], "source_video_sha256": annotations["source_video_sha256"],
        "decoded_video_sha256": annotations["decoded_video_sha256"], "split": "train", "classes": CLASSES,
        "image_count": len(frames), "label_count": len(frames), "box_count": len(frames) * 3,
        "boxes_per_class": {name: len(frames) for name in CLASSES.values()},
        "frame_indices": [item["frame"] for item in frames], "sampling_stride": annotations["selected_stride"],
        "dimensions": [width, height], "jpeg_quality": 95,
        "annotations": args.annotations.as_posix(), "annotation_file_sha256": sha256(args.annotations),
        "manifest_sha256": sha256(root / "video_dataset_manifest.json"),
        "archive": {"path": archive.as_posix(), "bytes": archive.stat().st_size, "sha256": sha256(archive), "members": len(files) + 1},
        "archive_sha256": sha256(archive), "all_file_hashes_and_zip_crc_passed": True,
        "reviewer": annotations["reviewer"], "all_48_frame_contact_sheets_visually_reviewed": True,
        "preview": args.preview.as_posix(), "manual_adjustments": annotations["manual_adjustments"],
        "truncated_highlighter_frames": [item["frame"] for item in frames if item["truncated_classes"]],
        "motion_blur_visible_frames": [item["frame"] for item in frames if item["motion_blur_visible"]],
        "baseline_data_modified": False, "validation_or_test_frames_added": False,
        "limitations": annotations["limitations"],
    }
    for path, expected in [(args.source, annotations["decoded_video_sha256"]), (args.original_video, annotations["source_video_sha256"])]:
        if sha256(path) != expected:
            raise ValueError(f"Source changed during extraction: {path}")
    write_new(args.summary, json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
