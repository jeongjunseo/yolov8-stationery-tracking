"""Separate CUDA fine-tuning on original train images plus reviewed video frames.

The original weights, data splits and metrics are never replaced. The source
video becomes training material, so later inference on it is a training-scene
demonstration and cannot be described as independent video evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import zipfile

if __package__:
    from .check_dataset import CLASS_NAMES, IMAGE_EXTENSIONS, validate_label
    from .colab_support import (atomic_copy, digest_file, enforce_explicit_batch_policy,
                                require_supported_ultralytics)
else:
    from check_dataset import CLASS_NAMES, IMAGE_EXTENSIONS, validate_label
    from colab_support import (atomic_copy, digest_file, enforce_explicit_batch_policy,
                               require_supported_ultralytics)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA256 = "72eec58f57a0decd694a98606e9759824cb7cb4d75af345e9ae8db735a4d66ea"
OUTPUT = ROOT / "outputs" / "video_finetune"
MODEL = ROOT / "models" / "video_finetune" / "best.pt"


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name
            or path.as_posix() != name):
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def extract_addition(archive: Path, destination: Path, expected_sha256: str) -> dict:
    """Verify a bounded, train-only archive before preserving/extracting members."""
    if digest_file(archive) != expected_sha256:
        raise ValueError("Video addition archive SHA-256 mismatch.")
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        seen = set()
        for entry in zipped.infolist():
            safe_member(entry.filename)
            if entry.filename in seen or entry.is_dir():
                raise ValueError(f"Duplicate or directory member: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink archive member: {entry.filename}")
            if entry.file_size > 100 * 1024 * 1024:
                raise ValueError(f"Oversized video-frame member: {entry.filename}")
            seen.add(entry.filename)
        manifest = json.loads(zipped.read("video_dataset_manifest.json"))
        records = manifest["files"]
        paths = [record["path"] for record in records]
        if len(paths) != len(set(paths)) or set(paths) | {"video_dataset_manifest.json"} != seen:
            raise ValueError("Video manifest membership mismatch.")
        if manifest["split"] != "train" or {int(k): v for k, v in manifest["classes"].items()} != CLASS_NAMES:
            raise ValueError("Video frames must use the original three classes and train split only.")
        for record in records:
            name = record["path"]
            relative = safe_member(name)
            if not (name.startswith("data/train/images/") or name.startswith("data/train/labels/")):
                raise ValueError(f"Video addition must not alter valid/test: {name}")
            content = zipped.read(name)  # Also checks the member CRC.
            import hashlib
            if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"Video-frame manifest checksum mismatch: {name}")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive path escapes addition root: {name}")
            if target.exists() and target.read_bytes() != content:
                raise FileExistsError(f"Preserving differing existing addition: {target}")
        # All members were checked before creating any extracted file.
        for name in [*paths, "video_dataset_manifest.json"]:
            target = destination / name
            if target.exists():
                if target.read_bytes() != zipped.read(name):
                    raise FileExistsError(f"Preserving differing existing manifest: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(zipped.read(name))
    return {"archive_sha256": expected_sha256, "manifest_files": len(records),
            "addition_root": str(destination), "split": "train"}


def validate_addition(addition_root: Path) -> dict:
    from PIL import Image
    images = sorted(p for p in (addition_root / "data/train/images").iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    labels = {p.stem: p for p in (addition_root / "data/train/labels").glob("*.txt")}
    if not images or {p.stem for p in images} != set(labels):
        raise ValueError("Video image/label pairs are empty or inconsistent.")
    counts = Counter()
    for image in images:
        with Image.open(image) as opened:
            opened.verify()
        present, errors = validate_label(labels[image.stem])
        rows = [line.split() for line in labels[image.stem].read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if errors or present != set(CLASS_NAMES) or len(rows) != 3:
            raise ValueError(f"Expected exactly the three reviewed objects in {image.name}: {errors}")
        if not all(math.isfinite(float(value)) for row in rows for value in row[1:]):
            raise ValueError(f"Non-finite annotation: {image.name}")
        counts.update(int(row[0]) for row in rows)
    return {"images": len(images), "labels": len(labels), "boxes": sum(counts.values()),
            "boxes_per_class": {CLASS_NAMES[k]: counts[k] for k in CLASS_NAMES}}


def prepare_data(archive: Path, addition_root: Path, expected_sha256: str, config: Path) -> dict:
    import yaml
    if config.exists():
        raise FileExistsError(f"Preserving an existing mixed-dataset configuration: {config}")
    extraction = extract_addition(archive, addition_root, expected_sha256)
    addition = validate_addition(addition_root)
    counts = {}
    for split in ("train", "valid", "test"):
        directory = ROOT / "data" / split / "images"
        counts[split] = sum(p.suffix.lower() in IMAGE_EXTENSIONS for p in directory.iterdir())
    if counts != {"train": 498, "valid": 116, "test": 73}:
        raise ValueError(f"Unexpected original dataset split: {counts}")
    config_data = {
        "train": [str((ROOT / "data/train/images").resolve()),
                  str((addition_root / "data/train/images").resolve())],
        "val": str((ROOT / "data/valid/images").resolve()),
        "test": str((ROOT / "data/test/images").resolve()),
        "names": CLASS_NAMES,
    }
    config.parent.mkdir(parents=True, exist_ok=True)
    with config.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config_data, stream, sort_keys=False, allow_unicode=True)
    record = {
        "original_counts": counts, "addition": addition, "extraction": extraction,
        "combined_train_images": counts["train"] + addition["images"],
        "config": str(config.resolve()), "config_sha256": digest_file(config),
        "valid_test_unchanged": True,
        "video_is_training_material": True,
        "video_evaluation_scope": "Same-source video inference is a training-scene check, not held-out video accuracy.",
    }
    write_json(OUTPUT / "dataset_check.json", record)
    return record


def train_options(data: Path, batch: int) -> dict:
    if batch not in (16, 8, 4):
        raise ValueError("Only the explicitly permitted batches 16, 8, 4 are supported.")
    return {"data": str(data.resolve()), "epochs": 100, "imgsz": 640,
            "batch": batch, "workers": 4, "project": str(ROOT / "runs/train_video_finetune"),
            "name": f"stationery_video_b{batch}", "seed": 42,
            "deterministic": True, "plots": True, "device": "0"}


def preserved_baseline() -> dict:
    return {name: digest_file(ROOT / name) for name in
            ("models/best.pt", "outputs/test_metrics.json", "scripts/train.py", "data/stationery.yaml")}


def train(data: Path, batch: int, backup_root: Path | None) -> dict:
    require_supported_ultralytics()
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Local CPU fallback is disabled.")
    if MODEL.exists() or (OUTPUT / "training_run.json").exists():
        raise FileExistsError("A completed adaptation already exists; preserving it instead of retraining.")
    baseline = preserved_baseline()
    if baseline["models/best.pt"] != BASELINE_SHA256:
        raise ValueError("The initialization checkpoint differs from the verified baseline.")
    check = json.loads((OUTPUT / "dataset_check.json").read_text(encoding="utf-8"))
    if digest_file(data) != check["config_sha256"]:
        raise ValueError("Mixed-dataset configuration changed after validation.")
    events = []

    def on_save(trainer):
        event = {"epoch": int(trainer.epoch) + 1, "run": str(trainer.save_dir), "status": "local_only"}
        if backup_root is not None:
            try:
                for relative in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
                    source = Path(trainer.save_dir) / relative
                    if source.is_file():
                        atomic_copy(source, backup_root / Path(trainer.save_dir).name / relative)
                event["status"] = "copied"
                print(f"[Drive backup] video fine-tune epoch {event['epoch']}", flush=True)
            except OSError as error:
                event.update(status="failed", error=str(error))
                print(f"[Drive backup WARNING] {error}; local training continues.", flush=True)
        events.append(event)
        path = OUTPUT / f"backup_events_b{batch}.json"
        write_json(path, events)
        if backup_root is not None:
            try:
                atomic_copy(path, backup_root / path.name)
            except OSError as error:
                print(f"[Drive backup WARNING] event record: {error}", flush=True)

    model = YOLO(str(ROOT / "models/best.pt"))
    if model.names != CLASS_NAMES:
        raise ValueError(f"Baseline classes changed: {model.names}")
    model.add_callback("on_train_epoch_start", lambda trainer: enforce_explicit_batch_policy(trainer, batch))
    model.add_callback("on_model_save", on_save)
    options = train_options(data, batch)
    print("Fine-tune initialization:", ROOT / "models/best.pt", flush=True)
    print("Fine-tune options:", json.dumps(options), flush=True)
    model.train(**options)
    trainer = model.trainer
    run = Path(trainer.save_dir)
    with (run / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    actual_epochs = int(trainer.epoch) + 1
    if trainer.device.type != "cuda" or actual_epochs != 100 or len(history) != 100 or int(trainer.batch_size) != batch:
        raise RuntimeError(f"Expected 100 CUDA epochs at batch {batch}; got {actual_epochs} / {len(history)} rows on {trainer.device}.")
    if preserved_baseline() != baseline:
        raise RuntimeError("Baseline files changed during the separate adaptation run.")
    copy_new(Path(trainer.best), MODEL)
    record = {
        "initial_model": "models/best.pt", "initial_model_sha256": BASELINE_SHA256,
        "published_model": "models/video_finetune/best.pt", "model_sha256": digest_file(MODEL),
        "requested_batch": batch, "actual_batch": int(trainer.batch_size),
        "device": str(trainer.device), "epochs_requested": 100, "epochs_completed": actual_epochs,
        "history_rows": len(history), "run_directory": str(run), "best_source": str(trainer.best),
        "training_options": options, "dataset_check": check,
        "checkpoint_selection": "Ultralytics fitness on the unchanged original valid split; not selected by the source video.",
        "backup_root": str(backup_root) if backup_root else None,
        "backup_events": len(events), "backup_failures": sum(e["status"] == "failed" for e in events),
        "baseline_files_preserved": baseline,
        "video_is_training_material": True,
    }
    write_json(OUTPUT / "training_run.json", record)
    if backup_root is not None:
        atomic_copy(MODEL, backup_root / "models/video_finetune/best.pt")
        atomic_copy(OUTPUT / "training_run.json", backup_root / "training_run.json")
    return record


def evaluate(data: Path, batch: int, backup_root: Path | None) -> dict:
    import torch
    from ultralytics import YOLO
    if __package__:
        from .evaluate import per_class_metrics
    else:
        from evaluate import per_class_metrics
    target = OUTPUT / "test_metrics.json"
    if target.exists():
        raise FileExistsError(f"Preserving existing fine-tune test metrics: {target}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab evaluation.")
    training = json.loads((OUTPUT / "training_run.json").read_text(encoding="utf-8"))
    if training["epochs_completed"] != 100 or digest_file(MODEL) != training["model_sha256"]:
        raise ValueError("Completed fine-tune model verification failed.")
    if digest_file(data) != training["dataset_check"]["config_sha256"]:
        raise ValueError("Dataset config changed before evaluation.")
    baseline = preserved_baseline()
    model = YOLO(str(MODEL))
    metrics = model.val(data=str(data.resolve()), split="test", imgsz=640, batch=batch,
                        device="0", project=str(ROOT / "runs/val_video_finetune"),
                        name="stationery_test", plots=True)
    classes = per_class_metrics(metrics, model.names)
    summary = {
        "split": "test", "model": "models/video_finetune/best.pt",
        "model_sha256": training["model_sha256"],
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50), "mAP75": float(metrics.box.map75),
        "mAP50-95": float(metrics.box.map),
        "per_class_mAP50-95": {name: row["mAP50-95"] for name, row in classes.items()},
        "per_class": classes, "run_directory": str(metrics.save_dir),
        "scope": "Original 73-image test split; includes the baseline scene-overlap limitations. No source-video frames are in this split.",
    }
    original = json.loads((ROOT / "outputs/test_metrics.json").read_text(encoding="utf-8"))
    comparison = {
        "baseline_model_sha256": BASELINE_SHA256, "adapted_model_sha256": training["model_sha256"],
        "same_original_test_split": True,
        "metrics": {key: {"baseline": original[key], "adapted": summary[key],
                          "delta": summary[key] - original[key]}
                    for key in ("precision", "recall", "mAP50", "mAP50-95")},
        "source_video_is_training_material": True,
    }
    if preserved_baseline() != baseline:
        raise RuntimeError("Baseline files changed during separate-model evaluation.")
    write_json(target, summary)
    write_json(OUTPUT / "test_comparison.json", comparison)
    if backup_root is not None:
        for path in (target, OUTPUT / "test_comparison.json"):
            atomic_copy(path, backup_root / path.name)
    return summary


def scene_recheck(batch: int, backup_root: Path | None) -> dict:
    """Compare both models on added train frames, explicitly not held-out data."""
    import torch
    import yaml
    from ultralytics import YOLO
    if __package__:
        from .evaluate import per_class_metrics
    else:
        from evaluate import per_class_metrics
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the Colab recheck.")
    training = json.loads((OUTPUT / "training_run.json").read_text(encoding="utf-8"))
    addition = Path(training["dataset_check"]["extraction"]["addition_root"])
    scene_config = addition / "training_scene_recheck.yaml"
    config = {"train": str((addition / "data/train/images").resolve()),
              "val": str((ROOT / "data/valid/images").resolve()), "names": CLASS_NAMES}
    if scene_config.exists():
        if yaml.safe_load(scene_config.read_text(encoding="utf-8")) != config:
            raise FileExistsError(f"Preserving a differing scene check configuration: {scene_config}")
    else:
        with scene_config.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    summaries = {}
    for kind, path, expected in (("baseline", ROOT / "models/best.pt", BASELINE_SHA256),
                                ("finetuned", MODEL, training["model_sha256"])):
        target = OUTPUT / f"training_scene_recheck_{kind}.json"
        if digest_file(path) != expected:
            raise ValueError(f"Rejecting changed {kind} weights.")
        if target.exists():
            previous = json.loads(target.read_text(encoding="utf-8"))
            if (previous.get("model_sha256") != expected
                    or previous.get("scope") != "training_scene_resubstitution"
                    or previous.get("data_sha256") != digest_file(scene_config)
                    or previous.get("batch") != batch or previous.get("imgsz") != 640):
                raise FileExistsError(f"Preserving differing scene recheck: {target}")
            summaries[kind] = previous
            if backup_root is not None:
                atomic_copy(target, backup_root / target.name)
            continue
        model = YOLO(str(path))
        metrics = model.val(data=str(scene_config), split="train", imgsz=640, batch=batch,
                            device="0", project=str(ROOT / "runs/video_training_scene_recheck"),
                            name=kind, plots=True)
        summary = {"scope": "training_scene_resubstitution", "split": "train",
                   "independent_video_evaluation": False, "model_kind": kind,
                   "model_sha256": expected, "images": training["dataset_check"]["addition"]["images"],
                   "data_sha256": digest_file(scene_config),
                   "imgsz": 640, "batch": batch, "precision": float(metrics.box.mp),
                   "recall": float(metrics.box.mr), "mAP50": float(metrics.box.map50),
                   "mAP50-95": float(metrics.box.map), "per_class": per_class_metrics(metrics, model.names),
                   "run_directory": str(metrics.save_dir),
                   "interpretation": "The adapted model trained on these frames. This quantifies fitting this scene, not generalization to another video."}
        if digest_file(path) != expected:
            raise RuntimeError(f"Model bytes changed while checking {kind}.")
        write_json(target, summary)
        summaries[kind] = summary
        if backup_root is not None:
            atomic_copy(target, backup_root / target.name)
    return summaries


def collect(destination: Path, backup_root: Path | None) -> dict:
    training = json.loads((OUTPUT / "training_run.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUTPUT / "test_metrics.json").read_text(encoding="utf-8"))
    if digest_file(MODEL) != training["model_sha256"]:
        raise ValueError("Model changed before artifact collection.")
    for kind, directory in (("train", Path(training["run_directory"])), ("test", Path(metrics["run_directory"]))):
        for source in directory.iterdir():
            if source.is_file() and source.suffix.lower() in {".png", ".jpg", ".csv", ".yaml"}:
                target = ROOT / "report/assets/video_finetune" / kind / source.name
                if target.exists():
                    if digest_file(target) != digest_file(source):
                        raise FileExistsError(f"Preserving differing plot: {target}")
                else:
                    copy_new(source, target)
    files = [MODEL, *sorted(OUTPUT.rglob("*.json")),
             *sorted(p for p in (ROOT / "report/assets/video_finetune").rglob("*") if p.is_file())]
    files = [path for path in files if path.name != "artifact_manifest.json"]
    manifest = {"scope": "Additional video-frame training; existing baseline artifacts are excluded.",
                "source_video_is_training_material": True,
                "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                           "sha256": digest_file(p)} for p in files]}
    if any(record["bytes"] > 100 * 1024 * 1024 for record in manifest["files"]):
        raise ValueError("An artifact exceeds the 100 MiB GitHub file limit.")
    manifest_path = OUTPUT / "artifact_manifest.json"
    write_json(manifest_path, manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for path in [*files, manifest_path]:
            zipped.write(path, path.relative_to(ROOT).as_posix())
    if backup_root is not None:
        atomic_copy(destination, backup_root / destination.name)
    return {"archive": str(destination), "bytes": destination.stat().st_size,
            "sha256": digest_file(destination), "artifact_files": len(files) + 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--archive-sha256", required=True)
    prepare.add_argument("--addition-root", type=Path, required=True)
    prepare.add_argument("--data", type=Path, required=True)
    for command in ("train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--batch", type=int, choices=(16, 8, 4), default=16)
        sub.add_argument("--backup-root", type=Path)
    scene = commands.add_parser("scene-recheck")
    scene.add_argument("--batch", type=int, choices=(16, 8, 4), default=16)
    scene.add_argument("--backup-root", type=Path)
    package = commands.add_parser("collect")
    package.add_argument("destination", type=Path)
    package.add_argument("--backup-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_data(args.archive, args.addition_root, args.archive_sha256, args.data)
    elif args.command == "train":
        result = train(args.data, args.batch, args.backup_root)
    elif args.command == "evaluate":
        result = evaluate(args.data, args.batch, args.backup_root)
    elif args.command == "scene-recheck":
        result = scene_recheck(args.batch, args.backup_root)
    else:
        result = collect(args.destination, args.backup_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
