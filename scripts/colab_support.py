"""Colab file handling and checkpoint backup; original training settings stay intact."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
SUPPORTED_ULTRALYTICS_VERSION = "8.4.215"


def require_supported_ultralytics() -> None:
    installed = version("ultralytics")
    if installed != SUPPORTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Explicit OOM retry control requires ultralytics=={SUPPORTED_ULTRALYTICS_VERSION}; "
            f"found {installed}. Review the trainer callback contract before changing this pin."
        )


def enforce_explicit_batch_policy(trainer, requested_batch: int) -> None:
    """Leave OOM retries to the notebook's permitted 16/8/4 attempts."""
    if requested_batch not in (16, 8, 4) or int(trainer.batch_size) != requested_batch:
        raise RuntimeError(
            f"Unexpected training batch: requested {requested_batch}, actual {trainer.batch_size}."
        )
    if not hasattr(trainer, "_oom_retries"):
        raise RuntimeError("Pinned Ultralytics OOM retry state is missing; refusing an unverified fallback policy.")
    # In 8.4.215 this callback runs after _oom_retries is initialized and before
    # the epoch's batches. The trainer retries internally only while it is < 3,
    # including backend errors and batch sizes below 4. Exhausting that internal
    # counter lets the original exception reach the notebook's stricter handler.
    trainer._oom_retries = 3


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_dataset_archive(target: Path, drive_project: Path | None) -> dict:
    """Copy a full archive or verify/reassemble bounded raw parts on local Colab storage."""
    manifest_path = drive_project / "stationery_dataset.parts.json" if drive_project else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.is_file() else None
    if target.is_file():
        if manifest and (target.stat().st_size != manifest["archive_size"]
                         or digest_file(target) != manifest["archive_sha256"]):
            raise ValueError(f"Existing archive checksum mismatch; preserving it: {target}")
        return {"archive": str(target), "source": "existing runtime file", "parts_manifest_verified": bool(manifest)}
    if drive_project is None:
        raise FileNotFoundError(f"Upload the prepared archive using the Colab file panel: {target}")
    complete_source = drive_project / "stationery_dataset.zip"
    if not complete_source.is_file() and manifest is None:
        raise FileNotFoundError("Neither the full dataset ZIP nor its parts manifest exists in the configured Drive folder.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".assembling")
    whole_digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("xb") as output:
            parts = ([{"name": complete_source.name, "bytes": complete_source.stat().st_size}]
                     if complete_source.is_file() else manifest["parts"])
            seen = set()
            for number, part in enumerate(parts, 1):
                name = part["name"]
                if name != Path(name).name or "\\" in name or name in seen:
                    raise ValueError(f"Unsafe or repeated archive part: {name}")
                seen.add(name)
                source = drive_project / name
                if source.stat().st_size != part["bytes"]:
                    raise ValueError(f"Archive part size mismatch: {name}")
                part_digest = hashlib.sha256()
                print(f"Copying/verifying {number}/{len(parts)}: {name}", flush=True)
                with source.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        output.write(chunk)
                        part_digest.update(chunk)
                        whole_digest.update(chunk)
                        total += len(chunk)
                if "sha256" in part and part_digest.hexdigest() != part["sha256"]:
                    raise ValueError(f"Archive part checksum mismatch: {name}")
        if manifest and (total != manifest["archive_size"] or whole_digest.hexdigest() != manifest["archive_sha256"]):
            raise ValueError("Reassembled dataset archive checksum mismatch.")
        if target.exists():
            raise FileExistsError(f"Preserving newly appeared archive: {target}")
        temporary.rename(target)
    except Exception:
        temporary.unlink(missing_ok=True)  # Only the new partial file created by this call.
        raise
    return {"archive": str(target), "bytes": total, "sha256": whole_digest.hexdigest(),
            "parts_manifest_verified": bool(manifest)}


def extract_dataset(archive: Path, root: Path = ROOT) -> dict:
    """Extract only prepared split files; reject traversal and preserve existing files."""
    root = root.resolve()
    restored = identical = skipped = 0
    manifest_verified = 0
    newline_equivalent_metadata = []
    seen = set()
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            relative = PurePosixPath(entry.filename)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in entry.filename
                    or ":" in entry.filename or (entry.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            path_text = relative.as_posix()
            allowed = path_text in {"data/stationery.yaml", "data/SOURCES.md"} or any(
                path_text.startswith(f"data/{split}/{kind}/")
                for split in SPLITS for kind in ("images", "labels")
            )
            if entry.is_dir() or not allowed or relative.name == ".gitkeep":
                skipped += 1
                continue
            if path_text in seen:
                raise ValueError(f"Duplicate archive member: {path_text}")
            seen.add(path_text)
            destination = (root / path_text).resolve()
            if not destination.is_relative_to(root / "data"):
                raise ValueError(f"Outside dataset root: {path_text}")
            # A member is bounded by its source image size, not the whole archive.
            content = zipped.read(entry)
            if destination.exists():
                existing = destination.read_bytes()
                equal = existing == content
                if path_text in {"data/stationery.yaml", "data/SOURCES.md"}:
                    equal = existing.replace(b"\r\n", b"\n") == content.replace(b"\r\n", b"\n")
                if not equal:
                    raise FileExistsError(f"Preserving differing existing file: {destination}")
                identical += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(content)
            restored += 1
        if "dataset_manifest.json" in zipped.namelist():
            manifest = json.loads(zipped.read("dataset_manifest.json"))
            records = manifest["files"]
            expected = {record["path"] for record in records if PurePosixPath(record["path"]).name != ".gitkeep"}
            if expected != seen:
                raise ValueError("Manifest members do not match the extracted dataset members.")
            for record in records:
                relative = record["path"]
                if PurePosixPath(relative).name == ".gitkeep":
                    continue
                destination = root / relative
                expected_size = record.get("size", record.get("bytes"))
                valid = destination.stat().st_size == expected_size and digest_file(destination) == record["sha256"]
                if not valid and relative in {"data/stationery.yaml", "data/SOURCES.md"}:
                    archived = zipped.read(relative)
                    valid = (len(archived) == expected_size
                             and hashlib.sha256(archived).hexdigest() == record["sha256"]
                             and archived.replace(b"\r\n", b"\n") == destination.read_bytes().replace(b"\r\n", b"\n"))
                    if valid:
                        newline_equivalent_metadata.append(relative)
                if not valid:
                    raise ValueError(f"Dataset manifest checksum mismatch: {relative}")
                manifest_verified += 1
    for split in SPLITS:
        for kind in ("images", "labels"):
            directory = root / "data" / split / kind
            if not directory.is_dir() or not any(p.is_file() and p.name != ".gitkeep" for p in directory.iterdir()):
                raise ValueError(f"Missing or empty dataset directory: {directory}")
    return {"restored": restored, "identical_existing": identical, "skipped": skipped,
            "manifest_verified_files": manifest_verified,
            "newline_equivalent_metadata": newline_equivalent_metadata}


def atomic_copy(source: Path, destination: Path) -> None:
    """Replace only this run's backup copy; never move or alter source artifacts."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def train_with_backup(batch: int, backup_root: Path | None) -> None:
    """Call unchanged train.main() with backup and explicit OOM-policy callbacks."""
    require_supported_ultralytics()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Colab GPU runtime; CPU fallback is disabled.")
    if (ROOT / "models" / "best.pt").exists():
        raise FileExistsError("Preserving existing models/best.pt; use a fresh checkout for another experiment.")
    spec = importlib.util.spec_from_file_location("stationery_original_train", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    original_yolo = module.YOLO
    created_models = []
    backup_events = []

    def on_model_save(trainer):
        if backup_root is None:
            return
        run = Path(trainer.save_dir)
        event = {"epoch": int(trainer.epoch) + 1, "run": run.name}
        try:
            for relative in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
                source = run / relative
                if source.is_file():
                    atomic_copy(source, backup_root / run.name / relative)
            event["status"] = "copied"
            print(f"[Drive backup] {run.name}: epoch {event['epoch']}", flush=True)
        except OSError as error:
            event.update(status="failed", error=str(error))
            print(f"[Drive backup WARNING] {error}; training continues locally.", flush=True)
        backup_events.append(event)
        (ROOT / "outputs" / "colab_backup_events.json").write_text(
            json.dumps(backup_events, indent=2) + "\n", encoding="utf-8"
        )

    def yolo_with_backup(*args, **kwargs):
        model = original_yolo(*args, **kwargs)
        model.add_callback("on_train_epoch_start", lambda trainer: enforce_explicit_batch_policy(trainer, batch))
        model.add_callback("on_model_save", on_model_save)
        created_models.append(model)
        return model

    module.YOLO = yolo_with_backup
    sys.argv = [str(ROOT / "scripts" / "train.py"), "--device", "0", "--batch", str(batch), "--epochs", "100"]
    print("Original training entry point:", " ".join([sys.executable, *sys.argv]), flush=True)
    module.main()
    trainer = created_models[-1].trainer
    run = Path(trainer.save_dir)
    record = {
        "requested_device": "0", "device": str(trainer.device),
        "requested_batch": batch, "actual_batch": int(trainer.batch_size),
        "epochs_requested": 100, "epochs_completed": int(trainer.epoch) + 1,
        "run_directory": str(run), "best_source": str(trainer.best),
        "published_model": "models/best.pt", "model_sha256": digest_file(ROOT / "models" / "best.pt"),
        "backup_root": str(backup_root) if backup_root else None,
        "backup_failures": sum(event["status"] == "failed" for event in backup_events),
    }
    (ROOT / "outputs" / "colab_training_run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if (trainer.device.type != "cuda" or record["epochs_completed"] != 100
            or record["actual_batch"] != batch):
        raise RuntimeError(
            f"Expected 100 completed CUDA epochs at batch {batch}; observed "
            f"{record['epochs_completed']} on {trainer.device} at batch {record['actual_batch']}."
        )
    if backup_root is not None:
        atomic_copy(ROOT / "models" / "best.pt", backup_root / "models" / "best.pt")
        atomic_copy(ROOT / "outputs" / "colab_training_run.json", backup_root / "colab_training_run.json")


def collect_artifacts(destination: Path, backup_root: Path | None = None) -> dict:
    training = json.loads((ROOT / "outputs" / "colab_training_run.json").read_text(encoding="utf-8"))
    run = Path(training["run_directory"])
    validation_runs = sorted((ROOT / "runs" / "val").glob("stationery_test*"), key=lambda path: path.stat().st_mtime)
    for kind, directory in [("train", run), *(([("test", validation_runs[-1])]) if validation_runs else [])]:
        for source in directory.iterdir():
            if source.is_file() and source.suffix.lower() in {".png", ".jpg", ".csv", ".yaml"}:
                target = ROOT / "report" / "assets" / kind / source.name
                if target.exists() and digest_file(source) != digest_file(target):
                    raise FileExistsError(f"Preserving existing report artifact: {target}")
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    required = ["models/best.pt", "outputs/test_metrics.json", "outputs/final_5s.mp4", "outputs/trajectory.csv"]
    manifest = {
        "purpose": "Collected measured Colab outputs. A missing source video leaves video deliverables pending.",
        "present": [path for path in required if (ROOT / path).is_file()],
        "pending": [path for path in required if not (ROOT / path).is_file()],
        "report_status": "Existing report prose must be reconciled with these measured outputs before final submission.",
        "training": training,
    }
    manifest_path = ROOT / "outputs" / "colab_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    excluded = {"outputs/environment_check.json", "outputs/training_status.json"}
    files = [ROOT / "models" / "best.pt"]
    for folder in (ROOT / "outputs", ROOT / "report"):
        files.extend(path for path in folder.rglob("*") if path.is_file() and path.name != ".gitkeep")
    files.extend(ROOT / name for name in ("README.md", "HANDOFF.md", "requirements.txt"))
    files.extend((ROOT / "scripts").glob("*.py"))
    files = sorted({path for path in files if path.relative_to(ROOT).as_posix() not in excluded})
    oversized = [path.relative_to(ROOT).as_posix() for path in files if path.stat().st_size > 100 * 1024 * 1024]
    if oversized:
        raise ValueError(f"Files exceed GitHub's 100 MiB limit; handle with LFS before submission: {oversized}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for source in files:
            archive.write(source, source.relative_to(ROOT).as_posix())
    if backup_root is not None:
        atomic_copy(destination, backup_root / destination.name)
    return {**manifest, "archive": str(destination), "archive_sha256": digest_file(destination), "files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("archive", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--batch", type=int, choices=(16, 8, 4), default=16)
    train.add_argument("--backup-root", type=Path)
    collect = commands.add_parser("collect")
    collect.add_argument("destination", type=Path)
    collect.add_argument("--backup-root", type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        print(json.dumps(extract_dataset(args.archive), indent=2))
    elif args.command == "train":
        train_with_backup(args.batch, args.backup_root)
    else:
        print(json.dumps(collect_artifacts(args.destination, args.backup_root), indent=2))


if __name__ == "__main__":
    main()
