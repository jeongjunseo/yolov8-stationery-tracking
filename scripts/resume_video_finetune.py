"""Resume the verified epoch-97 video experiment in a new CUDA run.

Only the pinned, unstripped checkpoint is accepted. Source files and the old
Drive run are read-only. Epochs 98--100 append to a copied 97-row CSV. This is
standard optimizer/EMA/scaler resume, not bit-identical uninterrupted training.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

if __package__:
    from . import finetune_video as ft
    from .colab_support import atomic_copy, digest_file, enforce_explicit_batch_policy, require_supported_ultralytics
else:
    import finetune_video as ft
    from colab_support import atomic_copy, digest_file, enforce_explicit_batch_policy, require_supported_ultralytics

ROOT = ft.ROOT
OUTPUT = ROOT / "outputs/video_finetune"
MODEL = ROOT / "models/video_finetune/best.pt"
ORIGINAL_RUN_ID = "20261120T123735Z_2e0735a1"
SOURCE_EPOCHS = 97
SOURCE_SECONDS = 9447.17
ADDITION_SHA = "22a7de84e451f3ad9c120773b74f42111c3955864ed21013d7614e83a980f4c9"
SOURCE_HASHES = {
    "last.pt": "c336147c23c3acef1fe55e39a603cb3afb79e8c82bb4f66bca4be0ef18bb8551",
    "best.pt": "61e35e27335a72841e5ded5075d1e05d76460a0aa4b03590c0c656b6f9121976",
    "results.csv": "68055f924f75ca6008d417eb6ef98f150a86aa7ca6533a23c9dc3c0a7a47e77b",
    "args.yaml": "812cc28358fa2d993bad8bdfd2eb086f88c12e581fe6c69b448fa50ea4664841",
    "backup_events_b16.json": "b6f07b7f35a123cb84e1ea0266085bbf9b7032ef99dce7c4c9f07cf47ff79110",
}


def require_no_live_training() -> None:
    """Never start alongside another Python training process in this runtime."""
    import psutil
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.info["pid"] == os.getpid():
            continue
        command = process.info.get("cmdline") or []
        if not command or not re.search(r"python|yolo", process.info.get("name") or "", re.I):
            continue
        leaves = {Path(argument).name for argument in command}
        training = ("train.py" in leaves
                    or (leaves & {"finetune_video.py", "resume_video_finetune.py", "colab_support.py"}
                        and "train" in command))
        if training:
            raise RuntimeError(f"Another training process exists (PID {process.info['pid']}); no process was changed.")


def read_history(path: Path, expected_rows: int) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    rows = [{key.strip(): value for key, value in row.items()} for row in rows]
    if len(rows) != expected_rows or [int(float(row["epoch"])) for row in rows] != list(range(1, expected_rows + 1)):
        raise ValueError("Training CSV epochs are not contiguous from 1 to the expected final epoch.")
    elapsed = [float(row["time"]) for row in rows]
    if not all(math.isfinite(value) for value in elapsed) or any(b <= a for a, b in zip(elapsed, elapsed[1:])):
        raise ValueError("Training CSV cumulative times are not finite and strictly increasing.")
    return rows


def verify_source(source_root: Path) -> dict:
    files = {}
    for name, expected in SOURCE_HASHES.items():
        path = source_root / name
        if path.is_symlink() or not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"Changed or missing pinned recovery source: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": expected}
    rows = read_history(source_root / "results.csv", SOURCE_EPOCHS)
    if float(rows[-1]["time"]) != SOURCE_SECONDS:
        raise ValueError("Unexpected epoch-97 cumulative time.")
    events = json.loads((source_root / "backup_events_b16.json").read_text(encoding="utf-8"))
    if [event["epoch"] for event in events] != list(range(1, 98)) or any(e["status"] != "copied" for e in events):
        raise ValueError("Original 97 successful backup events are inconsistent.")
    return {"files": files, "rows": rows, "events": events}


def validate_checkpoint(checkpoint: dict, rows: list[dict], *, best: bool = False) -> dict:
    wanted = 81 if best else 96
    if checkpoint.get("epoch") != wanted or checkpoint.get("version") != "8.4.215":
        raise ValueError("Wrong checkpoint epoch/version; stripped checkpoints cannot resume.")
    args = checkpoint.get("train_args", {})
    required = {"epochs": 100, "batch": 16, "imgsz": 640, "seed": 42, "deterministic": True,
                "workers": 4, "optimizer": "AdamW", "lr0": 0.001429, "lrf": 0.01,
                "cos_lr": False, "warmup_bias_lr": 0.0, "close_mosaic": 10, "amp": True}
    if any(args.get(key) != value for key, value in required.items()):
        raise ValueError("Checkpoint training settings differ from the original resolved settings.")
    optimizer = checkpoint.get("optimizer")
    if not optimizer or not optimizer.get("state") or len(optimizer.get("param_groups", [])) != 3:
        raise ValueError("Missing optimizer state; refusing a fresh-training fallback.")
    if checkpoint.get("ema") is None or not checkpoint.get("scaler") or not checkpoint.get("updates"):
        raise ValueError("EMA/scaler state required for resume.")
    if checkpoint["ema"].names != ft.CLASS_NAMES:
        raise ValueError("Checkpoint class mapping mismatch.")
    embedded = checkpoint.get("train_results", {})
    if list(embedded.get("epoch", [])) != list(range(1, wanted + 2)):
        raise ValueError("Checkpoint embedded epoch history is inconsistent.")
    for key, values in embedded.items():
        if len(values) != wanted + 1 or key not in rows[0]:
            raise ValueError("Checkpoint embedded history schema/length mismatch.")
        if any(not math.isclose(float(value), float(rows[i][key]), rel_tol=1e-7, abs_tol=1e-8)
               for i, value in enumerate(values)):
            raise ValueError(f"Checkpoint history differs from preserved CSV: {key}")
    if not math.isclose(float(checkpoint.get("best_fitness", -1)), 0.83085, abs_tol=1e-8):
        raise ValueError("Checkpoint best fitness differs from the verified snapshot.")
    return {"epoch_zero_based": wanted, "completed_epochs": wanted + 1,
            "ema_updates": checkpoint["updates"], "best_fitness": checkpoint["best_fitness"],
            "optimizer_state_entries": len(optimizer["state"]), "scaler": checkpoint["scaler"],
            "resolved_train_args": args}


def validate_data(data: Path) -> dict:
    import yaml
    check = json.loads((OUTPUT / "dataset_check.json").read_text(encoding="utf-8"))
    if (digest_file(data) != check["config_sha256"]
            or check["original_counts"] != {"train": 498, "valid": 116, "test": 73}
            or check["combined_train_images"] != 546 or check["addition"]["images"] != 48
            or check["extraction"]["archive_sha256"] != ADDITION_SHA):
        raise ValueError("Restored dataset does not match the original fine-tuning data.")
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    addition = Path(check["extraction"]["addition_root"])
    expected = {"train": [str((ROOT / "data/train/images").resolve()), str((addition / "data/train/images").resolve())],
                "val": str((ROOT / "data/valid/images").resolve()),
                "test": str((ROOT / "data/test/images").resolve()), "names": ft.CLASS_NAMES}
    if config != expected:
        raise ValueError("Mixed YAML paths/classes changed; no new split is permitted.")
    baseline_check = json.loads((OUTPUT / "baseline_dataset_check.json").read_text(encoding="utf-8"))
    if baseline_check.get("check_dataset") != "passed" or baseline_check["extraction"].get("manifest_verified_files") != 1376:
        raise ValueError("Original dataset manifest verification is required.")
    return check


def prepare(source_root: Path, data: Path, backup_root: Path, save_dir: Path | None = None) -> dict:
    require_supported_ultralytics()
    require_no_live_training()
    source_root, data, backup_root = source_root.resolve(), data.resolve(), backup_root.resolve()
    recovery_id = backup_root.name
    if recovery_id == ORIGINAL_RUN_ID or not re.fullmatch(r"[A-Za-z0-9_.-]+", recovery_id):
        raise ValueError("A distinct recovery run ID is required.")
    if ROOT.name != f"stationery_video_finetune_{recovery_id}" or backup_root.parent.name != "colab_runs":
        raise ValueError("Recovery checkout and new Drive run names must match the notebook's unique run ID.")
    if not backup_root.is_dir():
        raise FileNotFoundError("The new Drive backup folder must already be created by the notebook.")
    protected = ("recovery_preparation.json", "resume_provenance.json", "source_backup_events_b16.json",
                 "training_run.json", "training_attempts.json", "backup_events_b16.json", "models",
                 f"recovery_{recovery_id}")
    if any((backup_root / name).exists() for name in protected):
        raise FileExistsError("Existing Drive recovery artifacts are preserved; select a new run folder.")
    save_dir = (save_dir or ROOT / "runs/train_video_finetune" / f"recovery_{recovery_id}").resolve()
    if not save_dir.is_relative_to((ROOT / "runs").resolve()) or save_dir == (ROOT / "runs").resolve():
        raise ValueError("Recovery save_dir must be a new directory within this checkout's runs root.")
    if save_dir.is_relative_to(source_root) or source_root.is_relative_to(save_dir):
        raise ValueError("Recovery source and writable save_dir must be separate trees.")
    if save_dir.exists() or MODEL.exists() or any((OUTPUT / name).exists() for name in
            ("training_run.json", "recovery_preparation.json", "resume_provenance.json", "training_attempts.json")):
        raise FileExistsError("Existing recovery/output files are preserved; use a new checkout/run.")
    verified = verify_source(source_root)
    check = validate_data(data)
    baseline = ft.preserved_baseline()
    if baseline["models/best.pt"] != ft.BASELINE_SHA256:
        raise ValueError("Original baseline initialization model changed.")
    import torch
    last = torch.load(source_root / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(source_root / "best.pt", map_location="cpu", weights_only=False)
    last_summary = validate_checkpoint(last, verified["rows"])
    best_summary = validate_checkpoint(best, verified["rows"], best=True)
    save_dir.mkdir(parents=True, exist_ok=False)
    inputs = save_dir / "recovery_inputs"
    for name in SOURCE_HASHES:
        ft.copy_new(source_root / name, inputs / name)
    for name in ("last.pt", "best.pt"):
        ft.copy_new(inputs / name, save_dir / "weights" / name)
    ft.copy_new(inputs / "results.csv", save_dir / "results.csv")
    ft.copy_new(inputs / "args.yaml", save_dir / "args.yaml")
    ft.copy_new(inputs / "backup_events_b16.json", OUTPUT / "source_backup_events_b16.json")
    record = {"phase": "prepared", "original_run_id": ORIGINAL_RUN_ID, "recovery_run_id": recovery_id,
              "original_drive_backup_root": str(backup_root.parent / ORIGINAL_RUN_ID),
              "recovery_drive_backup_root": str(backup_root),
              "original_run_directory": last["train_args"]["save_dir"],
              "recovery_run_directory": str(save_dir), "recovery_checkout": str(ROOT),
              "source_root": str(source_root), "immutable_local_inputs": str(inputs),
              "source_files": verified["files"], "source_completed_epochs": 97, "source_csv_rows": 97,
              "source_elapsed_seconds": SOURCE_SECONDS, "resumed_epoch_range": [98, 100],
              "source_checkpoint": last_summary, "source_best_checkpoint": best_summary,
              "data": str(data), "dataset_check": check, "baseline_sha256": baseline,
              "runtime_callback_evidence": {},
              "limitation": "Standard optimizer/EMA/scaler resume; checkpoint has no raw live model, pending gradients or RNG state, so bit-identical uninterrupted training is not claimed."}
    verify_source(inputs)
    ft.write_json(OUTPUT / "recovery_preparation.json", record)
    ft.write_json(OUTPUT / "resume_provenance.json", record)
    for name in ("recovery_preparation.json", "resume_provenance.json", "source_backup_events_b16.json"):
        atomic_copy(OUTPUT / name, backup_root / name)
    return record


def equal_state(actual, expected) -> bool:
    if hasattr(expected, "detach"):
        import torch
        return hasattr(actual, "detach") and torch.equal(actual.detach().cpu().float(), expected.detach().cpu().float())
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(equal_state(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (tuple, list)):
        return len(actual) == len(expected) and all(equal_state(a, b) for a, b in zip(actual, expected))
    return actual == expected


def validate_runtime_start(trainer, checkpoint: dict, record: dict) -> dict:
    if (trainer.device.type != "cuda" or trainer.start_epoch != 97 or trainer.epochs != 100
            or trainer.batch_size != 16 or Path(trainer.save_dir).resolve() != Path(record["recovery_run_directory"])):
        raise RuntimeError("Actual resume device/epoch/batch/save_dir differs from the verified plan.")
    for key, expected in checkpoint["train_args"].items():
        if key not in {"model", "data", "save_dir", "name", "resume", "device"} and getattr(trainer.args, key, None) != expected:
            raise RuntimeError(f"Resume changed checkpoint training setting: {key}")
    actual = trainer.optimizer.state_dict()
    expected = checkpoint["optimizer"]
    if len(actual["param_groups"]) != len(expected["param_groups"]) or not equal_state(actual["state"], expected["state"]):
        raise RuntimeError("Optimizer state was not restored exactly from the checkpoint.")
    for left, right in zip(actual["param_groups"], expected["param_groups"]):
        if not equal_state({k: v for k, v in left.items() if k != "fused"}, {k: v for k, v in right.items() if k != "fused"}):
            raise RuntimeError("Optimizer group values changed on resume.")
    if (not equal_state(trainer.scaler.state_dict(), checkpoint["scaler"])
            or trainer.ema.updates != checkpoint["updates"]
            or not equal_state(trainer.ema.ema.state_dict(), checkpoint["ema"].float().state_dict())
            or trainer.best_fitness != checkpoint["best_fitness"] or trainer.scheduler.last_epoch != 96):
        raise RuntimeError("Scaler/EMA/best fitness/scheduler resume validation failed.")
    if any(getattr(trainer.stopper, key, None) != value for key, value in (checkpoint.get("stopper") or {}).items()):
        raise RuntimeError("Early-stopping state was not restored.")
    initial_lrs = [group["initial_lr"] for group in expected["param_groups"]]
    if trainer.scheduler.base_lrs != initial_lrs:
        raise RuntimeError("Scheduler base learning rates changed.")
    trainer.train_time_start -= SOURCE_SECONDS
    return {"start_epoch_zero_based": 97, "next_epoch_one_based": 98, "device": str(trainer.device),
            "optimizer_state_matches": True, "optimizer_groups_match_except_runtime_fused": True,
            "ema_matches": True, "ema_updates": trainer.ema.updates, "scaler_matches": True,
            "scheduler_last_epoch": trainer.scheduler.last_epoch, "scheduler_base_lrs": initial_lrs,
            "cumulative_time_offset_seconds": SOURCE_SECONDS}


def train(data: Path, backup_root: Path) -> dict:
    require_supported_ultralytics()
    require_no_live_training()
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; CPU fallback is disabled.")
    record = json.loads((OUTPUT / "recovery_preparation.json").read_text(encoding="utf-8"))
    data, backup_root = data.resolve(), backup_root.resolve()
    if str(data) != record["data"] or str(backup_root) != record["recovery_drive_backup_root"]:
        raise ValueError("Recovery data/backup paths differ from the prepared new run.")
    if MODEL.exists() or (OUTPUT / "training_run.json").exists() or (OUTPUT / "training_attempts.json").exists():
        raise FileExistsError("Preserving existing recovery execution/output; use a new recovery run.")
    validate_data(data)
    save_dir = Path(record["recovery_run_directory"])
    inputs = Path(record["immutable_local_inputs"])
    verified = verify_source(inputs)
    if ft.preserved_baseline() != record["baseline_sha256"]:
        raise ValueError("Baseline files changed after recovery preparation.")
    for name, target in (("last.pt", save_dir / "weights/last.pt"), ("best.pt", save_dir / "weights/best.pt"),
                         ("results.csv", save_dir / "results.csv"), ("args.yaml", save_dir / "args.yaml")):
        if digest_file(target) != SOURCE_HASHES[name]:
            raise ValueError(f"Prepared recovery file changed before training: {name}")
    checkpoint = torch.load(inputs / "last.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(checkpoint, verified["rows"])
    events = list(verified["events"])
    first_batch_checks = {}
    attempt = {"phase": "recovery", "original_run_id": ORIGINAL_RUN_ID,
               "recovery_run_id": record["recovery_run_id"], "resumed_from_completed_epoch": 97,
               "batch": 16, "exit_code": None, "status": "running"}
    with (save_dir / "recovery_execution.lock").open("x", encoding="utf-8") as lock:
        lock.write(json.dumps({"pid": os.getpid(), "recovery_run_id": record["recovery_run_id"]}))
    ft.write_json(OUTPUT / "training_attempts.json", [attempt])

    def persist_provenance():
        ft.write_json(OUTPUT / "resume_provenance.json", record)
        atomic_copy(OUTPUT / "resume_provenance.json", backup_root / "resume_provenance.json")

    def on_start(trainer):
        record["runtime_callback_evidence"] = validate_runtime_start(trainer, checkpoint, record)
        if Path(trainer.args.data).resolve() != data or digest_file(trainer.csv) != SOURCE_HASHES["results.csv"]:
            raise RuntimeError("Runtime dataset/97-row CSV differs from the recovery plan.")
        record["phase"] = "running"
        record["actual_resume_args"] = vars(trainer.args).copy()
        persist_provenance()

    def on_first_batch(trainer):
        epoch = int(trainer.epoch) + 1
        if epoch in first_batch_checks:
            return
        expected_lr = 0.001429 * (1 - (epoch - 1) / 100 * (1 - 0.01))
        actual = [float(group["lr"]) for group in trainer.optimizer.param_groups]
        if epoch not in (98, 99, 100) or any(not math.isclose(lr, expected_lr, rel_tol=1e-8, abs_tol=1e-12) for lr in actual):
            raise RuntimeError(f"Recovery epoch/learning-rate schedule mismatch: epoch={epoch}, lr={actual}.")
        first_batch_checks[epoch] = {"epoch": epoch, "expected_lr": expected_lr, "actual_lrs": actual}

    def on_save(trainer):
        epoch = int(trainer.epoch) + 1
        if epoch != len(events) + 1 or epoch not in (98, 99, 100):
            raise RuntimeError("Recovery checkpoint sequence differs from epochs 98 through 100.")
        event = {"epoch": epoch, "run": str(trainer.save_dir), "phase": "resumed",
                 "recovery_run_id": record["recovery_run_id"], "status": "copied"}
        try:
            for name in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
                atomic_copy(save_dir / name, backup_root / save_dir.name / name)
        except OSError as error:
            event.update(status="failed", error=str(error))
            print(f"[Drive recovery backup WARNING] {error}; local training continues.", flush=True)
        events.append(event)
        ft.write_json(OUTPUT / "backup_events_b16.json", events)
        try:
            atomic_copy(OUTPUT / "backup_events_b16.json", backup_root / "backup_events_b16.json")
        except OSError as error:
            print(f"[Drive recovery event WARNING] {error}", flush=True)
        print(f"[Drive recovery backup] epoch {epoch}: {event['status']}", flush=True)

    try:
        model = YOLO(str(inputs / "last.pt"))
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_epoch_start", lambda trainer: enforce_explicit_batch_policy(trainer, 16))
        model.add_callback("on_train_batch_start", on_first_batch)
        model.add_callback("on_model_save", on_save)
        # 8.4.215 restores all resolved checkpoint args; data/device/save_dir
        # are the only overrides here. Never use epochs=3 or initialize anew.
        model.train(resume=True, data=str(data), device="0", save_dir=str(save_dir))
        trainer = model.trainer
        history = read_history(save_dir / "results.csv", 100)
        if (trainer.start_epoch != 97 or int(trainer.epoch) + 1 != 100 or trainer.device.type != "cuda"
                or int(trainer.batch_size) != 16 or sorted(first_batch_checks) != [98, 99, 100]
                or len(events) != 100):
            raise RuntimeError("Recovery did not complete the verified 100-epoch CUDA run.")
        if not (save_dir / "results.csv").read_bytes().startswith((inputs / "results.csv").read_bytes()):
            raise RuntimeError("The original 97-row CSV prefix was changed.")
        verify_source(inputs)
        verify_source(Path(record["source_root"]))
        if ft.preserved_baseline() != record["baseline_sha256"]:
            raise RuntimeError("Baseline files changed during recovery.")
        ft.copy_new(Path(trainer.best), MODEL)
        record["phase"] = "complete"
        record["optimizer_restore_validation"] = record["runtime_callback_evidence"]
        record["learning_rate_checks"] = list(first_batch_checks.values())
        record["final_validation"] = {"epochs_completed": 100, "new_epochs_completed": 3,
            "csv_rows": len(history), "source_csv_prefix_unchanged": True,
            "source_files_unchanged": True, "baseline_files_unchanged": True,
            "final_elapsed_seconds": float(history[-1]["time"])}
        persist_provenance()
        training = {"initial_model": "models/best.pt", "initial_model_sha256": ft.BASELINE_SHA256,
            "published_model": "models/video_finetune/best.pt", "model_sha256": digest_file(MODEL),
            "requested_batch": 16, "actual_batch": 16, "device": str(trainer.device),
            "epochs_requested": 100, "epochs_completed": 100, "history_rows": 100,
            "run_directory": str(save_dir), "best_source": str(trainer.best),
            "training_options": vars(trainer.args).copy(), "dataset_check": record["dataset_check"],
            "checkpoint_selection": "Original valid-split fitness, including preserved epoch-82 best and resumed epochs 98--100.",
            "backup_root": str(backup_root), "backup_events": len(events),
            "backup_failures": sum(event["status"] == "failed" for event in events),
            "baseline_files_preserved": record["baseline_sha256"], "video_is_training_material": True,
            "resumed_from_checkpoint_sha256": SOURCE_HASHES["last.pt"],
            "resume_provenance_path": "outputs/video_finetune/resume_provenance.json"}
        ft.write_json(OUTPUT / "training_run.json", training)
        attempt.update(exit_code=0, status="completed")
        ft.write_json(OUTPUT / "training_attempts.json", [attempt])
        for name in ("training_run.json", "training_attempts.json", "backup_events_b16.json"):
            atomic_copy(OUTPUT / name, backup_root / name)
        atomic_copy(MODEL, backup_root / "models/video_finetune/best.pt")
        return training
    except Exception as error:
        attempt.update(exit_code=1, status="failed", error_type=type(error).__name__, error=str(error))
        ft.write_json(OUTPUT / "training_attempts.json", [attempt])
        record.update(phase="failed", failure={"type": type(error).__name__, "message": str(error)})
        ft.write_json(OUTPUT / "resume_provenance.json", record)
        try:
            for name in ("training_attempts.json", "resume_provenance.json"):
                atomic_copy(OUTPUT / name, backup_root / name)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        command = commands.add_parser(name)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--backup-root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--source-root", type=Path, required=True)
            command.add_argument("--save-dir", type=Path)
    args = parser.parse_args()
    result = prepare(args.source_root, args.data, args.backup_root, args.save_dir) if args.command == "prepare" else train(args.data, args.backup_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
