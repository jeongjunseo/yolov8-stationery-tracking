"""Isolate published fine-tune artifacts immediately after a fresh clone.

This helper refuses dirty, ignored-file-bearing, or linked checkouts. It moves
only three allowlisted artifact trees to a new sibling snapshot, verifies the
file inventory and hashes, and leaves all baseline files unchanged. It must not
be used on an active training checkout. No existing file is deleted or replaced.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ARTIFACT_TREES = (
    "outputs/video_finetune",
    "models/video_finetune",
    "report/assets/video_finetune",
)
BASELINE_FILES = (
    "models/best.pt",
    "outputs/test_metrics.json",
    "scripts/train.py",
    "data/stationery.yaml",
)
REUSABLE_INPUT = "report/assets/video_finetune/annotation_preview.jpg"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def checked_path(root: Path, relative: str) -> Path:
    """Reject linked ancestors as well as final paths outside the named root."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Path escapes the named root: {relative}")
    current = root
    for part in rel.parts:
        current /= part
        if is_link(current):
            raise ValueError(f"Linked artifact path is not allowed: {current}")
    if not current.resolve().is_relative_to(root):
        raise ValueError(f"Resolved path escapes the named root: {current}")
    return current


def inventory(root: Path) -> list[dict]:
    records = []
    for relative in ARTIFACT_TREES:
        directory = checked_path(root, relative)
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(f"Artifact tree is not a directory: {directory}")
        for current, directories, files in os.walk(directory, followlinks=False):
            for name in sorted([*directories, *files]):
                child = checked_path(root, (Path(current) / name).relative_to(root).as_posix())
                if child.is_dir():
                    continue
                if not child.is_file():
                    raise ValueError(f"Nonregular artifact file is not allowed: {child}")
                records.append({"path": child.relative_to(root).as_posix(),
                                "bytes": child.stat().st_size, "sha256": digest(child)})
    return sorted(records, key=lambda record: record["path"])


def write_new_json(path: Path, record: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def preserve_published_artifacts(checkout: Path) -> dict:
    """Called once, directly after clone and before creating sessions or data."""
    checkout = checkout.absolute()
    if is_link(checkout) or not checkout.is_dir():
        raise ValueError(f"Expected a real new checkout directory: {checkout}")
    checkout = checkout.resolve()
    git_root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=checkout, text=True).strip()).resolve()
    if git_root != checkout:
        raise ValueError("Expected the checkout root, not a nested directory.")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--ignored=matching"],
        cwd=checkout, text=True)
    if dirty:
        raise RuntimeError("Only a clean fresh clone is supported; existing work or runtime files are preserved.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    snapshot = checkout.with_name(checkout.name + "_published_snapshot")
    if os.path.lexists(snapshot):
        raise FileExistsError(f"Preserving an existing snapshot: {snapshot}")
    if (snapshot.parent.resolve() != checkout.parent.resolve()
            or snapshot.resolve().is_relative_to(checkout)):
        raise ValueError("Snapshot must be a new sibling outside this checkout.")

    # Complete every path, regular-file, clean-checkout, and baseline check
    # before creating a snapshot or moving any artifact directory.
    records = inventory(checkout)
    trees = [relative for relative in ARTIFACT_TREES if checked_path(checkout, relative).exists()]
    baseline = {}
    for relative in BASELINE_FILES:
        path = checked_path(checkout, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing required baseline file: {path}")
        baseline[relative] = digest(path)
    record = {
        "purpose": "Preserve previously published artifacts before a separate new fine-tune run.",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_checkout": str(checkout), "source_commit": commit,
        "snapshot_directory": str(snapshot), "artifact_trees": trees,
        "files": records, "baseline_sha256_before": baseline,
        "reusable_inputs_restored": [], "status": "planned",
    }
    snapshot.mkdir(exist_ok=False)
    write_new_json(snapshot / "planned_snapshot.json", record)
    for relative in trees:
        source = checked_path(checkout, relative)
        target = checked_path(snapshot, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(target):
            raise FileExistsError(f"Preserving an existing snapshot target: {target}")
        # Both paths were resolved and checked against their explicit roots.
        # Sibling placement keeps the move on the same filesystem; there is
        # no recursive-delete or copy-and-delete fallback.
        source.rename(target)
    if inventory(snapshot) != records:
        raise RuntimeError("Snapshot inventory/hash mismatch; preserve both directories for inspection.")

    # The reviewed annotation thumbnail is shared input provenance, not a
    # previous training result. Only this explicitly named file is copied back.
    reusable = next((item for item in records if item["path"] == REUSABLE_INPUT), None)
    if reusable is not None:
        source = checked_path(snapshot, REUSABLE_INPUT)
        target = checked_path(checkout, REUSABLE_INPUT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if digest(target) != reusable["sha256"]:
            raise RuntimeError("Restored annotation-preview hash mismatch; preserve snapshot for inspection.")
        record["reusable_inputs_restored"].append(reusable)
    after = {relative: digest(checked_path(checkout, relative)) for relative in BASELINE_FILES}
    if after != baseline:
        raise RuntimeError("Baseline files changed; preserve snapshot and checkout for inspection.")
    remaining = inventory(checkout)
    expected_remaining = [reusable] if reusable is not None else []
    if remaining != expected_remaining:
        raise RuntimeError("Previously published outputs remain in the new experiment checkout.")
    record.update(status="verified", baseline_sha256_after=after,
                  snapshot_files_verified=len(records),
                  current_checkout_has_no_published_training_outputs=True)
    write_new_json(snapshot / "snapshot_manifest.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    print(json.dumps(preserve_published_artifacts(args.checkout), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
