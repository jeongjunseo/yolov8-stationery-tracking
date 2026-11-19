from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CLASS_NAMES = {
    0: "highlighter",
    1: "correction_tape",
    2: "eraser",
}
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 문구류 데이터셋 검사")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--min-per-class", type=int, default=100)
    parser.add_argument("--min-total", type=int, default=300)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label(path: Path) -> tuple[set[int], list[str]]:
    classes: set[int] = set()
    errors: list[str] = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}: {line_number}행은 값이 5개가 아닙니다.")
            continue

        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path}: {line_number}행에 숫자가 아닌 값이 있습니다.")
            continue

        if class_id not in CLASS_NAMES:
            errors.append(f"{path}: {line_number}행의 클래스 ID {class_id}가 잘못되었습니다.")
        else:
            classes.add(class_id)

        x_center, y_center, width, height = coordinates
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            errors.append(f"{path}: {line_number}행 좌표가 0~1 범위를 벗어났습니다.")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path}: {line_number}행 너비 또는 높이가 0 이하입니다.")
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            errors.append(f"{path}: {line_number}행 박스가 이미지의 가로 범위를 벗어났습니다.")
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            errors.append(f"{path}: {line_number}행 박스가 이미지의 세로 범위를 벗어났습니다.")

    return classes, errors


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    split_counts: dict[str, int] = {}
    class_image_counts = defaultdict(int)
    hashes: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for split in SPLITS:
        image_dir = data_root / split / "images"
        label_dir = data_root / split / "labels"

        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"필수 폴더가 없습니다: {image_dir} 또는 {label_dir}")
            continue

        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        split_counts[split] = len(images)

        if not images:
            errors.append(f"{split} split에 이미지가 없습니다.")

        image_stems = {path.stem for path in images}
        for label_stem, label_path in labels.items():
            if label_stem not in image_stems:
                errors.append(f"대응 이미지가 없는 라벨입니다: {label_path}")

        for image_path in images:
            hashes[file_hash(image_path)].append((split, image_path))
            label_path = labels.get(image_path.stem)
            if label_path is None:
                warnings.append(f"라벨이 없는 이미지입니다: {image_path}")
                continue

            present_classes, label_errors = validate_label(label_path)
            errors.extend(label_errors)
            for class_id in present_classes:
                class_image_counts[class_id] += 1

    for duplicates in hashes.values():
        duplicate_splits = {split for split, _ in duplicates}
        if len(duplicate_splits) > 1:
            locations = ", ".join(str(path) for _, path in duplicates)
            errors.append(f"서로 다른 split에 완전히 같은 이미지가 있습니다: {locations}")

    total_images = sum(split_counts.values())
    if total_images < args.min_total:
        errors.append(
            f"전체 이미지가 {total_images}장입니다. 최소 {args.min_total}장이 필요합니다."
        )

    for class_id, class_name in CLASS_NAMES.items():
        count = class_image_counts[class_id]
        if count < args.min_per_class:
            errors.append(
                f"{class_name}가 포함된 이미지가 {count}장입니다. "
                f"최소 {args.min_per_class}장이 필요합니다."
            )

    print("\n[split별 이미지 수]")
    for split in SPLITS:
        print(f"- {split}: {split_counts.get(split, 0)}")
    print(f"- total: {total_images}")

    print("\n[클래스별 포함 이미지 수]")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"- {class_id} {class_name}: {class_image_counts[class_id]}")

    if warnings:
        print("\n[경고]")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\n[오류]")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\n데이터셋 검사를 통과했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

