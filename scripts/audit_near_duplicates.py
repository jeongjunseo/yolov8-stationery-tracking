"""Read-only cross-split similarity screening; candidates require visual review.

No training imports, dataset writes, label edits, or split changes are performed.
The hashes are screening heuristics, not proof of duplicates or independence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from PIL import __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def bits_to_int(values: np.ndarray) -> int:
    result = 0
    for value in values.reshape(-1):
        result = (result << 1) | int(value)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cross_split_groups(groups: dict) -> list:
    return [{"key": key, "images": values} for key, values in sorted(groups.items())
            if len({item["split"] for item in values}) > 1]


def make_sheets(root: Path, selected: list, destination: Path) -> list[str]:
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default(size=16)
    output_names = []
    for offset in range(0, len(selected), 6):
        page = selected[offset:offset + 6]
        sheet = Image.new("RGB", (1000, 360 * len(page)), "#ffffff")
        drawing = ImageDraw.Draw(sheet)
        for row, candidate in enumerate(page):
            y = row * 360
            caption = (f"Pair {candidate['review_index']:02d} | pHash64 {candidate['phash_distance']} | "
                       f"dHash128 {candidate['dhash_distance']} | RGB MAE {candidate['rgb_mae']:.3f}")
            drawing.text((10, y + 4), caption, fill="#111111", font=font)
            for column, side in enumerate(("left", "right")):
                x = column * 500
                record = candidate[side]
                path = root / record["path"]
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    fitted = ImageOps.contain(image, (480, 265), Image.Resampling.LANCZOS)
                    sheet.paste(fitted, (x + (500 - fitted.width) // 2, y + 28 + (265 - fitted.height) // 2))
                label = record["split"] + " | " + Path(record["path"]).name
                drawing.multiline_text((x + 8, y + 295), "\n".join(textwrap.wrap(label, 53)),
                                       fill="#111111", font=font, spacing=2)
        name = f"near_duplicate_candidates_{offset // 6 + 1:02d}.jpg"
        target = destination / name
        if target.exists():
            raise FileExistsError(target)
        sheet.save(target, quality=88)
        output_names.append(name)
    return output_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "dataset_audit")
    parser.add_argument("--max-review-pairs", type=int, default=12)
    parser.add_argument("--threshold-review-pairs", type=int, default=6)
    args = parser.parse_args()
    if args.max_review_pairs < 1 or args.threshold_review_pairs < 0:
        raise ValueError("Review counts must be positive (threshold count may be zero)")
    data, destination = args.data.resolve(), args.output_dir.resolve()
    if destination.is_relative_to(data):
        raise ValueError("Audit output must be outside the dataset directory")
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "near_duplicate_audit.json"
    hash_path = destination / "near_duplicate_image_hashes.json"
    for path in (report_path, hash_path, *destination.glob("near_duplicate_candidates_*.jpg")):
        if path.exists():
            raise FileExistsError(f"Preserving existing audit artifact: {path}; use a new --output-dir")

    dct_basis = np.cos(np.pi * (2 * np.arange(32) + 1)[None, :] * np.arange(32)[:, None] / 64)
    records, thumbs, canonical_groups, time_groups = [], [], defaultdict(list), defaultdict(list)
    for split in SPLITS:
        for path in sorted((data / split / "images").iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            before = path.stat()
            content_sha = sha256_file(path)
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                gray = image.convert("L")
                pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
                low = (dct_basis @ pixels @ dct_basis.T)[:8, :8]
                phash = bits_to_int(low > np.median(low))
                horizontal = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
                vertical = np.asarray(gray.resize((8, 9), Image.Resampling.LANCZOS))
                dhash = (bits_to_int(horizontal[:, 1:] > horizontal[:, :-1]) << 64
                         | bits_to_int(vertical[1:, :] > vertical[:-1, :]))
                thumb = np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.int16)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"Image changed during read-only audit: {path}")
            relative = path.relative_to(ROOT).as_posix()
            canonical = path.name.split(".rf.", 1)[0]
            source_class = next((name for name in ("correction_tape", "highlighter", "eraser")
                                 if path.name.startswith(name + "_")), "unknown")
            record = {"path": relative, "split": split, "source_class": source_class,
                      "width": width, "height": height, "bytes": before.st_size,
                      "sha256": content_sha, "phash64": f"{phash:016x}", "dhash128": f"{dhash:032x}"}
            records.append(record)
            thumbs.append(thumb)
            short = {"path": relative, "split": split}
            canonical_groups[canonical].append(short)
            timestamp = re.fullmatch(r"(eraser_\d{14})(?: \(\d+\))?_jpg", canonical)
            if timestamp:
                time_groups[timestamp.group(1)].append(short)
            if len(records) % 100 == 0:
                print(f"Hashed {len(records)} images", flush=True)

    candidates, exact_pairs = [], []
    comparisons = 0
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            if left["split"] == right["split"]:
                continue
            comparisons += 1
            if left["sha256"] == right["sha256"]:
                exact_pairs.append([left["path"], right["path"]])
            phash_distance = (int(left["phash64"], 16) ^ int(right["phash64"], 16)).bit_count()
            dhash_distance = (int(left["dhash128"], 16) ^ int(right["dhash128"], 16)).bit_count()
            if phash_distance <= 6 and dhash_distance <= 12:
                mae = float(np.abs(thumbs[left_index] - thumbs[right_index]).mean())
                if mae <= 12:
                    candidates.append({"left": left, "right": right, "phash_distance": phash_distance,
                                       "dhash_distance": dhash_distance, "rgb_mae": round(mae, 6)})
    candidates.sort(key=lambda item: (item["phash_distance"], item["dhash_distance"], item["rgb_mae"],
                                       item["left"]["path"], item["right"]["path"]))
    selected, used = [], set()
    for candidate in candidates:
        paths = {candidate["left"]["path"], candidate["right"]["path"]}
        if used & paths:
            continue
        selected.append({**candidate, "review_index": len(selected) + 1})
        used.update(paths)
        if len(selected) >= args.max_review_pairs:
            break
    sheets = make_sheets(ROOT, selected, destination)
    edge_selected = []
    for candidate in reversed(candidates):
        if len(edge_selected) >= args.threshold_review_pairs:
            break
        paths = {candidate["left"]["path"], candidate["right"]["path"]}
        if used & paths:
            continue
        edge_selected.append({**candidate, "review_index": len(selected) + len(edge_selected) + 1})
        used.update(paths)
    edge_sheets = []
    if edge_selected:
        edge_destination = destination / "near_duplicate_threshold"
        edge_destination.mkdir(exist_ok=False)
        edge_sheets = make_sheets(ROOT, edge_selected, edge_destination)
        (edge_destination / "selected_pairs.json").write_text(json.dumps({
            "selection": "Weakest-threshold candidate pairs, disjoint from strongest sample; purposive review, not statistical estimation",
            "pairs": edge_selected, "contact_sheets": edge_sheets,
        }, indent=2) + "\n", encoding="utf-8")
    unique_images = {
        split: len({candidate[side]["path"] for candidate in candidates for side in ("left", "right")
                    if candidate[side]["split"] == split})
        for split in SPLITS
    }
    test_with_train = len({candidate[side]["path"] for candidate in candidates
                          if {candidate["left"]["split"], candidate["right"]["split"]} == {"train", "test"}
                          for side in ("left", "right") if candidate[side]["split"] == "test"})
    report = {
        "purpose": "Read-only screening for cross-split near duplicates; no dataset or training changes",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "Pillow": pillow_version},
        "method": {"phash": "32x32 LANCZOS grayscale, unnormalized DCT-II, median-thresholded low 8x8 coefficients (64 bits)",
                   "dhash": "9x8 horizontal plus 8x9 vertical LANCZOS grayscale comparisons (128 bits)",
                   "thresholds": {"phash_hamming_max": 6, "dhash_hamming_max": 12, "rgb32x32_mae_max": 12},
                   "canonical_filename": "Remove .rf.<hash> suffix, retain source prefix and original filename",
                   "timestamp_family": "eraser 14-digit filename timestamps grouped after removing optional (n); names only, not proof of same capture",
                   "visual_sample": "Strongest candidates sorted by pHash, dHash, RGB MAE, selecting disjoint image pairs"},
        "image_counts": dict(Counter(record["split"] for record in records)),
        "cross_split_pairs_compared": comparisons,
        "cross_split_exact_sha256_pairs": exact_pairs,
        "canonical_filename_cross_split_groups": cross_split_groups(canonical_groups),
        "eraser_same_second_filename_cross_split_groups": cross_split_groups(time_groups),
        "strong_candidate_pair_count": len(candidates),
        "candidate_unique_image_counts": unique_images,
        "test_images_with_train_candidates": test_with_train,
        "candidate_counts_by_source_pair": dict(Counter(" / ".join(sorted({item["left"]["source_class"], item["right"]["source_class"]})) for item in candidates)),
        "candidate_counts_by_split_pair": dict(Counter(" / ".join(sorted({item["left"]["split"], item["right"]["split"]})) for item in candidates)),
        "strong_candidates": candidates,
        "selected_for_visual_review": selected,
        "contact_sheets": sheets,
        "threshold_edge_visual_review_selection": edge_selected,
        "threshold_edge_contact_sheets": ["near_duplicate_threshold/" + name for name in edge_sheets],
        "limitations": ["Thresholds are heuristic; similarity alone does not establish duplicated capture or leakage.",
                        "Hash thresholds can miss crops, rotations, viewpoint changes, and other same-scene frames.",
                        "Simple backgrounds and lookalike objects can produce false positives.",
                        "Filename timestamps are not authenticated capture metadata.",
                        "Visual findings are recorded separately; this file alone does not assert scene independence."]}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hash_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(records), "compared": comparisons, "candidate_pairs": len(candidates),
                      "exact_pairs": len(exact_pairs), "canonical_groups": len(report["canonical_filename_cross_split_groups"]),
                      "timestamp_groups": len(report["eraser_same_second_filename_cross_split_groups"]),
                      "contact_sheets": sheets}, indent=2))


if __name__ == "__main__":
    main()
