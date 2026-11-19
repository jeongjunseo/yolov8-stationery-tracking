from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="학습된 YOLOv8 모델 test split 평가")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "stationery.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="예: 0, cpu")
    return parser.parse_args()


def per_class_metrics(
    metrics, names: dict[int, str] | list[str]
) -> dict[str, dict[str, float | int | None]]:
    """평가 행을 실제 클래스 ID에 대응시키고 평가되지 않은 값은 비워 둔다."""
    rows = {
        int(class_id): row
        for row, class_id in enumerate(metrics.box.ap_class_index)
    }
    target_counts = getattr(metrics, "nt_per_class", None)
    class_items = names.items() if isinstance(names, dict) else enumerate(names)
    results = {}
    for class_id, name in class_items:
        values = {
            "class_id": int(class_id),
            "precision": None,
            "recall": None,
            "mAP50": None,
            "mAP50-95": None,
            "target_count": (
                int(target_counts[class_id])
                if target_counts is not None and 0 <= class_id < len(target_counts)
                else None
            ),
        }
        if class_id in rows:
            # class_result의 인자는 클래스 ID가 아니라 ap_class_index의 행 번호다.
            precision, recall, map50, map50_95 = metrics.box.class_result(rows[class_id])
            values.update(
                precision=float(precision),
                recall=float(recall),
                mAP50=float(map50),
            )
            values["mAP50-95"] = float(map50_95)
        results[str(name)] = values
    return results


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_path = args.data.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"학습 모델을 찾을 수 없습니다: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"데이터 설정 파일을 찾을 수 없습니다: {data_path}")

    model = YOLO(str(model_path))
    val_options = {
        "data": str(data_path),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(PROJECT_ROOT / "runs" / "val"),
        "name": "stationery_test",
        "plots": True,
    }
    if args.device is not None:
        val_options["device"] = args.device

    metrics = model.val(**val_options)
    names = model.names
    class_metrics = per_class_metrics(metrics, names)
    summary = {
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "mAP50-95": float(metrics.box.map),
        "per_class_mAP50-95": {
            name: values["mAP50-95"] for name, values in class_metrics.items()
        },
        "per_class": class_metrics,
    }

    output_path = PROJECT_ROOT / "outputs" / "test_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n[test 성능]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n성능 요약을 저장했습니다: {output_path}")


if __name__ == "__main__":
    main()

