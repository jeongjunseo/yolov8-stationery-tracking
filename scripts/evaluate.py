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
    class_maps = metrics.box.maps.tolist()
    summary = {
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "mAP50-95": float(metrics.box.map),
        "per_class_mAP50-95": {
            str(names[class_id]): float(class_map)
            for class_id, class_map in enumerate(class_maps)
        },
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

