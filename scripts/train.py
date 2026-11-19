from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 문구류 객체 감지 모델 학습")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "stationery.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="예: 0, cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="stationery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"데이터 설정 파일을 찾을 수 없습니다: {data_path}")

    model = YOLO(args.model)
    train_options = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(PROJECT_ROOT / "runs" / "train"),
        "name": args.name,
        "seed": 42,
        "deterministic": True,
        "plots": True,
    }
    if args.device is not None:
        train_options["device"] = args.device

    model.train(**train_options)

    best_weight = Path(model.trainer.best)
    if not best_weight.is_file():
        raise FileNotFoundError(f"최적 가중치를 찾을 수 없습니다: {best_weight}")

    destination = PROJECT_ROOT / "models" / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, destination)
    print(f"\n최적 가중치를 복사했습니다: {destination}")


if __name__ == "__main__":
    main()

