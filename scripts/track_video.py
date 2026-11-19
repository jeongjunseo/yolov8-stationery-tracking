from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack 5초 객체 추적")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "inputs" / "source.mp4")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "final_5s.mp4")
    parser.add_argument(
        "--trajectory", type=Path, default=PROJECT_ROOT / "outputs" / "trajectory.csv"
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--trail-length", type=int, default=60)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    return parser.parse_args()


def class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    return names[class_id]


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    trajectory_path = args.trajectory.resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"학습 모델을 찾을 수 없습니다: {model_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {source_path}")
    if args.duration <= 0:
        raise ValueError("duration은 0보다 커야 합니다.")
    if args.trail_length < 2:
        raise ValueError("trail-length는 2 이상이어야 합니다.")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"입력 영상을 열 수 없습니다: {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError("입력 영상의 FPS를 읽을 수 없습니다.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(round(fps * args.duration))
    if source_frames > 0 and source_frames < max_frames:
        capture.release()
        raise ValueError(
            f"입력 영상이 {source_frames / fps:.2f}초입니다. "
            f"최소 {args.duration:.2f}초가 필요합니다."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"출력 영상을 만들 수 없습니다: {output_path}")

    model = YOLO(str(model_path))
    histories: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=args.trail_length)
    )
    frame_index = 0

    try:
        with trajectory_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "frame",
                    "time_sec",
                    "track_id",
                    "class_id",
                    "class_name",
                    "confidence",
                    "center_x",
                    "center_y",
                ]
            )

            while frame_index < max_frames:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("5초를 처리하기 전에 입력 영상이 끝났습니다.")

                result = model.track(
                    frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                )[0]
                annotated_frame = result.plot(line_width=2, labels=True, conf=True)

                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    centers = boxes.xywh.cpu().tolist()
                    track_ids = boxes.id.int().cpu().tolist()
                    class_ids = boxes.cls.int().cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()

                    for center, track_id, class_id, confidence in zip(
                        centers, track_ids, class_ids, confidences
                    ):
                        center_x, center_y = int(center[0]), int(center[1])
                        histories[track_id].append((center_x, center_y))
                        points = list(histories[track_id])
                        if len(points) >= 2:
                            for start, end in zip(points, points[1:]):
                                cv2.line(annotated_frame, start, end, (0, 255, 255), 3)

                        csv_writer.writerow(
                            [
                                frame_index,
                                round(frame_index / fps, 4),
                                track_id,
                                class_id,
                                class_name(model.names, class_id),
                                round(float(confidence), 6),
                                center_x,
                                center_y,
                            ]
                        )

                writer.write(annotated_frame)
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    print(f"추적 영상 저장 완료: {output_path}")
    print(f"궤적 데이터 저장 완료: {trajectory_path}")
    print(f"처리 길이: {frame_index / fps:.2f}초 ({frame_index} frames)")


if __name__ == "__main__":
    main()

