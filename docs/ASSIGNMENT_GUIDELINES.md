# 과제 요구사항 및 참고자료

이건명 교수 · 충북대학교

## 과제 요구사항

- 3개 부류에 대해 100개 이상의 이미지 수집
- 3개 부류에 대한 학습
- 학습된 부류의 객체 감지·추적 결과를 5초 분량 동영상으로 제작
- 개인 GitHub에 학습 모델과 동영상 저장 및 공개
- 결과보고서에 학습 과정과 성능, GitHub 링크, 과제 과정에서 경험한 내용 포함

## 프로젝트 설정

- 대상: `0 highlighter`, `1 correction_tape`, `2 eraser`
- 모델·환경: YOLOv8n, Google Colab GPU
- 학습: epochs 100, batch 16, imgsz 640, seed 42
- 평가: test split의 precision, recall, mAP50, mAP50-95와 클래스별 실패 사례
- 추적: 클래스명·ID·이동 궤적을 표시한 5초 영상과 궤적 CSV
- 산출물: `models/best.pt`, `outputs/test_metrics.json`, `outputs/final_5s.mp4`, `outputs/trajectory.csv`, 학습 그래프, 결과보고서
- 저장소: https://github.com/jeongjunseo/yolov8-stationery-tracking

100 epoch는 프로젝트에서 정한 학습 설정이다. 별도의 객체 counting 산출물이나 Re-ID 모델은 위 과제 요구사항에 포함되지 않는다.

영상 프레임을 추가한 보완 실험은 기존 모델에서 시작한다. 기존 train에 영상 48장을 추가하고 validation·test 분할은 유지한다. 보완 결과는 별도 경로에 저장하며, 같은 영상으로 학습한 뒤 얻은 결과는 학습 장면 재검사로 구분한다.

## 데이터 수집·라벨링 참고자료

- Label studio(LabelImg) — 제공 자료의 표기: [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio)
- CVAT: [cvat-ai/cvat](https://github.com/cvat-ai/cvat)
- Roboflow — 추천 도구: [서비스](https://app.roboflow.com) · [Ultralytics 연동 문서](https://docs.ultralytics.com/ko/integrations/roboflow/)

Label Studio와 CVAT는 오픈소스 도구다.

## YOLOv8 학습·추론 참고자료

- [Ultralytics 저장소](https://github.com/ultralytics/ultralytics)
- [Colab 학습 예제](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)
- [학습 문서](https://docs.ultralytics.com/modes/train/)

## 객체 추적·counting 참고자료

- [Colab 추적 예제](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_tracking.ipynb)
- [추적 문서](https://docs.ultralytics.com/modes/track/#real-world-applications)
- [추적기 구현](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/trackers)
- [Colab counting 예제](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_counting.ipynb)

추적 자료는 `track` 메서드를 사용하고, 매 프레임 객체의 중심 좌표를 연결해 시각화하며, trajectory를 별도로 저장하도록 코드를 수정하는 방법을 설명한다. 자료의 “re-ID를 위해 사용하는 듯”은 추정 표현이다. 이 프로젝트의 ByteTrack은 별도 Re-ID 모델 없이 ID를 연결한다.
