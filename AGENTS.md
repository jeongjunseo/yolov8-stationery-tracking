# 프로젝트 작업 지침

작업을 시작할 때 이 파일과 `HANDOFF.md`, `README.md`, `docs/ASSIGNMENT_GUIDELINES.md`를 읽는다. 최신 사용자 지시가 우선이며, `HANDOFF.md`의 가장 최근 기록을 실행 상태의 기준으로 삼는다.

## 과제와 현재 실행 방향

- 교수님이 제공한 과제 참고자료와 사용자 요구사항은 `docs/ASSIGNMENT_GUIDELINES.md`에 구분해 기록되어 있다. 참고자료의 추정 표현을 확정된 기술 사실이나 새로운 필수 조건으로 바꾸지 않는다.
- 사용자가 2026-11-19 **Colab에서 학습하라**고 명시했다. 이후 학습은 Colab GPU에서 수행한다. 최신 문서의 모델명이 달라도 이 과제는 YOLOv8n(`yolov8n.pt`)을 사용한다.
- 기본 학습은 epochs 100, batch 16, imgsz 640, seed 42 및 기존 `scripts/train.py` 설정을 유지한다. GPU 메모리 부족이 실제로 발생하면 batch만 8 또는 4로 낮추고 원인을 기록한다.
- 클래스 ID는 `0: highlighter`, `1: correction_tape`, `2: eraser`로 유지한다.
- 사용자가 2026-11-20 업로드 영상의 이미지를 추출해 추가 학습하라고 명시했다. 이 후속 실험은 기존 `models/best.pt`에서 시작하며 원래 train 데이터와 새 영상의 검토된 라벨을 함께 사용한다. 새 영상은 train 전용이고 기존 valid/test 분할을 유지한다. 같은 영상의 재추론 결과는 학습 장면 재검사이며 독립적인 일반화 평가로 보고하지 않는다.
- 보완 모델·지표·영상은 `models/video_finetune/`, `outputs/video_finetune/` 등 별도 경로에 저장한다. 현재 기준 모델과 기존 공개 결과는 덮어쓰지 않는다. 추가 학습 역시 Colab GPU, batch 16, epochs 100, imgsz 640 및 기존 학습 설정을 사용한다.

## 기존 작업 보존

- 데이터·라벨·원본 압축파일·사용자 파일·이전 결과를 삭제하거나 임의로 재생성·초기화하지 않는다. Colab에서도 기존 데이터 분할을 재사용한다.
- 로컬 CPU 학습은 사용자 요청으로 일시정지되어 있다. Colab 작업 때문에 이를 자동 재개하거나 중복 실행하지 않는다. 상태와 재개 정보는 로컬 `outputs/training_status.json` 및 `HANDOFF.md`를 확인한다.
- 다른 컴퓨터의 `.venv`나 Windows junction을 Colab으로 복사하지 않는다. Colab 런타임의 의존성·GPU를 확인한다.
- 데이터가 실제로 있는지 확인하고 `scripts/check_dataset.py`를 실행한 뒤 학습한다. 빈 라벨의 의미적 이상과 교차 분할의 지우개 유사 장면 문제는 현재 데이터의 한계로 기록되어 있으므로 숨기지 않는다. `outputs/dataset_audit/near_duplicate_audit.json`의 자동 후보 수와 실제 시각 검토 결과를 구분하며, 모든 후보를 확정 중복이라고 부르지 않는다.

## 검증과 제출

- 실제 실행에서 측정된 성능만 보고한다. 진행 중 결과, validation 지표, test 지표, 합성 회귀 테스트를 구분한다.
- 사용자가 제공한 실제 원본 `1000030242.mp4`로 5초 영상과 CSV를 생성했다. 원본·중간 실험 파일은 보존하며 합성 영상이나 수동으로 만든 검출로 대체하지 않는다. 재실행은 새 출력 경로를 사용한다.
- 영상 파일의 시간·좌표 검증과 실제 세 물체의 검출 성공을 구분한다. 현재 영상에는 실제 형광펜 미검출과 지우개를 highlighter로 오분류하는 문제가 남아 있다. `outputs/tracking_verification.json`의 결과를 기준으로 이 한계를 숨기지 않는다. 기존 학습 모델·공식 test 지표를 영상에 맞춰 덮어쓰지 않는다.
- 최종 영상은 정확히 5초이며 클래스명·추적 ID·이동 궤적을 표시해야 한다. 궤적은 `outputs/trajectory.csv`에도 저장하고 영상과 시간·좌표가 일치하는지 검증한다.
- 최종 모델은 `models/best.pt`, test 지표는 `outputs/test_metrics.json`, 영상은 `outputs/final_5s.mp4`에 둔다. 학습·평가 그래프, 실제 실패 사례, 보고서 및 README도 갱신한다.
- Git에서 제외된 `runs/`의 제출용 그래프는 공개 경로로 복사한다. 파일당 100 MiB 제한을 확인하고, 원본 데이터·환경·민감 정보를 의도치 않게 커밋하지 않는다.
