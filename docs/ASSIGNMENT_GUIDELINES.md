# 교수님 과제 지침 및 제공 참고자료

기록일: 2026-11-19 (Asia/Seoul)

사용자는 아래 자료가 **교수님의 과제 지침사항**이라고 확인하고 별도로 기록해 달라고 요청했다. 자료에 표시된 교수·소속은 **이건명, 충북대**이다. 이 문서는 교수님이 제공한 참고자료, 사용자가 처음 요청한 프로젝트 요구사항, 이후 사용자가 선택한 실행 환경을 구분해 보존한다.

## 1. 교수님이 제공한 자료

아래는 사용자가 대화에 붙여 넣은 내용을 항목별로 정리한 것이다. 링크의 불필요한 이스케이프 문자와 붙어 있던 줄바꿈만 정리했다.

### 데이터 세트 만드는 방법 — 참고

- **Label studio(LabelImg)** — 원문 표기
  - [HumanSignal/label-studio](https://github.com/HumanSignal/label-studio)
  - 오픈소스
- **Computer Vision Annotation Tool (CVAT)**
  - [cvat-ai/cvat](https://github.com/cvat-ai/cvat)
  - 오픈소스
- **Roboflow(추천)** — 원문 추천 표시
  - [Roboflow 앱](https://app.roboflow.com)
  - [Ultralytics의 Roboflow 통합 안내](https://docs.ultralytics.com/ko/integrations/roboflow/)

`Label studio(LabelImg)`는 제공 자료의 표기를 보존한 것이며, 두 도구가 동일하다는 별도의 사실 판정은 아니다.

### YOLOv8 학습·추론·배포 — 참고

- 오픈소스 저장소: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- YOLOv8 학습 및 추론, 배포 방법:
  - [Colab tutorial.ipynb](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)
  - [학습 문서](https://docs.ultralytics.com/modes/train/)

### YOLOv8 비디오 객체 추적 — 참고

- [Colab object_tracking.ipynb](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_tracking.ipynb)
- [추적 문서](https://docs.ultralytics.com/modes/track/#real-world-applications)
- 원문에서 설명한 내용:
  - `track`이라는 새로운 메서드 사용
  - “re-ID를 위해 사용하는 듯”
  - [ultralytics/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/trackers)
  - 매 프레임마다 객체 좌표 중점을 Plot으로 만들어주는 방식으로 시각화
  - `trajectory`를 따로 저장하도록 코드 수정 필요

“re-ID를 위해 사용하는 듯”은 제공 자료의 추정 표현을 그대로 기록한 것이다. 모든 추적기가 Re-ID를 사용한다고 확정하는 요구사항으로 바꾸지 않는다. 현재 프로젝트가 사용하는 ByteTrack은 프레임 사이 객체 ID를 연결하며 별도 Re-ID 모델을 사용하지 않는 것으로 README에 설명되어 있다.

### YOLOv8 객체 counting — 참고

- [Colab object_counting.ipynb](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/object_counting.ipynb)

Counting은 위 자료에서 제공한 참고 항목이다. 현재 사용자가 요청한 최종 산출물에 객체 수 집계가 별도 필수 요구사항으로 지정된 것은 아니다.

## 2. 사용자가 요청한 프로젝트 요구사항

다음은 위 참고자료에 새로 덧붙인 교수님 발언이 아니라, 사용자가 처음 이 프로젝트 작업을 요청하면서 명시한 요구사항이다.

- 저장소: https://github.com/jeongjunseo/yolov8-stationery-tracking
- YOLOv8으로 `highlighter`, `correction_tape`, `eraser`의 세 클래스를 학습한다.
- 기존 코드·데이터를 보존하고 `HANDOFF.md`를 작업 인계와 상태의 기준으로 사용한다.
- 클래스 ID는 `0: highlighter`, `1: correction_tape`, `2: eraser`로 유지한다.
- train/valid/test의 이미지·라벨 폴더를 확인하고 `scripts/check_dataset.py`로 검사한다. 데이터가 없으면 학습을 시작하지 않는다.
- 기본 학습 명령은 다음과 같다. 기존 스크립트의 다른 설정을 유지하고, GPU 메모리 부족이 실제로 발생할 때만 batch를 8 또는 4로 낮춘다.

  ```text
  python scripts/train.py --device 0 --batch 16 --epochs 100
  ```

- 학습 모델로 test split을 평가하고 precision, recall, mAP50, mAP50-95와 클래스별 성능 차이·실패 사례를 분석한다.
- 실제 원본 영상이 준비되면 클래스명·추적 ID·이동 궤적을 표시한 **정확히 5초**의 결과 영상을 만든다.
- 프레임별 이동 궤적을 별도 CSV로 저장한다.
- 모델, 성능, 영상, 궤적, 학습 그래프 및 실제 결과를 반영한 보고서를 GitHub에 공개한다.
- 최종 제출 경로는 `models/best.pt`, `outputs/test_metrics.json`, `outputs/final_5s.mp4`, `outputs/trajectory.csv`, 학습 성능 그래프, 완성된 결과보고서이다.
- README와 보고서에 데이터 구성, 학습 설정, 성능, 추적 방식, 실제 경험 및 GitHub 링크를 포함한다. 파일당 100 MiB 제한을 확인한 뒤 업로드 가능한 결과물을 커밋·push한다.

## 3. 최신 실행 방향 — 사용자 지시

2026-11-19 사용자는 “그리고 colab에서 학습을 해”라고 명시했다. 따라서 **이후 학습은 Colab GPU에서 진행한다.**

- 교수님 자료가 Colab 예제를 제공한다는 사실과, 사용자가 실제 실행 환경을 Colab으로 지정했다는 사실을 구분한다. 참고 링크만을 근거로 “교수님이 반드시 Colab만 쓰라고 했다”는 추가 조건을 만들지 않는다.
- 기존 검증 데이터와 프로젝트 코드를 Colab에서 재사용한다. 데이터 세트를 다시 만들거나 재분할하지 않는다.
- 최신 Ultralytics 문서의 예제가 다른 YOLO 버전을 사용하더라도 이 과제의 모델은 YOLOv8n(`yolov8n.pt`)으로 유지한다.
- Colab에서 실제 GPU와 CUDA 사용 가능 상태를 확인하고 기본 batch 16, epochs 100, imgsz 640 및 기존 학습 설정으로 실행한다.
- 로컬 CPU 실행은 사용자의 앞선 요청으로 일시정지되어 있다. Colab으로 전환한다는 요청을 로컬 CPU 실행 재개 요청으로 해석하지 않는다. 로컬 체크포인트·로그와 기존 파일은 보존한다.
- 원본 영상은 사용자가 “준비 예정”이라고 답했다. 현재 존재하지 않는 영상을 생성 완료했다고 기록하거나 합성 테스트 영상을 실제 제출 영상으로 대체하지 않는다.
- 실제 실행 상태와 결과는 `HANDOFF.md` 및 검증된 산출물로 갱신한다. 이 메모 자체는 Colab 연결·학습 완료를 증명하지 않는다.

## 4. 후속 사용자 지시 — 영상 프레임 추가 학습

2026-11-20 사용자는 실제 영상의 오분류 설명을 확인한 뒤 “이 동영상에 있는 것을 이미지로 가져와서 추가 학습 시켜봐”라고 요청했다. 제공 영상의 검토된 프레임·라벨을 새 학습 자료로 사용하는 후속 실험이다. 이는 교수님 원문의 추가 조건이 아니라 사용자의 새 지시다.

기존 모델·데이터·결과는 보존하고 별도 보완 모델을 Colab GPU에서 학습한다. 기존 train과 새 프레임을 함께 사용하며 원래 valid/test 분할은 유지한다. 이 영상으로 학습한 뒤 같은 영상을 추론한 결과는 학습 장면 재검사로 명시하고 독립적인 영상 성능으로 주장하지 않는다.

## 5. 교수님 필수 과제 조건 — 2026-11-21 추가 제공

사용자가 다음 필수 조건을 추가로 제공하고 수행 여부를 확인했다. 앞의 참고자료와 구분해 이 내용을 실제 과제 항목으로 기록한다.

- 데이터 수집: 3개 부류 100개 이상의 이미지 수집
- 3개 부류에 대한 학습
- 학습된 부류에 대한 객체 감지 및 추적 결과를 5초 분량 동영상으로 제작
- 개인별 GitHub에 학습 모델 및 동영상 저장과 공개
- 결과보고서에 학습 과정 및 성능, GitHub 링크, 과제 과정에서 경험한 내용 포함

이 필수 조건에는 100 epoch 또는 추가 학습 100 epoch 완료가 지정돼 있지 않다. 이는 기존 사용자 학습 설정과 후속 실험의 목표이며, 교수님 필수 제출 조건과 구분한다. 사용자가 앞서 선택한 추가 학습 100 epoch 계획 자체를 이 확인 요청으로 취소하지는 않는다.

확인 결과, 기존 데이터의 클래스 포함 이미지 수는 highlighter 219장, correction_tape 129장, eraser 318장으로 각 클래스 100장 이상이다. 기준 모델은 세 클래스를 학습했고, 대응하는 `models/best.pt`, `outputs/final_5s.mp4`, `outputs/trajectory.csv`, `outputs/test_metrics.json`과 학습 그래프를 공개했다. `report/REPORT.md` 3·5절에는 학습 과정·성능, 첫머리에는 GitHub 링크, 7절에는 과정에서 겪은 문제가 기록돼 있다. 기준 영상의 형광펜 미검출·지우개 오분류도 숨기지 않았다.

따라서 위 필수 항목의 작업과 기준 결과 공개는 수행했다. 보완 모델의 최신 5초 영상에서는 세 물체가 정상 표시되지만 해당 보완 가중치와 최종 test 평가는 아직 공개 전이다. 최신 개선 영상을 보완 실험의 최종 결과로 제출할 때에는 대응 모델과 실제 측정 성능을 함께 공개해야 하며, 기준 모델의 test 수치를 대신 붙이지 않는다.
