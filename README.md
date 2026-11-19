# YOLOv8 문구류 객체 감지 및 추적

형광펜(`highlighter`), 수정테이프(`correction_tape`), 지우개(`eraser`)의 공개 Roboflow 데이터셋을 병합·가공해 YOLOv8을 학습하고, ByteTrack으로 객체 ID와 이동 궤적을 표시하는 과제 프로젝트입니다. 데이터 출처와 CC BY 4.0 표기는 [data/SOURCES.md](data/SOURCES.md)를 따릅니다.

## 현재 진행 상태 — 2026-11-19

데이터 복원·검증과 실행 코드 보완을 완료했습니다. **학습과 성능 평가, 실제 추적 영상 제작은 아직 실행하지 않았습니다.** 현재 PC에서는 AMD Radeon 내장 그래픽만 확인되며 NVIDIA GPU가 없어 CUDA 장치를 사용할 수 없습니다. `inputs/source.mp4`도 제공되지 않았습니다.

- 인계 및 다음 실행 조건: [HANDOFF.md](HANDOFF.md)
- 실제 데이터 감사 결과: [outputs/dataset_summary.json](outputs/dataset_summary.json)
- 확인된 사실과 미완료 항목: [report/REPORT.md](report/REPORT.md)
- 공개 저장소: https://github.com/jeongjunseo/yolov8-stationery-tracking

`models/best.pt`, `outputs/test_metrics.json`, `outputs/final_5s.mp4`, `outputs/trajectory.csv`는 아직 생성되지 않았습니다. 아래 학습·평가·추적 명령은 실행 절차이며 완료된 결과를 의미하지 않습니다.

## 1. 데이터 기준

- 안전 기준: 클래스별 100장 이상, 총 300장 이상
- 권장 분할: train 70%, valid 20%, test 10%
- 같은 장면에서 연속 촬영한 사진은 반드시 같은 split에 배치
- 클래스 번호는 변경하지 않음

| ID | 클래스 | 한글명 |
|---:|---|---|
| 0 | `highlighter` | 형광펜 |
| 1 | `correction_tape` | 수정테이프 |
| 2 | `eraser` | 지우개 |

Roboflow에서 **YOLOv8** 형식으로 내려받은 뒤 아래 구조로 배치합니다.

```text
data/
├─ stationery.yaml
├─ train/
│  ├─ images/
│  └─ labels/
├─ valid/
│  ├─ images/
│  └─ labels/
└─ test/
   ├─ images/
   └─ labels/
```

복원한 기존 데이터의 실제 구성은 다음과 같습니다. 재분할하거나 기존 라벨을 변경하지 않았습니다.

| 구분 | train | valid | test | 합계 |
|---|---:|---:|---:|---:|
| 이미지 / 라벨 파일 | 498 / 498 | 116 / 116 | 73 / 73 | 687 / 687 |
| 이미지 비율 | 72.49% | 16.89% | 10.63% | 100% |
| 객체 박스 | 614 | 146 | 108 | 868 |
| 빈 라벨 파일 | 15 | 4 | 2 | 21 |

| 클래스 | 포함 이미지 | 객체 박스 |
|---|---:|---:|
| highlighter | 219 | 224 |
| correction_tape | 129 | 259 |
| eraser | 318 | 385 |

이미지 687장 디코딩과 기존 검사 스크립트의 좌표·완전 중복 검사는 통과했습니다. 다만 [빈 라벨 이미지 연락시트](outputs/dataset_audit/empty_labels_contact_sheet.jpg)의 02번에는 형광펜이 보이는데 라벨은 비어 있어 추가 검토가 필요합니다. 검사 통과가 라벨 내용의 완전성을 보장하지는 않습니다. 유사 장면이나 연속 프레임의 split 간 누출은 아직 검사하지 않았습니다.

## 2. 설치

PowerShell에서 프로젝트 루트로 이동한 뒤 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

GPU 학습 PC에서는 `nvidia-smi`로 NVIDIA GPU와 드라이버를 확인한 후 [PyTorch 공식 설치 안내](https://docs.pytorch.org/get-started/locally/)에서 해당 환경에 맞는 CUDA 지원 빌드를 설치합니다. 이전 PC의 `.venv`는 복사하지 않습니다. `pip install -r requirements.txt`만으로 CUDA 사용 가능 상태가 보장되지는 않습니다.

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

이번 작업에서는 `torch.cuda.is_available()`이 `True`인 경우에만 GPU 학습을 시작합니다. NVIDIA GPU가 없는 현재 PC에서 CPU 학습으로 임의 전환하지 않습니다.

## 3. 데이터 검사

라벨 범위, 이미지-라벨 대응, split별 수량, 클래스별 포함 이미지 수, split 사이의 완전 중복 이미지를 검사합니다.

```powershell
python scripts/check_dataset.py
```

수업에서 요구한 100장이 전체 합계라는 것이 확인되면 다음처럼 기준을 낮출 수 있습니다.

```powershell
python scripts/check_dataset.py --min-per-class 0 --min-total 100
```

## 4. 학습

YOLOv8 nano 사전 학습 가중치로 100 epoch 전이학습을 수행합니다.

```powershell
python scripts/train.py
```

GPU를 명시하거나 메모리에 맞게 batch 크기를 바꾸려면 다음처럼 실행합니다.

```powershell
python scripts/train.py --device 0 --batch 16 --epochs 100
```

설정은 YOLOv8n, imgsz 640, workers 4, seed 42, deterministic=True입니다. GPU 메모리 부족이 실제로 발생할 때만 batch를 8 또는 4로 낮추고 나머지 조건은 유지합니다.

학습 결과는 `runs/train/` 아래에 저장되고 최적 가중치는 자동으로 `models/best.pt`에 복사됩니다. 보고서에는 다음 자료를 사용합니다.

- `results.csv`: epoch별 loss와 성능
- `results.png`: 학습 곡선
- `confusion_matrix.png`: 혼동 행렬
- `PR_curve.png`: 정밀도-재현율 곡선
- `weights/best.pt`: 최적 학습 모델

## 5. test 데이터 평가

```powershell
python scripts/evaluate.py --device 0
```

전체 Precision, Recall, mAP50, mAP75, mAP50-95와 클래스별 Precision, Recall, mAP50, mAP50-95, 정답 객체 수가 `outputs/test_metrics.json`에 저장됩니다. 평가되지 않은 클래스의 수치는 `null`로 기록합니다. 평가 그래프는 `runs/val/`에 저장됩니다.

## 6. 5초 추적 영상 제작

세 객체가 모두 보이고 천천히 이동하는 5초 이상의 원본 영상을 `inputs/source.mp4`에 둡니다.

```powershell
python scripts/track_video.py --source inputs/source.mp4 --model models/best.pt --device 0
```

생성 결과:

- `outputs/final_5s.mp4`: 클래스명, 신뢰도, 추적 ID, 이동 궤적이 표시된 5초 영상
- `outputs/trajectory.csv`: 프레임별 객체 중심 좌표와 추적 ID

신뢰도 임계값은 필요할 때만 조정합니다.

```powershell
python scripts/track_video.py --conf 0.35 --trail-length 90
```

ByteTrack은 프레임 사이의 객체 ID를 연결하는 추적기이며 별도의 재식별(Re-ID) 모델을 사용하지 않습니다.

출력 프레임 수를 `round(원본 FPS × 5)`로 정하고, 출력 FPS를 `프레임 수 / 5`로 정규화합니다. 예를 들어 29.97 FPS 입력에서는 150프레임을 30 FPS로 저장해 출력 시간축을 5초로 맞춥니다. 이 경우 재생 속도가 약 0.1% 달라지며 CSV의 `time_sec`는 출력 영상 기준입니다. 실제 제출 영상은 생성 후 컨테이너 재생 시간과 프레임 수까지 검증해야 합니다.

기존 영상이나 CSV가 있으면 덮어쓰지 않고 중단합니다. 재실행 시 `--output`과 `--trajectory`로 새로운 경로를 지정합니다.

## 7. 보고서와 GitHub 제출

[report/REPORT.md](report/REPORT.md)는 현재 데이터·환경 검증까지 기록한 진행 보고서입니다. 학습 후 실제 성능과 오류 사례를 추가해야 최종 결과보고서가 됩니다. 공개 저장소에는 최소한 다음 파일을 포함합니다.

```text
models/best.pt
outputs/final_5s.mp4
outputs/test_metrics.json
outputs/trajectory.csv
report/REPORT.md
README.md
scripts/
data/stationery.yaml
```

[GitHub 공식 제한](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)에 따라 파일 하나가 100 MiB를 넘으면 일반 push가 거부되므로, 그런 경우에만 Git LFS를 사용합니다. 원본 데이터 이미지와 라벨은 기본적으로 `.gitignore`에 포함되어 있습니다.

`runs/`도 Git에서 제외되므로 학습 완료 후 실제 생성된 `results.csv`, `results.png`, 혼동 행렬, PR 곡선을 `report/assets/` 등에 복사해 공개해야 합니다. 데이터 감사 그림은 학습 성능 그래프를 대신하지 않습니다.

코드 회귀 검증은 `python -m unittest discover -s tests -v`로 실행합니다. 이 테스트의 영상은 임시 합성 데이터와 모의 검출 결과를 사용하며 실제 학습·검출 성능이나 ByteTrack의 ID 유지 성능을 입증하지 않습니다.

