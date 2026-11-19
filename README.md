# YOLOv8 문구류 객체 감지 및 추적

형광펜(`highlighter`), 수정테이프(`correction_tape`), 지우개(`eraser`)를 직접 수집·라벨링한 데이터로 YOLOv8을 학습하고, ByteTrack으로 객체 ID와 이동 궤적을 표시하는 과제 프로젝트입니다.

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

## 2. 설치

PowerShell에서 프로젝트 루트로 이동한 뒤 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

NVIDIA GPU가 있는 환경에서는 설치된 CUDA 버전에 맞는 PyTorch가 필요합니다. GPU 인식 여부는 다음 명령으로 확인합니다.

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

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

학습 결과는 `runs/train/` 아래에 저장되고 최적 가중치는 자동으로 `models/best.pt`에 복사됩니다. 보고서에는 다음 자료를 사용합니다.

- `results.csv`: epoch별 loss와 성능
- `results.png`: 학습 곡선
- `confusion_matrix.png`: 혼동 행렬
- `PR_curve.png`: 정밀도-재현율 곡선
- `weights/best.pt`: 최적 학습 모델

## 5. test 데이터 평가

```powershell
python scripts/evaluate.py
```

전체 Precision, Recall, mAP50, mAP50-95와 클래스별 mAP50-95가 `outputs/test_metrics.json`에 저장됩니다. 평가 그래프는 `runs/val/`에 저장됩니다.

## 6. 5초 추적 영상 제작

세 객체가 모두 보이고 천천히 이동하는 5초 이상의 원본 영상을 `inputs/source.mp4`에 둡니다.

```powershell
python scripts/track_video.py
```

생성 결과:

- `outputs/final_5s.mp4`: 클래스명, 신뢰도, 추적 ID, 이동 궤적이 표시된 5초 영상
- `outputs/trajectory.csv`: 프레임별 객체 중심 좌표와 추적 ID

신뢰도 임계값은 필요할 때만 조정합니다.

```powershell
python scripts/track_video.py --conf 0.35 --trail-length 90
```

ByteTrack은 프레임 사이의 객체 ID를 연결하는 추적기이며 별도의 재식별(Re-ID) 모델을 사용하지 않습니다.

## 7. 보고서와 GitHub 제출

`report/REPORT_TEMPLATE.md`를 복사해 실제 수치와 경험을 작성합니다. 공개 저장소에는 최소한 다음 파일을 포함합니다.

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

GitHub는 파일 하나가 100 MiB를 넘으면 일반 push를 거부하므로, 그런 경우에만 Git LFS를 사용합니다. 원본 촬영 이미지와 라벨은 기본적으로 `.gitignore`에 포함되어 있습니다.

