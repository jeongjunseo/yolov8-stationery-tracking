# YOLOv8 문구류 객체 감지 및 추적

형광펜(`highlighter`), 수정테이프(`correction_tape`), 지우개(`eraser`)의 공개 Roboflow 데이터셋을 병합·가공해 YOLOv8을 학습하고, ByteTrack으로 객체 ID와 이동 궤적을 표시하는 과제 프로젝트입니다. 데이터 출처와 CC BY 4.0 표기는 [data/SOURCES.md](data/SOURCES.md)를 따릅니다.

**최신 영상 재검증 — 2026-11-21:** 현재까지 추가 학습한 모델로 원본 전체 287프레임을 새로 추론했습니다. [5초 결과](outputs/video_finetune/recheck_5s.mp4)는 세 물체의 클래스·ID가 유지됩니다. [전체 약 9.57초 결과](outputs/video_finetune/recheck_full.mp4)에는 후반 형광펜 ID 변경과 바닥을 수정테이프로 잡는 오탐이 남아 있습니다. [검증 기록](outputs/video_finetune/video_recheck.json)은 97 epoch까지 완료된 실행의 best82를 사용한 학습 장면 재검사이며, 최종 test 평가와 구분합니다.

**제출 점검 — 2026-11-21:** 기본 100 epoch 실험의 모델·test 지표·5초 영상·궤적 CSV·그래프·보고서는 모두 공개돼 있습니다. 위 최신 개선 영상의 모델은 공개된 `models/best.pt`와 다릅니다. 보완 실험은 97/100 epoch 상태로, 최종 `models/video_finetune/best.pt`와 `outputs/video_finetune/test_metrics.json`은 아직 없습니다. 따라서 기존 test 수치를 최신 개선 영상 모델의 성능으로 제출하면 안 됩니다. GitHub 공개는 확인했으며 학교 제출 시스템의 제출 여부는 확인할 기록이 없습니다.

## 기준 모델의 완료된 실험

교수님이 제공하신 자료는 [과제 지침 메모](docs/ASSIGNMENT_GUIDELINES.md)에 별도로 보존했습니다. **Colab Tesla T4의 CUDA 장치 0에서 batch 16으로 100 epoch 학습과 test 평가를 완료했습니다.** 기존 데이터 687장의 분할·라벨·설정을 유지했으며, 학습 CSV의 누적 시간은 **2시간 43분 54.78초**입니다. GPU 메모리 부족이나 batch 축소 없이 한 번의 시도로 완료했고, 100회의 epoch 백업이 성공했습니다.

Test 73장·정답 객체 108개에서 **Precision 93.70%, Recall 88.77%, mAP50 93.58%, mAP50-95 76.68%**를 얻었습니다. 이 수치는 아래의 빈 라벨과 eraser 유사 장면 한계를 포함한 기존 분할의 결과입니다. 사용자가 제공한 실제 영상으로 **5초 결과 영상과 궤적 CSV를 생성했습니다. 다만 위쪽 노란 형광펜을 놓쳤고, 지우개는 형광펜으로도 오분류되어 세 실제 물체의 정확한 추적 목표는 달성하지 못했습니다.**

로컬 CPU 학습은 batch 16·8의 메모리 오류 후 batch 4로 1 epoch를 완료했습니다. 2번째 epoch 44%에서 19:28 KST 사용자 요청으로 일시정지했으며 메모리 상태·체크포인트·로그를 보존했습니다. 이 중간 결과를 최종 모델 또는 test 성능으로 사용하지 않습니다. 영상 추론은 완료된 Colab 모델을 로컬 CPU에서 실행했으며, 일시정지한 학습을 재개하지 않았습니다.

- 인계 및 다음 실행 조건: [HANDOFF.md](HANDOFF.md)
- 실제 데이터 감사 결과: [outputs/dataset_summary.json](outputs/dataset_summary.json)
- 실제 Colab 환경과 데이터 검사: [colab_environment.json](outputs/colab_environment.json), [colab_dataset_check.json](outputs/colab_dataset_check.json)
- 학습 완료·산출물 검증: [colab_training_run.json](outputs/colab_training_run.json), [artifact_verification.json](outputs/artifact_verification.json)
- 학습 모델·test 지표: [best.pt](models/best.pt), [test_metrics.json](outputs/test_metrics.json)
- 실제 5초 영상·궤적: [final_5s.mp4](outputs/final_5s.mp4), [trajectory.csv](outputs/trajectory.csv)
- 로컬 CPU 실행 상태: `outputs/training_status.json` (로컬 전용, Git 제외)
- 확인된 사실과 미완료 항목: [report/REPORT.md](report/REPORT.md)
- 공개 저장소: https://github.com/jeongjunseo/yolov8-stationery-tracking

`models/best.pt`는 6,500,778 bytes이며 Colab 기록과 SHA-256이 일치합니다. 로컬 CPU에서 모델을 불러와 세 클래스와 파일 무변경을 확인했습니다. 아래 명령은 완료된 학습·평가와 실제 영상 처리의 재현 절차입니다.

## 영상 프레임을 추가한 보완 학습

사용자는 실제 영상의 물체를 이미지로 추출해 추가 학습하도록 요청했습니다. 이 후속 실험은 완료된 `models/best.pt`에서 시작하며, 기존 학습 이미지 498장에 영상에서 0.2초 간격으로 추출한 48장과 검토한 세 물체의 정답 박스를 추가합니다. 원래 validation 116장·test 73장은 유지합니다.

보완 학습은 Colab GPU에서 batch 16, epochs 100, imgsz 640 및 기존 설정으로 실행하며, 실제 CUDA 메모리 부족일 때만 batch를 8/4로 낮춥니다. 모델은 `models/video_finetune/best.pt`, 측정 기록은 `outputs/video_finetune/`에 저장해 기존 결과를 보존합니다. run `20261120T123735Z_2e0735a1`의 T4·CUDA와 **97 epoch·97회 성공한 Drive 백업**을 확인했습니다. 이후 중단과 GPU 재할당 제한을 사용자가 확인했습니다. **보완 학습 100 epoch 완료와 최종 test 평가는 대기 중**이며, 현재 모델의 영상 재검증 결과를 별도로 공개했습니다. 재개 가능한 체크포인트와 기록을 보존했으므로 노트북을 처음부터 다시 학습시키지 않습니다. 확인 시점과 최신 관찰은 [진행 기록](outputs/video_finetune_setup.json)을 따릅니다.

**97 epoch에서 중단된 현재 실행의 복구:** 2026-11-21 사용자가 재연결 후에도 학습이 멈춰 있음을 확인했고, GPU 사용 한도로 재할당도 불가능하다고 알렸습니다. 현재 복구 학습은 시작하지 않았습니다. GPU를 다시 사용할 수 있게 되면 [97 epoch 복구 노트북](https://colab.research.google.com/github/jeongjunseo/yolov8-stationery-tracking/blob/main/notebooks/resume_video_finetune_colab.ipynb)에서 **98–100 epoch만** 이어 학습할 수 있습니다. 원래 97행의 CSV·이전 best·optimizer 상태를 보존하고 새 Drive 폴더에 기록합니다. 원래 백업이 갱신되거나 같은 런타임의 학습 프로세스가 살아 있으면 재개를 거부합니다. 다른 런타임의 실행 여부까지 자동으로 확인하지는 못하므로 기존 실행 상태도 먼저 확인해야 합니다. 준비된 복구 절차이며 실제 재개·완료를 뜻하지 않습니다. 현재 한도 소진이 최초 중단 원인이었다고 단정하지 않습니다.

[보완 학습 Colab 노트북](https://colab.research.google.com/github/jeongjunseo/yolov8-stationery-tracking/blob/main/notebooks/finetune_video_colab.ipynb)은 **처음부터 별도 실험을 재현할 때** 사용합니다. 중단된 현재 실행의 재개용이 아니므로 이를 다시 모두 실행하지 않습니다. 두 노트북 모두 데이터 SHA-256을 검사하고, 매 epoch 체크포인트 백업·원래 test 평가·추가 프레임의 학습 전후 재검사·결과 ZIP 저장을 수행합니다.

GPU 대기 중 저장된 epoch 82 best로 [예비 5초 영상](outputs/video_finetune/interim_best82_5s.mp4)과 [궤적 CSV](outputs/video_finetune/interim_best82_trajectory.csv)를 CPU에서 생성했습니다. 150프레임에 세 클래스가 각각 같은 ID로 유지돼 CSV는 450행입니다. 20개 파일·시간 검사를 통과했고, 정답이 있는 25프레임의 클래스별 중심 75점이 모두 대응 정답 박스 안에 있었습니다([검증 기록](outputs/video_finetune/interim_best82_preview.json), [화면 미리보기](report/assets/video_finetune/interim_best82_preview.jpg)). 이는 **97/100 epoch 상태의 학습 장면 예비 결과**이며, 최종 test 성능이나 독립 영상 정확도를 뜻하지 않습니다. 기존 기준 영상은 보존했습니다.

2026-11-21 사용자는 **100 epoch 조건 유지**를 선택했습니다. GPU 재할당 후 남은 3 epoch를 재개하며, 그 뒤 최종 test 평가와 결과 영상·보고서를 마무리합니다. 97 epoch로 실험을 종료하지 않습니다.

이후 사용자의 즉시 영상 검증 요청에 따라 **원본 전체를 새로 추론**했습니다. 48개 학습 라벨 프레임·144개 정답을 반환된 추적 박스와 클래스별 IoU≥0.5로 대조한 결과 TP144·FP1·FN0이었습니다. 그러나 전체 영상에서는 형광펜이 화면 밖/경계에 걸린 212–215프레임 및 다시 보이는 219프레임에 추적 출력이 없었고, ID가 3→10→11로 바뀌었습니다. 276·277·279프레임(약 9.2–9.3초)에는 화면 아래 바닥을 수정테이프 ID12로 오탐했습니다. 48개 표본에 없는 프레임의 현상까지 표본 recall 100%로 덮어 해석하지 않습니다. [전체 궤적](outputs/video_finetune/recheck_full_trajectory.csv), [라벨 프레임 대조](outputs/video_finetune/recheck_labeled_frames.json), [ID 변경 장면](report/assets/video_finetune/recheck_id_gap.jpg), [오탐 장면](report/assets/video_finetune/recheck_background_fp.jpg)을 함께 공개합니다.

추가 자료는 [데이터 검증 기록](outputs/video_finetune_dataset.json), [프레임별 정답 좌표와 작성 방법](outputs/video_finetune_annotations.json), [라벨 미리보기](report/assets/video_finetune/annotation_preview.jpg)에 기록했습니다. 실제 학습 이미지는 Git에서 제외된 별도 폴더와 개인 Drive의 `yolov8-stationery-tracking/video_finetune/video_training_addition_v2.zip`에 보존합니다. v2는 화면 끝에 걸친 박스의 소수 반올림 오류를 바로잡은 검증본이며, 이전 ZIP도 보존했습니다.

영상 전체가 새 학습 자료이므로, 보완 모델로 같은 영상을 다시 처리한 결과는 **학습 장면 재검사**입니다. 기존 test 성능 비교와 별도로 보고하며 새로운 영상에 대한 일반화 성능으로 해석하지 않습니다.

## Colab에서 실행

[Colab에서 노트북 열기](https://colab.research.google.com/github/jeongjunseo/yolov8-stationery-tracking/blob/main/notebooks/train_colab.ipynb)

1. GPU 런타임을 연결하고 노트북의 셀을 순서대로 실행합니다. `nvidia-smi`와 PyTorch CUDA 검사를 통과해야 학습을 시작합니다.
2. Google Drive 연결을 승인합니다. 기본 데이터 위치는 `MyDrive/yolov8-stationery-tracking/`이며, 전체 ZIP 또는 `stationery_dataset.parts.json`에 나열된 분할 파일을 읽습니다. 분할 파일은 `/content/stationery_dataset.zip`으로 재조립하고 각 조각·전체 ZIP·이미지·라벨의 SHA-256을 검사합니다.
3. 기존 `scripts/train.py`를 YOLOv8n, batch 16, epochs 100, imgsz 640, seed 42로 실행합니다. 실제 CUDA 메모리 부족이 발생할 때만 batch를 8 또는 4로 낮춥니다. 콜백으로 체크포인트를 백업하고 라이브러리의 내부 batch 자동 변경을 차단하여 이 조건을 지킵니다. 모델·증강·최적화·시드 설정은 그대로 유지합니다.
4. 학습 완료 후 test 평가와 고정 임계값 실패 분석을 실행합니다. 원본 영상이 없으면 영상 단계는 보류하고 모델·성능·그래프를 먼저 저장합니다.
5. 체크포인트와 결과 ZIP은 Drive의 `yolov8-stationery-tracking/colab_runs/<run-id>/`에 저장합니다. 측정된 결과를 검토한 뒤 README와 보고서를 완성하고 GitHub에 반영합니다.

Colab은 로컬 `.venv`나 Windows junction을 사용하지 않습니다. 노트북은 실제 GPU 장치와 완료 epoch 수를 확인하며, CPU 중간 체크포인트를 완료된 Colab 학습 결과로 대체하지 않습니다.

## 1. 데이터 기준

- 프로젝트 자체 검사 기준: 클래스별 100장 이상, 총 300장 이상
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

이미지 687장 디코딩과 기존 검사 스크립트의 좌표·완전 중복 검사는 통과했습니다. 다만 [빈 라벨 이미지 연락시트](outputs/dataset_audit/empty_labels_contact_sheet.jpg)의 02번에는 형광펜이 보이는데 라벨은 비어 있어 추가 검토가 필요합니다. 검사 통과가 라벨 내용의 완전성을 보장하지는 않습니다.

추가 [유사 장면 감사](outputs/dataset_audit/near_duplicate_audit.json)에서는 split이 다른 이미지 조합 102,590쌍을 비교했습니다. 바이트가 완전히 같은 쌍은 0개였지만, 이미지 해시와 색상 차이 기준을 만족한 유사 후보 494쌍이 모두 eraser에서 나왔습니다(train–valid 239쌍, train–test 210쌍, valid–test 45쌍). 후보에 포함된 고유 이미지는 train 136장·valid 54장·test 30장이며, test 29장은 train 이미지와 후보 관계에 있었습니다.

가장 유사한 후보 중 이미지가 서로 겹치지 않는 12쌍을 [연락시트 1](outputs/dataset_audit/near_duplicate_candidates_01.jpg)·[연락시트 2](outputs/dataset_audit/near_duplicate_candidates_02.jpg)에서 검토한 결과, 거의 같은 배경·물체 배치·시점이 split을 넘나드는 사례를 확인했습니다. 다만 [임계값 근처 6쌍](outputs/dataset_audit/near_duplicate_threshold/near_duplicate_candidates_01.jpg)에는 회전·시점 차이가 분명한 사례도 있어 **494쌍 전체를 확정 중복으로 해석하지 않습니다.** 상세 해시는 [이미지별 기록](outputs/dataset_audit/near_duplicate_image_hashes.json)에 있습니다. 이 결과는 특히 eraser의 validation·test 성능을 새로운 장면에 대한 일반화 성능으로 해석할 때의 한계입니다. 영향의 크기는 아직 측정하지 않았습니다.

이번 학습은 기존 분할·라벨·설정을 유지한 기준 실험으로 완료했으며, 이미지·라벨 1,374개 파일의 SHA-256이 원본 ZIP 매니페스트와 일치함을 확인했습니다. 개별 시각 판정과 원본 보존 검사는 [시각 검토 기록](outputs/dataset_audit/near_duplicate_visual_review.json)에 남겼습니다. 후속 실험에서는 촬영 세션·장면별로 이미지를 묶어 같은 그룹을 하나의 split에만 배치하고, 누락 라벨을 검토한 뒤 동일 학습 설정으로 결과를 비교하는 것이 필요합니다.

## 2. 설치

로컬 Windows 환경을 새로 구성할 때는 PowerShell에서 프로젝트 루트로 이동한 뒤 실행합니다. 현재 Colab 실행은 위 노트북의 설치 절차를 사용합니다.

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

GPU 실행은 `torch.cuda.is_available()`이 `True`여야 합니다. 실제 Colab에서는 Tesla T4, Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, Ultralytics 8.4.215와 CUDA 인식 `True`를 확인했습니다. Colab의 `pip check`는 기존 IPython의 `jedi` 누락을 경고했으며, 이후 핵심 패키지 import와 CUDA 검사는 통과했습니다. 로컬 CPU 환경의 설치·검사 이력은 [보고서](report/REPORT.md)에 별도로 기록합니다.

## 3. 데이터 검사

라벨 범위, 이미지-라벨 대응, split별 수량, 클래스별 포함 이미지 수, split 사이의 완전 중복 이미지를 검사합니다.

```powershell
python scripts/check_dataset.py
```

위 수량 기준은 기존 프로젝트 검사 설정이며, 제공된 교수님 자료에 명시된 최소 수량으로 단정하지 않습니다. 현재 데이터는 기본 검사 조건을 유지해 확인했습니다.

## 4. 학습

YOLOv8 nano 사전 학습 가중치로 100 epoch 전이학습을 수행합니다. 현재 Colab GPU 실행의 기본 명령은 다음과 같습니다.

```powershell
python scripts/train.py --device 0 --batch 16 --epochs 100
```

설정은 YOLOv8n, imgsz 640, workers 인자 4, seed 42, deterministic=True, optimizer=auto이며 [실제 args.yaml](report/assets/train/args.yaml)에 저장했습니다. [학습 완료 기록](outputs/colab_training_run.json)과 [100행 CSV](report/assets/train/results.csv)로 실제 batch 16과 100 epoch 완료를 검증했습니다. 실제 CUDA 메모리 부족이 발생할 때만 batch를 16에서 8, 4 순서로 낮추는 조건이지만 이번 GPU 실행에는 필요하지 않았습니다. 이전 로컬 CPU 시도는 batch 16·8의 메모리 오류 후 batch 4로 1 epoch를 완료했고, 사용자 요청으로 일시정지되어 있습니다.

학습 결과는 `runs/train/` 아래에 저장되고 완료 후 최적 가중치는 자동으로 `models/best.pt`에 복사됩니다. Colab은 해당 런타임의 로컬 디스크를 사용하고 Drive에 체크포인트를 백업합니다. 이전 Windows 실행의 `runs/`는 D:에 연결되어 있으며 그대로 보존합니다. 보고서에는 다음 자료를 사용합니다.

- [results.csv](report/assets/train/results.csv): epoch별 loss와 validation 성능
- [results.png](report/assets/train/results.png): 학습 곡선
- [test 혼동 행렬](report/assets/test/confusion_matrix.png), [test PR 곡선](report/assets/test/BoxPR_curve.png)
- [models/best.pt](models/best.pt): 완료된 실행에서 선택된 학습 모델

![100 epoch 학습 손실 및 validation 성능](report/assets/train/results.png)

## 5. test 데이터 평가

```powershell
python scripts/evaluate.py --device 0
```

전체 Precision, Recall, mAP50, mAP75, mAP50-95와 클래스별 Precision, Recall, mAP50, mAP50-95, 정답 객체 수가 `outputs/test_metrics.json`에 저장됩니다. 평가되지 않은 클래스의 수치는 `null`로 기록합니다. 평가 그래프는 `runs/val/`에 저장됩니다.

완료된 GPU 평가의 실측값은 다음과 같습니다. 모든 성능 수치는 %이며, 전체 행은 클래스 평균입니다.

| 클래스 | 정답 객체 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| 전체 | 108 | 93.70 | 88.77 | 93.58 | 76.68 |
| highlighter | 23 | 89.07 | 91.30 | 93.68 | 87.09 |
| correction_tape | 44 | 93.06 | 75.00 | 87.56 | 48.35 |
| eraser | 41 | 98.98 | 100.00 | 99.50 | 94.60 |

전체 mAP75는 80.10%입니다. 수정테이프는 여러 객체를 묶거나 물체 일부만 잡는 위치 오류가 관찰됐고, 엄격한 IoU 기준에서 성능 하락이 컸습니다. 지우개의 높은 성능은 위 유사 장면 문제 때문에 새로운 장면에 대한 일반화 성능으로 그대로 해석할 수 없습니다. [보고서](report/REPORT.md)에 실제 실패 그림과 클래스별 분석을 정리했습니다.

[고정 임계값 진단](outputs/failure_analysis.json)은 confidence 0.25, NMS IoU 0.5, 정답 매칭 IoU 0.5에서 TP 95 / FP 10 / FN 13을 기록했습니다. 이 진단의 전체 precision·recall은 객체 수를 합산한 값이므로 위 공식 평가의 클래스 평균·신뢰도 곡선 기반 값과 구분합니다.

## 6. 5초 추적 영상 제작

사용자가 업로드한 `1000030242.mp4`는 9.566089초·287프레임의 HEVC 10-bit HDR(PQ/BT.2020) 영상이며, 회전 메타데이터를 적용한 화면은 1080 × 1920입니다. 원본 26,398,062 bytes를 보존하고 별도 SDR(BT.709) 입력을 만든 뒤 앞 5초를 처리했습니다. 원본은 프로젝트 상위 폴더에 있으며 공개 저장소에 포함하지 않습니다.

아래는 프로젝트 루트에서 실행하는 재현 예시입니다. [FFmpeg 공식 다운로드 페이지](https://ffmpeg.org/download.html)의 Windows 빌드 링크를 통해 확보한 FFmpeg 9.0.2 gyan 빌드를 사용했습니다. `ffmpeg`가 PATH에 없으면 해당 실행 파일의 경로로 바꿉니다. `zscale`, `tonemap`, `libx264`가 필요하며 HDR 변환 방법은 [FFmpeg tonemap 문서](https://ffmpeg.org/ffmpeg-filters.html#tonemap)를 참고합니다.

```powershell
ffmpeg -n -i ../1000030242.mp4 -map 0:v:0 -an -map_metadata -1 -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p -fps_mode passthrough -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart inputs/source_sdr.mp4
python scripts/track_video.py --source inputs/source_sdr.mp4 --model models/best.pt --device cpu --imgsz 1280 --rotate-inference 90 --agnostic-nms --conf 0.25 --iou 0.5 --trail-length 60 --tracker bytetrack.yaml --duration 5 --output outputs/final_5s_render.mp4 --trajectory outputs/trajectory.csv
ffmpeg -n -i outputs/final_5s_render.mp4 -map 0:v:0 -an -map_metadata -1 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -fps_mode passthrough -movflags +faststart outputs/final_5s.mp4
```

추론 입력만 시계방향 90도로 회전하며 예측 박스·중심 좌표는 원래 세로 영상 좌표로 복원합니다. `--agnostic-nms`는 클래스가 달라도 겹치는 검출 박스를 억제합니다. ByteTrack은 `persist=True`로 ID를 연결하고 최근 60개 중심 좌표를 궤적으로 표시하며, 별도 재식별(Re-ID) 모델을 사용하지 않습니다. 예측 클래스나 좌표를 수동으로 정답에 맞추지 않았습니다.

HDR/SDR, 입력 크기 640/1280, 추론 회전 0/90도를 같은 영상의 3개 프레임에서 진단한 뒤 전체 5초 실행들을 비교해 위 설정을 선택했습니다. 표본 진단의 confidence 하한은 0.01, 전체 5초 실행의 confidence는 0.25였으며 [탐색 기록](outputs/tracking_experiments.json)에 구분했습니다. 이는 이 영상에 대한 탐색이며 학습 설정이나 공식 test 평가를 변경한 것이 아닙니다. Class-agnostic NMS 적용 전의 14개 ID·410행은 적용 후 2개 ID·300행으로 줄었지만, 분류 오류는 남았습니다.

| 추적 ID | 실제 관찰 대상 | CSV 예측 클래스와 행 수 | 한계 |
|---:|---|---|---|
| 1 | 지우개 | eraser 82 / highlighter 68 | 같은 물체의 클래스가 16회 바뀜 |
| 2 | 수정테이프 | correction_tape 150 | 150프레임에 같은 ID 기록 |
| 없음 | 위쪽 노란 형광펜 | 검출 없음 | 추적 실패 |

CSV의 세 클래스 이름은 **실제 물체 3개를 모두 추적했다는 뜻이 아닙니다.** 형광펜 예측 68행은 지우개에 붙은 오분류입니다. ID 1·2는 각각 150프레임에 기록됐으며, 정식 IDF1·MOTA 등 추적 정확도는 측정하지 않았습니다. 물체가 대체로 놓여 있는 상태에서 카메라가 움직이므로 노란 궤적은 화면 좌표의 변화이며 실제 공간에서 물체가 이동한 경로는 아닙니다.

[최종 영상 다운로드](outputs/final_5s.mp4) · [궤적 CSV](outputs/trajectory.csv)

![실제 영상의 추적 결과 미리보기](report/assets/tracking/preview.jpg)

영상은 H.264/yuv420p, 1080 × 1920, 30 FPS, 150프레임이며 컨테이너 길이는 정확히 5초이고 오디오는 없습니다. CSV는 300행, 프레임 0–149, 출력 시간 0–4.9667초입니다. 마지막 프레임의 시작 시각이 149/30초이고 그 프레임까지 재생하면 5초가 됩니다. 전체 디코딩·프레임 시간·CSV 범위·파일 보존 등 20개 검사가 통과했으며 노란 궤적과 CSV 이력도 대조했습니다. [tracking_verification.json](outputs/tracking_verification.json)은 파일·시간·렌더링 검증 통과와 세 물체 추적 목표 미달성을 별도로 기록합니다.

출력 프레임 수를 `round(원본 FPS × 5)`로 정하고 출력 FPS를 `프레임 수 / 5`로 정규화합니다. 소수 FPS 입력에서는 재생 속도가 조금 달라질 수 있으며 CSV 시간은 출력 영상 기준입니다. OpenCV의 `mp4v` 중간 파일을 보존하고, [GitHub에서 권장하는 H.264](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)로 최종 파일을 따로 만듭니다. FFmpeg의 `-n`과 추적 스크립트는 기존 출력을 덮어쓰지 않습니다. 이미 최종 영상·CSV가 있는 저장소에서 재실행할 때에는 중간·최종 영상 및 `--trajectory`에 새 경로를 지정합니다.

## 7. 보고서와 GitHub 제출

[report/REPORT.md](report/REPORT.md)에 Colab 학습·test 성능·실패 사례와 실제 영상의 추적 결과를 기록했습니다. 기준 실험의 모델·지표·영상·CSV는 생성했으나 기준 영상에서 세 물체를 모두 정확히 감지·추적하는 목표는 미달성입니다. 이후 별도 보완 학습과 현재 모델의 영상 재검증을 수행했으며, 그 완료 범위는 위 보완 학습 절과 보고서 9–10절에 구분했습니다. 기준 실험의 제출 경로는 다음과 같습니다.

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

`runs/`는 Git에서 제외되므로 실제 생성된 CSV·설정·학습 곡선은 [report/assets/train/](report/assets/train/), test 그래프는 [report/assets/test/](report/assets/test/), 실패 그림은 [report/assets/failures/](report/assets/failures/)로 복사했습니다. 기존 42개 학습·평가·실패 그림의 무결성과 모델 로딩을 확인했습니다. 2026-11-21 제출 점검에서 Git 추적 파일 129개가 모두 100 MiB 미만임을 확인했으며, 가장 큰 파일은 17,522,497 bytes의 `outputs/video_finetune/recheck_full.mp4`입니다. 기준 모델·평가는 [artifact_verification.json](outputs/artifact_verification.json), 기준 영상·CSV와 추적 한계는 [tracking_verification.json](outputs/tracking_verification.json), 보완 영상은 [video_recheck.json](outputs/video_finetune/video_recheck.json)을 함께 확인합니다.

코드 회귀 검증은 `python -m unittest discover -s tests -v`로 실행합니다. 이 테스트의 영상은 임시 합성 데이터와 모의 검출 결과를 사용하며 실제 학습·검출 성능이나 ByteTrack의 ID 유지 성능을 입증하지 않습니다.

