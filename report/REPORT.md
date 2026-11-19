# YOLOv8 문구류 객체 감지 및 추적 — 학습·평가 결과보고서

> 2026-11-19 19:28 KST: 사용자 요청으로 로컬 CPU 학습을 일시정지했다. 메모리와 체크포인트를 보존했으며 이후에도 재개하지 않았다.

> 이후 사용자는 교수님 자료를 별도 기록하고 Colab에서 학습하도록 지시했다. [과제 지침 메모](../docs/ASSIGNMENT_GUIDELINES.md)와 [Colab 실행 노트북](../notebooks/train_colab.ipynb)을 사용하며, Colab의 실제 환경·데이터 검증 결과와 이전 CPU 시도를 구분해 기록한다.

기록일: 2026-11-20 (Asia/Seoul)

저장소: https://github.com/jeongjunseo/yolov8-stationery-tracking

**현재 상태: Colab Tesla T4의 CUDA 장치 0에서 batch 16으로 100 epoch 학습과 test 평가를 완료했다. Test 73장·108개 정답 객체에서 Precision 93.70%, Recall 88.77%, mAP50 93.58%, mAP50-95 76.68%를 얻었다. 모델 해시·CPU 로딩과 실제 그래프도 검증했다. 원본 영상이 없어 5초 추적 영상과 궤적 CSV는 미생성이며, 전체 과제는 아직 완료되지 않았다. 로컬 CPU 학습은 사용자 요청에 따른 일시정지 상태를 유지한다.**

## 1. 목적과 현재 결과

YOLOv8n으로 형광펜(`highlighter`, ID 0), 수정테이프(`correction_tape`, ID 1), 지우개(`eraser`, ID 2)를 학습하고, 별도 test split에서 성능을 평가한 뒤 ByteTrack으로 클래스명·추적 ID·이동 궤적이 표시되는 5초 영상을 만드는 것이 목표이다.

이전 컴퓨터의 `HANDOFF.md`를 기준으로 작업을 재개했다. 저장소를 복제하고 제공된 `data.zip`에서 기존 분할 데이터를 복원했다. 데이터와 라벨을 새로 만들거나 재분할하지 않았다. 로컬 PC에서는 NVIDIA GPU가 확인되지 않아 사용자의 승인으로 CPU 학습을 시도했으며, 이후 일시정지하고 사용자 지시에 따라 Colab GPU로 실행 환경을 전환했다. 이번 보고서는 완료된 학습·평가와 실제 실패 사례를 다루며, 입력 영상 제공 후 추적 결과를 추가한다.

## 2. 데이터 출처와 전처리

직접 촬영한 데이터라고 주장하지 않는다. [data/SOURCES.md](../data/SOURCES.md)에 기록된 Roboflow Universe의 세 공개 데이터셋을 사용하며 원본은 CC BY 4.0으로 명시되어 있다. 직접 촬영 기기·기간·라벨링 경험은 제공된 근거가 없어 기재하지 않았다.

| 클래스 | 출처 | 원본 이미지 | 기존 처리 |
|---|---|---:|---|
| highlighter | [Highlighter](https://universe.roboflow.com/visioncoprocessor/highlighter-g3fru) | 240 | 폴리곤 라벨을 검출 박스로 변환 |
| correction_tape | [Assignment_Correction tape](https://universe.roboflow.com/kyonggiuniv/assignment_correction-tape) | 129 | 경계를 조금 벗어난 박스 6개를 이전 작업에서 보정 |
| eraser | [eraser](https://universe.roboflow.com/reder/eraser-xnm4s) | 318 | 세 색상 클래스를 eraser로 통합 |

원본 파일명에는 데이터셋별 접두어가 붙어 있다. 위 전처리는 이전 인계 문서에 기록된 내용이며 이번 실행에서 반복하지 않았다.

## 3. 복원 및 데이터 검사

| 구분 | train | valid | test | 합계 |
|---|---:|---:|---:|---:|
| 이미지 수 | 498 | 116 | 73 | 687 |
| 이미지 비율 | 72.49% | 16.89% | 10.63% | 100% |
| 라벨 파일 수 | 498 | 116 | 73 | 687 |
| 객체 박스 수 | 614 | 146 | 108 | 868 |
| 빈 라벨 파일 수 | 15 | 4 | 2 | 21 |

비율은 소수 둘째 자리 반올림 때문에 합계가 100.01%로 보일 수 있다.

다음 표의 각 셀은 **클래스 포함 이미지 수 / 객체 박스 수**이다.

| 클래스 | train | valid | test | 전체 |
|---|---:|---:|---:|---:|
| highlighter | 177 / 180 | 21 / 21 | 21 / 23 | 219 / 224 |
| correction_tape | 89 / 163 | 26 / 52 | 14 / 44 | 129 / 259 |
| eraser | 217 / 271 | 65 / 73 | 36 / 41 | 318 / 385 |

`python scripts/check_dataset.py`는 정상 종료했다. 기존 스크립트가 검사하는 이미지·라벨 대응, 좌표 범위, 클래스 구성, split 간 완전 중복 검사에 통과했다. 별도 Pillow 검사로 이미지 687장 전체를 실제 디코딩했고 오류는 0건이었다. 대응 라벨 파일 누락도 0건이었다. 상세 집계는 [dataset_summary.json](../outputs/dataset_summary.json)에 저장했다.

Colab에서는 2,451,554,279 bytes의 기존 데이터 ZIP을 25개 조각에서 복원했다. 조각 목록과 전체 SHA-256을 검증했고, 이미지·라벨 1,374개와 메타데이터 2개를 포함한 1,376개 파일의 매니페스트 검사가 통과했다. 두 메타데이터 파일은 Windows/Linux 개행 차이만 동등하게 인정해 저장소의 파일을 유지했다. Colab에서 다시 실행한 데이터 검사도 train 498장, valid 116장, test 73장으로 통과했다. 근거는 [colab_dataset_check.json](../outputs/colab_dataset_check.json)이다.

### 빈 라벨과 남은 데이터 품질 문제

highlighter 원본 240장 중 21장의 라벨 파일이 비어 있어 해당 클래스의 라벨 포함 이미지는 219장이다. 빈 라벨 파일이 있다는 사실만으로 올바른 배경 이미지라고 판단할 수 없다.

![빈 라벨 21장 감사 연락시트](../outputs/dataset_audit/empty_labels_contact_sheet.jpg)

연락시트 02번 train 이미지 `highlighter_IMG_20260713_162103695_HDR_jpg.rf.Sbrd0MiyGuUjja7V5nXf.jpg`에는 왼쪽 가장자리에 녹색 뚜껑의 형광펜이 보이는데 라벨은 비어 있다. 최소 이 사례는 누락 객체 라벨 검토 대상이다. 이번 실행은 기존 데이터 보존 원칙에 따라 원본 라벨을 그대로 사용하며, 이 문제를 결과 해석의 한계로 기록한다. 이 사례는 학습 모델의 미검출 사례가 아니라 **정답 데이터의 문제**이다.

나머지 빈 라벨도 썸네일 검토만 했으므로 원해상도에서의 정답 완전성을 보증하지 않는다.

### Split 간 유사 장면 검사와 평가의 한계

학습 중 기존 파일을 변경하지 않고 이미지 687장의 split 간 조합 102,590쌍을 추가로 비교했다. SHA-256이 같은 완전 중복 쌍은 0개였다. 유사 후보는 64비트 pHash 해밍 거리 ≤ 6, 128비트 dHash 거리 ≤ 12, 32×32 RGB 평균 절대 차이 ≤ 12라는 세 조건을 모두 만족하는 쌍으로 선별했다. 결과와 방법은 [near_duplicate_audit.json](../outputs/dataset_audit/near_duplicate_audit.json), 이미지별 해시는 [near_duplicate_image_hashes.json](../outputs/dataset_audit/near_duplicate_image_hashes.json)에 기록했다.

| 서로 다른 split | 자동 선별된 유사 후보 쌍 |
|---|---:|
| train–valid | 239 |
| train–test | 210 |
| valid–test | 45 |
| 합계 | 494 |

494쌍은 모두 eraser 이미지 쌍이었다. 이 후보에 포함된 고유 이미지는 train 136장, valid 54장, test 30장이며, test 29장은 train 이미지와 후보 관계에 있었다. 한 이미지가 여러 후보 쌍에 참여할 수 있으므로 쌍의 수와 이미지 수는 다르다. 다른 클래스에서 후보가 없었다는 결과만으로 그 클래스의 장면 독립성이 입증되지는 않는다.

유사도가 높은 순서에서 같은 이미지를 재사용하지 않고 선택한 12쌍(train 12장·valid 6장·test 6장)을 [연락시트 1](../outputs/dataset_audit/near_duplicate_candidates_01.jpg)과 [연락시트 2](../outputs/dataset_audit/near_duplicate_candidates_02.jpg)로 시각 검토했다. 이 표본에서는 배경·물체 배치·시점이 거의 같은 장면이 서로 다른 split에 분포하는 것을 확인했다. 유사도순으로 선택했기 때문에 이 검토 결과를 전체 494쌍의 중복 비율로 확대해석하지 않는다.

별도로 [임계값 근처 6쌍](../outputs/dataset_audit/near_duplicate_threshold/near_duplicate_candidates_01.jpg)을 검토했을 때는 같은 배경에서도 물체 회전·시점 변화가 뚜렷한 사례가 있었다. 따라서 **494쌍은 자동 선별 후보 수이며 확정 중복 수가 아니다.** 해시 기반 방법은 단순한 배경 때문에 유사도를 높게 평가하거나, 반대로 잘림·회전·시점 변화가 있는 같은 장면을 놓칠 수 있다. 파일명 시각도 실제 촬영 메타데이터로 인증된 것은 아니다.

18쌍의 개별 시각 판정과 원본 보존 검사는 [near_duplicate_visual_review.json](../outputs/dataset_audit/near_duplicate_visual_review.json)에 기록했다. 이미지·라벨 1,374개 파일의 SHA-256이 원본 ZIP 매니페스트와 모두 일치했으며, 이번 학습의 데이터·라벨·설정은 변경하지 않았다.

이 발견은 특히 eraser의 validation·test 성능이 새로운 장면에서의 성능보다 낙관적으로 나타날 수 있음을 의미한다. 그 영향의 크기나 성능 순위에 미친 원인은 아직 측정하지 않았으므로 확정하지 않는다. 현재 실행은 기존 분할을 유지한 기준 실험으로 보고하며, 이 한계를 성능과 함께 제시한다.

후속 실험에서는 촬영 세션·장면 단위로 이미지를 묶고 같은 그룹을 하나의 split에만 배치해야 한다. 누락 라벨도 검토한 뒤 동일한 학습 설정으로 별도 실험을 수행해 현재 기준 실험과 비교하는 것이 필요하다. 이번 기준 실험에서는 분할을 바꾸거나 데이터를 삭제하지 않았다.

## 4. 실행 환경과 학습 설정

### 현재 Colab 환경과 실행 조건

| 항목 | 확인 결과 또는 실행 설정 |
|---|---|
| 실행 환경 / Python | Google Colab / 3.13.16 |
| GPU | Tesla T4, 런타임 보고 메모리 15,637,086,208 bytes |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| CUDA 런타임 / 사용 가능 | 13.0 / True |
| Ultralytics / OpenCV / PyYAML | 8.4.215 / 4.14.0.94 / 6.0.3 |
| 모델 / epochs / imgsz | YOLOv8n / 100 / 640 |
| 요청 batch / 실제 batch / device | 16 / 16 / cuda:0 |
| workers 인자 / seed / deterministic | 4 / 42 / True |
| optimizer 설정 / AMP / cache | auto / True / False |
| 완료 epoch / 학습 CSV 누적 시간 | 100 / 9,834.78초 (2시간 43분 54.78초) |

실제 환경은 [colab_environment.json](../outputs/colab_environment.json), 실행 설정은 [args.yaml](assets/train/args.yaml)에 저장했다. 기존 `scripts/train.py`의 설정을 유지했다. [학습 완료 기록](../outputs/colab_training_run.json)과 [시도 기록](../outputs/colab_training_attempts.json)은 batch 16의 단일 시도 성공과 CUDA 사용, 100 epoch 완료를 확인한다. GPU 메모리 부족이나 batch 축소는 없었고, [100회 백업 기록](../outputs/colab_backup_events.json)에 실패가 없다. 시간은 학습 CSV에 기록된 누적 값이며 환경 설치·데이터 전송·후속 평가 시간은 포함하지 않는다.

```text
python scripts/train.py --device 0 --batch 16 --epochs 100
```

첫 Drive 연결 시도는 `ValueError: mount failed`로 끝났으나 사용자가 정상 연결 절차로 다시 실행한 뒤 `Mounted at /content/drive`와 백업 폴더 생성을 확인했다. 최초 오류의 세부 원인은 확정하지 않았다. Colab의 `pip check`는 기존 IPython 7.34.0에서 `jedi`가 누락됐다는 경고를 출력했다. 이후 핵심 패키지 import와 CUDA 확인은 통과했으며, 전체 의존성 검사가 모두 통과했다고 기록하지 않는다.

### 학습 곡선과 최종 모델 검증

![100 epoch의 학습 손실과 validation 성능](assets/train/results.png)

[results.csv](assets/train/results.csv)는 epoch 1부터 100까지 100행을 포함한다. 학습 box loss는 0.94820에서 0.37691로, validation box loss는 0.88072에서 0.53754로 감소했다. Validation mAP50-95는 첫 epoch 0.56468에서 마지막 epoch 0.82376으로 상승했으며, CSV 최고값은 epoch 88의 0.82416이다. 이 값은 validation 결과이며 아래 test 결과와 구분한다. CSV 최고값만으로 저장된 모델의 선택 epoch를 단정하지 않는다.

[models/best.pt](../models/best.pt)는 6,500,778 bytes, 파라미터 3,138,489개이며 SHA-256은 다음과 같다.

```text
72eec58f57a0decd694a98606e9759824cb7cb4d75af345e9ae8db735a4d66ea
```

해시는 Colab 완료 기록과 일치했다. 이 PC의 새 가상환경에서 모델을 CPU로 불러와 클래스 ID 0·1·2와 이름을 확인했고, 읽기 전후 파일 해시가 같았다. 학습·평가·실패 분석 그림 42개의 이미지 무결성 검사도 통과했다. [artifact_verification.json](../outputs/artifact_verification.json)에 ZIP 검증, 모델 로딩, 100행 CSV, 지표 범위 및 파일 크기 검사 결과를 남겼다. 이는 저장된 결과 검증이며 로컬에서 학습이나 test 평가를 다시 실행한 결과는 아니다.

### 이전 로컬 CPU 시도 — 사용자 요청으로 일시정지

| 항목 | 확인 결과 |
|---|---|
| 실행 환경 | Windows, 새 Python 3.14.0 가상환경 |
| CPU | AMD Ryzen 5 5500GT, 6코어 / 12스레드 |
| 그래픽 | AMD Radeon(TM) Graphics, Parsec 가상 디스플레이 |
| NVIDIA GPU / nvidia-smi | 장치 확인 안 됨 / 명령 및 표준 경로에 없음 |
| 시스템 PyTorch | 2.9.0+cu126, CUDA 12.6 빌드 |
| 시스템 CUDA 인식 | False, 장치 수 0 |
| 새 가상환경 PyTorch / Ultralytics | 2.15.0+cpu / 8.4.215 |
| 새 가상환경 torchvision / OpenCV | 0.29.2 / 4.14.0.94 |
| 새 가상환경 CUDA 인식 | False, 장치 수 0 |
| 가상환경 위치 | 프로젝트 `.venv`에서 D:의 새 환경으로 연결 |

이전 PC의 `.venv`를 사용하지 않았다. 이번 새 환경의 첫 의존성 설치는 C: 공간 부족으로 실패했고, 해당 불완전 환경을 D:에 보존한 뒤 D:에 새 환경을 구성했다. 기존 사용자 파일을 삭제하지 않았다. 시스템 CUDA 빌드가 설치되어 있어도 NVIDIA 하드웨어가 없어 CUDA를 사용할 수 없다는 점을 확인했다.

로컬의 새 환경에서는 `requirements.txt` 설치와 `pip check`가 성공했다. 사용자가 CPU 학습을 허용한 후 이 CPU 빌드로 학습을 실행했다. [로컬 환경 확인 JSON](../outputs/environment_check.json)에 실제 버전과 CUDA 상태를 저장했다. 이 검사 결과와 아래 CPU 설정은 위 Colab 환경의 결과와 구분한다.

최초 CPU 실행은 기존 `scripts/train.py`에서 장치만 `cpu`로 지정했다. 실제 메모리 부족 발생에 따라 batch만 16에서 8, 다시 4로 낮췄다.

| 항목 | 설정 |
|---|---|
| 모델 | COCO 사전 학습 YOLOv8n (`yolov8n.pt`) |
| epochs / imgsz | 100 / 640 |
| batch | 16과 8에서 CPU 메모리 부족 발생 후 4로 재시도 |
| workers | 스크립트 인자 4 / Ultralytics CPU 실행 실제값 0 |
| seed / deterministic | 42 / True |
| device | cpu (사용자 명시적 승인) |
| optimizer | auto → 실제 선택 AdamW |
| 실제 초기 학습률 | 0.001429 |

```powershell
python scripts/train.py --device cpu --batch 4 --epochs 100
```

학습 로그와 체크포인트는 프로젝트의 `runs/`를 통해 D:에 저장한다. 최초 실행은 2026-11-19 19:17:51 KST에 시작했으나 19:18:27 KST에 `DefaultCPUAllocator: not enough memory` 오류로 종료됐다. 오류 메시지의 할당 요청 크기는 13,107,200 bytes이며 첫 epoch 결과는 생성되지 않았다. 사용자 프로그램이나 데이터를 삭제·변경하지 않고, 메모리 부족 시 batch를 낮추는 대응을 승인된 CPU 실행에 적용했다. 실패 로그와 상태는 `runs/session_cpu_20261119/`에 보존했다.

batch 8 재시도는 19:20:06 KST에 시작했으나 첫 epoch의 11/63 배치에서 CPU 메모리 부족으로 실패했고, 19:21:02 KST에 종료됐다. 할당 요청 크기는 16,384,000 bytes였다. 이 시도에서도 완료 epoch는 0개이며, 로그와 `status.json`은 `runs/session_cpu_20261119_b8/`에 보존했다.

batch 4 재시도는 19:22:27 KST에 시작해 1 epoch를 완료했다. 2번째 epoch 44%에서 19:28 KST 사용자 요청으로 프로세스를 일시정지했고, 메모리·체크포인트·로그를 보존했다. 상태는 로컬 전용 `outputs/training_status.json`에 기록되어 있으며, 로그는 `runs/session_cpu_20261119_b4/`, 결과 디렉터리는 `runs/train/stationery-3/`이다. 이 중간 결과를 완료된 Colab 모델이나 test 성능으로 사용하지 않는다.

## 5. Test 성능과 실제 실패 사례

Colab에서 100 epoch 학습 완료 후 `python scripts/evaluate.py --device 0`으로 `models/best.pt`를 평가했다. Test split은 73장·정답 객체 108개이며, [test_metrics.json](../outputs/test_metrics.json)에 전체·클래스별 실측값을 저장했다. 아래 성능 수치는 %로 표시하고 소수 둘째 자리로 반올림했다.

| 클래스 | 정답 객체 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| 전체 (클래스 평균) | 108 | 93.70 | 88.77 | 93.58 | 76.68 |
| highlighter | 23 | 89.07 | 91.30 | 93.68 | 87.09 |
| correction_tape | 44 | 93.06 | 75.00 | 87.56 | 48.35 |
| eraser | 41 | 98.98 | 100.00 | 99.50 | 94.60 |

전체 mAP75는 **80.10%**이다. mAP50은 IoU 0.5에서의 AP를 클래스 평균한 값이며, mAP50-95는 IoU 0.50부터 0.95까지 0.05 간격의 10개 기준과 클래스를 평균한다. AP는 신뢰도에 따른 precision–recall 곡선으로 계산한다. 표의 precision·recall은 평활화한 클래스 평균 F1이 최대가 되는 공통 신뢰도 지점에서 구한 클래스 평균으로, confidence 0.25의 단일 임계값 결과가 아니다. 클래스 행은 `ap_class_index`로 대응시켜 저장했으며 세 클래스 모두 평가됐다.

### Test 그래프와 클래스별 차이

![Test precision–recall 곡선](assets/test/BoxPR_curve.png)

![Test 혼동 행렬](assets/test/confusion_matrix.png)

혼동 행렬은 가로축이 정답, 세로축이 예측이다. 이 실행의 Ultralytics 8.4.215 혼동 행렬은 confidence > 0.001의 예측을 집계한 반면, 표의 precision·recall은 평활화한 클래스 평균 F1이 최대가 되는 공통 신뢰도 지점에서 계산한 클래스 평균이므로 행렬의 단순 비율과 일치하지 않는다. 또한 공식 평가의 NMS IoU 0.7과 아래 진단의 0.5는 다르다. [F1 곡선](assets/test/BoxF1_curve.png)과 [정규화 혼동 행렬](assets/test/confusion_matrix_normalized.png)도 함께 저장했다.

- **수정테이프:** Recall 75.00%, mAP50-95 48.35%로 가장 낮다. mAP50에서 mAP50-95로 39.21%p 하락했고, 아래 그림에서는 이웃 객체를 묶은 박스·물체 일부만 잡은 박스가 확인됐다. 엄격한 IoU에서 위치 정확도가 주요 약점임을 시사한다. Train 포함 이미지가 89장으로 형광펜 177장·지우개 217장보다 적고, test는 14장에 44객체로 여러 물체가 함께 있는 장면이 많다. 이 차이는 설명 후보이며 데이터량 하나를 성능 차이의 원인으로 확정하지 않는다.
- **형광펜:** mAP50-95 87.09%이나 검은 띠를 형광펜으로 검출한 배경 오검출과, 복잡한 배경 속 작은 정답 박스를 놓친 사례가 있다. 높은 전체 점수가 모든 장면의 검출을 보장하지 않는다.
- **지우개:** mAP50-95 94.60%로 가장 높다. 그러나 3절에서 확인한 split 간 거의 같은 장면이 평가를 낙관적으로 만들 수 있다. 새로운 배경·배치에 대한 일반화 성능이나 다른 클래스보다 본질적으로 쉬운 문제라고 단정할 수 없다.

### 고정 임계값에서의 오류 진단

[failure_analysis.json](../outputs/failure_analysis.json)은 공식 AP와 별도로 confidence 0.25, NMS IoU 0.5, imgsz 640, rect=False에서 예측했다. 같은 클래스의 정답·예측 박스를 IoU ≥ 0.5 조건으로 IoU가 큰 순서대로 일대일 매칭했다. 이 기준에서 test 73장 중 10장에 하나 이상의 FP 또는 FN이 있었다.

| 클래스 | TP | FP | FN |
|---|---:|---:|---:|
| highlighter | 21 | 3 | 2 |
| correction_tape | 33 | 7 | 11 |
| eraser | 41 | 0 | 0 |
| 합계 | 95 | 10 | 13 |

객체 수를 합산한 micro precision은 90.48%, recall은 87.96%이다. 평가 임계값·전처리·집계 방식이 달라 위 공식 지표와 직접 대체할 수 없다. 미매칭 박스 사이에서 정의된 다른 클래스 중첩 쌍은 0개였지만, 이는 배경을 다른 클래스로 잘못 검출한 FP가 없다는 뜻이 아니다. Test를 이용해 임계값을 재선택하거나 현재 모델을 재학습하지 않았다.

오류가 있는 클래스별 대표 이미지와 오류 수가 많은 이미지를 우선해 총 9장을 저장하고 시각 검토했다. 이는 오류 사례 모음이며 test 전체의 무작위 표본이 아니다. 다음은 실제 관찰 결과다.

| 그림 | 관찰한 현상 | 진단 결과 |
|---|---|---|
| [01](assets/failures/failure_01.jpg) | 수정테이프 3개는 검출했으나 위·아래 검은 띠를 형광펜으로 검출 (신뢰도 0.82 / 0.87) | TP 3, FP 2 |
| [02](assets/failures/failure_02.jpg) | 손·텍스트가 있는 제품 합성 이미지에서 여러 물체와 배경을 묶은 큰 박스 | TP 1, FP 1, FN 3 |
| [03](assets/failures/failure_03.jpg) | 인접한 수정테이프를 함께 덮는 큰 박스와 누락 | TP 3, FP 1, FN 3 |
| [04](assets/failures/failure_04.jpg) | 길쭉한 수정테이프 하나를 두 부분으로 나눠 검출 | FP 2, FN 1 |
| [05](assets/failures/failure_05.jpg) | 촘촘한 제품 배열에서 여러 개를 묶은 큰 박스와 좌하단 누락 | TP 9, FP 2, FN 1 |
| [06](assets/failures/failure_06.jpg) | 포장 안 오른쪽 제품의 일부만 작은 박스로 검출해 IoU 기준 미달 | TP 1, FP 1, FN 1 |
| [07](assets/failures/failure_07.jpg) | 나뭇가지·잎 주변 작은 정답 박스 2개 모두 미검출 | FN 2 |
| [08](assets/failures/failure_08.jpg) | 손에 잡혀 일부가 가려진 오른쪽 제품 누락 | TP 3, FN 1 |
| [09](assets/failures/failure_09.jpg) | 화면 왼쪽 아래 경계에서 잘린 부분 제품 누락 | TP 4, FN 1 |

![배경의 검은 띠를 형광펜으로 검출한 사례](assets/failures/failure_01.jpg)

그림 01은 배경 단서에 대한 취약성을 보여준다. 이 관찰만으로 특정 학습 이미지가 원인이라고 확정하지는 않는다.

![여러 객체와 배경을 묶은 수정테이프 박스 및 미검출](assets/failures/failure_02.jpg)

그림 02와 03·04·06은 분류 정답 여부뿐 아니라 객체별 박스 범위와 위치가 중요함을 보여준다. IoU가 기준에 못 미치는 박스는 FP와 FN으로 함께 집계될 수 있다.

![복잡한 배경의 작은 정답 박스 미검출](assets/failures/failure_07.jpg)

그림 07의 원본 해상도는 6,144 × 8,192이다. 긴 변을 640으로 축소하면 두 정답 박스는 대략 18 × 24, 47 × 28 픽셀 크기가 된다. 작은 입력상 크기와 주변 배경이 관찰되지만 각각의 원인 기여도는 별도 실험하지 않았다. 9개 실패 그림에서 추가 정답 라벨 오류를 확정하지 않았으며, 앞서 확인한 빈 train 라벨 문제와 구분한다.

## 6. 추적 구현과 영상 상태

검출 결과를 ByteTrack에 전달하고 `persist=True`로 프레임 사이 추적 상태를 유지한다. ID별 객체 중심 좌표를 저장하고 최근 60개 좌표를 선으로 연결해 궤적을 표시한다. 기본 confidence는 0.25, 검출 NMS IoU는 0.5이다. 별도 Re-ID 모델은 사용하지 않는다.

`--device` 인자가 없던 문제를 보완해 CPU 또는 GPU 장치를 지정할 수 있도록 했다. Colab에서는 모델과 입력 영상 준비 후 `--device 0`으로 실행한다. 출력 프레임 수 `N=round(원본 FPS×5)`에 대해 출력 FPS를 `N/5`로 설정하고 CSV 시간을 같은 출력 시간축에 맞췄다. 예를 들어 29.97 FPS 입력은 150프레임 / 30 FPS로 출력하여 5초를 구성하며 재생 속도는 약 0.1% 달라진다. 기존 영상·CSV가 있으면 덮어쓰지 않는다.

학습 모델의 검증은 완료했으나 `inputs/source.mp4`가 아직 준비되지 않아 실제 제출 영상은 생성되지 않았다. 원본을 제공받으면 검증된 모델로 `outputs/final_5s_render.mp4`를 만들고, [README의 FFmpeg 절차](../README.md#6-5초-추적-영상-제작)에 따라 별도 H.264 `outputs/final_5s.mp4`로 변환한다. 원래 렌더링 파일을 보존하고 기존 파일을 덮어쓰지 않는다. 최종 파일의 프레임 수·타임스탬프·정확히 5초의 재생 길이, 세 클래스의 표시, ID·궤적과 CSV 좌표 일치를 실제로 확인해야 한다. 아직 실제 영상에서 ID 유지·가림 후 연결·궤적 품질을 평가하지 않았다.

코드 테스트에서 사용하는 합성 영상과 모의 검출 결과는 제출 영상이나 모델 성능의 근거가 아니다.

로컬 새 환경에서 기존 `tests/test_pipeline.py`의 회귀 테스트 5개가 통과했다. 합성 29.97 FPS 입력에서 실제 OpenCV 출력·재디코딩을 통해 150프레임, 30 FPS, 5초를 확인했고 CSV의 마지막 시각 `149/30`과 ID별 궤적을 검사했다. 장치 인자 전달, 기존 출력 파일 보호, 클래스별 지표 순서·누락 처리도 확인했다. 해당 테스트 본체는 1.235초, import를 포함한 명령은 11.846초였다. 이후 실패 사례 매칭 테스트 6개와 Colab batch 정책 테스트 4개도 별도 실행에서 통과했다. 검출과 ByteTrack ID 연결 자체는 모의 결과를 사용하므로 성능 검증 대상이 아니다.

## 7. 이번 작업에서 확인한 내용

1. 데이터 파일이 Git에서 제외되어 있으므로 저장소 복제만으로 학습 자료가 준비되지 않는다. 이번에는 별도로 있던 `data.zip`의 기존 분할을 복원할 수 있었다.
2. 라벨 구문·좌표 검사와 이미지 디코딩은 통과해도 라벨 누락을 의미적으로 찾아내지는 못한다. 빈 라벨 연락시트에서 실제 형광펜이 보이는 사례를 발견했다.
3. CUDA 지원 PyTorch 설치와 CUDA 사용 가능한 하드웨어는 별개의 조건이다. 이 PC의 CUDA 빌드는 장치를 인식하지 못했다.
4. 저장 공간 부족은 패키지 설치도 중단시킨다. 원본을 삭제하지 않고 여유 있는 D:에 새 환경을 구성했다.
5. 소수 FPS 영상에서 프레임 수만 반올림하면 정확히 5초가 되지 않을 수 있다. 출력 시간축과 CSV 시간을 함께 맞춰야 한다.
6. 감지는 프레임의 클래스·박스를 예측하고, 추적은 시간에 따라 ID를 연결한다. 검출 정확도와 추적 ID 유지 성능은 각각 실험해야 한다.
7. 바이트 단위 완전 중복이 없어도 거의 같은 장면이 train·valid·test에 나뉠 수 있다. 추가 유사 장면 검사는 이 평가 한계를 드러냈으며, 자동 후보 수와 시각적으로 확인한 사례를 구분해 기록해야 한다.
8. Colab에서는 T4의 실제 CUDA 사용과 학습 기록을 확인한 뒤 100 epoch를 완료할 수 있었다. Drive 연결은 첫 실패 후 정상 재시도로 복구됐고, epoch별 백업은 100회 모두 성공했다. 브라우저 표시가 갱신되지 않을 때도 실제 CSV와 완료 JSON으로 진행 상태를 검증할 수 있었다.
9. mAP50이 높아도 수정테이프의 일부 박스·통합 박스 오류는 엄격한 IoU 평가에서 큰 차이를 만든다. 공식 AP/P/R, 저신뢰도 혼동 행렬, 고정 임계값 진단을 같은 지표로 읽지 않아야 한다.

## 8. 제출물 상태와 다음 단계

| 제출물 | 현재 상태 |
|---|---|
| 공개 GitHub 저장소 | 공개 상태 확인 |
| 데이터 구성·검사 기록 | 작성 완료 |
| Colab GPU 환경·전송·데이터 검사 | 실제 JSON 기록 확보, 검사 통과 |
| [models/best.pt](../models/best.pt) | 100 epoch 완료, SHA-256·CPU 로딩·세 클래스 검증 통과 |
| [outputs/test_metrics.json](../outputs/test_metrics.json) | Test 73장·108객체 실측값 확보 |
| 학습·평가 성능 그래프 | [train](assets/train/)·[test](assets/test/) 경로에 실제 그래프·CSV·설정 저장 |
| 실패 분석 | [진단 JSON](../outputs/failure_analysis.json)과 [9개 시각 사례](assets/failures/) 검토 완료 |
| outputs/final_5s.mp4 | 미생성 — 실제 입력 영상 대기 |
| outputs/trajectory.csv | 미생성 — 실제 입력 영상 대기 |
| 결과보고서 | 학습·평가·실패 분석 반영 완료, 실제 추적 결과 추가 필요 |

Colab 결과 ZIP의 CRC·경로·해시를 확인하고 모델·지표·그래프만 선택적으로 가져왔다. ZIP에 들어 있던 이전 README·보고서 사본으로 현재 문서를 덮어쓰지 않았다. ZIP 수집 당시의 문서 보완 대기 표시는 원본 수집 기록으로 보존하며, 최신 결과 설명은 이 보고서와 README를 기준으로 한다.

`runs/`가 Git에서 제외되므로 완료된 학습·평가·실패 그림을 `report/assets/`에 저장했다. 현재 결과 파일은 모두 GitHub 일반 push 제한 100 MiB 이내이며, 최대 파일은 6,500,778 bytes의 모델이다. 검증된 모델·지표·그래프·보고서를 공개 대상으로 준비했으며 원본 데이터와 로컬 환경은 Git에서 계속 제외한다.

남은 필수 작업은 실제 원본 영상을 받아 5초 추적 영상과 CSV를 제작하고, 영상·ID·궤적·시간 일치를 검증한 뒤 보고서와 공개 저장소에 추가하는 것이다. 로컬 CPU 학습은 일시정지 상태로 유지한다. 기존 라벨·장면 독립성 문제는 이번 기준 실험의 한계로 공개하며, 이후 보정 실험과 구분한다.
