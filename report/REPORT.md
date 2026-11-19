# YOLOv8 문구류 객체 감지 및 추적 — 진행 보고서

기록일: 2026-11-19 (Asia/Seoul)

저장소: https://github.com/jeongjunseo/yolov8-stationery-tracking

**현재 상태: 데이터·실행 환경 검증 완료 범위를 기록한 진행 보고서이다. GPU 학습, test 평가, 실제 추적 영상 제작이 완료되지 않아 최종 결과보고서로 제출할 수 없다.**

## 1. 목적과 현재 결과

YOLOv8n으로 형광펜(`highlighter`, ID 0), 수정테이프(`correction_tape`, ID 1), 지우개(`eraser`, ID 2)를 학습하고, 별도 test split에서 성능을 평가한 뒤 ByteTrack으로 클래스명·추적 ID·이동 궤적이 표시되는 5초 영상을 만드는 것이 목표이다.

이전 컴퓨터의 `HANDOFF.md`를 기준으로 작업을 재개했다. 저장소를 복제하고 제공된 `data.zip`에서 기존 분할 데이터를 복원했다. 데이터와 라벨을 새로 만들거나 재분할하지 않았다. 현재 PC에서는 NVIDIA GPU가 확인되지 않아 요청된 CUDA 학습을 시작하지 않았다. 입력 영상도 아직 없다.

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

### 빈 라벨과 남은 데이터 품질 문제

highlighter 원본 240장 중 21장의 라벨 파일이 비어 있어 해당 클래스의 라벨 포함 이미지는 219장이다. 빈 라벨 파일이 있다는 사실만으로 올바른 배경 이미지라고 판단할 수 없다.

![빈 라벨 21장 감사 연락시트](../outputs/dataset_audit/empty_labels_contact_sheet.jpg)

연락시트 02번 train 이미지 `highlighter_IMG_20260713_162103695_HDR_jpg.rf.Sbrd0MiyGuUjja7V5nXf.jpg`에는 왼쪽 가장자리에 녹색 뚜껑의 형광펜이 보이는데 라벨은 비어 있다. 최소 이 사례는 누락 객체 라벨 검토 대상이다. 원본 라벨은 수정하지 않았다. 이 사례는 학습 모델의 미검출 사례가 아니라 **정답 데이터의 문제**이다.

나머지 빈 라벨도 썸네일 검토만 했으므로 원해상도에서의 정답 완전성을 보증하지 않는다. SHA-256 완전 중복 검사와 달리 유사 장면·연속 프레임의 split 간 누출은 확인하지 않았다. 따라서 장면 단위 분리가 완료됐다고 단정하지 않는다.

## 4. 실행 환경과 학습 설정

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

새 환경에서 `requirements.txt` 설치와 `pip check`는 성공했다. 이 환경은 코드 검증용 CPU 빌드이며 CUDA 학습 환경이 마련됐다는 뜻은 아니다. [환경 확인 JSON](../outputs/environment_check.json)에 실제 버전과 CUDA 상태를 저장했다. NVIDIA GPU PC에서는 해당 하드웨어·드라이버에 맞는 CUDA PyTorch를 별도로 설치해야 한다.

예정 학습 조건은 기존 `scripts/train.py` 그대로이다.

| 항목 | 설정 |
|---|---|
| 모델 | COCO 사전 학습 YOLOv8n (`yolov8n.pt`) |
| epochs / imgsz | 100 / 640 |
| batch / workers | 16 / 4 |
| seed / deterministic | 42 / True |
| device | 0 (CUDA 사용 가능 확인 후) |

```powershell
python scripts/train.py --device 0 --batch 16 --epochs 100
```

GPU 메모리 부족이 실제로 발생할 때만 batch를 8 또는 4로 낮춘다. 아직 학습을 시작하지 않아 학습 손실, 최적 epoch, 과적합 여부, 학습 시간, 성능 그래프는 측정되지 않았다.

## 5. 성능 평가 상태와 분석 계획

학습 완료 후 `python scripts/evaluate.py --device 0`으로 test 73장, 정답 객체 108개를 평가한다. 결과는 `outputs/test_metrics.json`에 저장한다.

| 지표 | 현재 결과 |
|---|---|
| Precision | 미측정 |
| Recall | 미측정 |
| mAP50 | 미측정 |
| mAP75 | 미측정 |
| mAP50-95 | 미측정 |

클래스별 Precision, Recall, mAP50, mAP50-95와 정답 객체 수도 저장하도록 평가 스크립트를 보완했다. `ap_class_index`를 기준으로 클래스와 평가 행을 대응시켜 순서가 달라도 올바르게 기록하며, 평가되지 않은 클래스는 평균값 대신 `null`로 기록한다.

현재 데이터에는 클래스별 이미지 수 차이가 있지만 실제 성능 순위를 추정해 결과처럼 쓰지 않는다. 학습 후 test 예측과 정답을 비교해 오검출·미검출·위치 오차의 실제 사례를 선정하고, 각도·크기·배경·조명·가림과의 관계를 해당 이미지에서 확인해야 한다. test 결과를 보고 임계값을 조정한 경우에는 원래 평가와 탐색 결과를 구분해야 한다.

## 6. 추적 구현과 영상 상태

검출 결과를 ByteTrack에 전달하고 `persist=True`로 프레임 사이 추적 상태를 유지한다. ID별 객체 중심 좌표를 저장하고 최근 60개 좌표를 선으로 연결해 궤적을 표시한다. 기본 confidence는 0.25, 검출 NMS IoU는 0.5이다. 별도 Re-ID 모델은 사용하지 않는다.

`--device` 인자가 없던 문제를 보완해 요청된 GPU 명령을 받을 수 있도록 했다. 출력 프레임 수 `N=round(원본 FPS×5)`에 대해 출력 FPS를 `N/5`로 설정하고 CSV 시간을 같은 출력 시간축에 맞췄다. 예를 들어 29.97 FPS 입력은 150프레임 / 30 FPS로 출력하여 5초를 구성하며 재생 속도는 약 0.1% 달라진다. 기존 영상·CSV가 있으면 덮어쓰지 않는다.

`inputs/source.mp4`와 학습된 `models/best.pt`가 아직 없으므로 실제 영상은 생성되지 않았다. 코드 테스트에서 사용하는 합성 영상과 모의 검출 결과는 제출 영상이나 모델 성능의 근거가 아니다. 실제 생성 후 영상 재생 길이, 세 클래스의 표시, ID와 궤적, CSV 좌표의 움직임을 확인해야 한다.

새 환경에서 `python -m unittest discover -s tests -v`의 회귀 테스트 5개가 통과했다. 합성 29.97 FPS 입력에서 실제 OpenCV 출력·재디코딩을 통해 150프레임, 30 FPS, 5초를 확인했고 CSV의 마지막 시각 `149/30`과 ID별 궤적을 검사했다. 장치 인자 전달, 기존 출력 파일 보호, 클래스별 지표 순서·누락 처리도 확인했다. 테스트 본체는 1.235초, import를 포함한 명령은 11.846초였다. 검출과 ByteTrack ID 연결 자체는 모의 결과를 사용하므로 성능 검증 대상이 아니다.

## 7. 이번 작업에서 확인한 내용

1. 데이터 파일이 Git에서 제외되어 있으므로 저장소 복제만으로 학습 자료가 준비되지 않는다. 이번에는 별도로 있던 `data.zip`의 기존 분할을 복원할 수 있었다.
2. 라벨 구문·좌표 검사와 이미지 디코딩은 통과해도 라벨 누락을 의미적으로 찾아내지는 못한다. 빈 라벨 연락시트에서 실제 형광펜이 보이는 사례를 발견했다.
3. CUDA 지원 PyTorch 설치와 CUDA 사용 가능한 하드웨어는 별개의 조건이다. 이 PC의 CUDA 빌드는 장치를 인식하지 못했다.
4. 저장 공간 부족은 패키지 설치도 중단시킨다. 원본을 삭제하지 않고 여유 있는 D:에 새 환경을 구성했다.
5. 소수 FPS 영상에서 프레임 수만 반올림하면 정확히 5초가 되지 않을 수 있다. 출력 시간축과 CSV 시간을 함께 맞춰야 한다.
6. 감지는 프레임의 클래스·박스를 예측하고, 추적은 시간에 따라 ID를 연결한다. 검출 정확도와 추적 ID 유지 성능은 각각 실험해야 한다.

## 8. 제출물 상태와 다음 단계

| 제출물 | 현재 상태 |
|---|---|
| 공개 GitHub 저장소 | 공개 상태 확인 |
| 데이터 구성·검사 기록 | 작성 완료 |
| models/best.pt | 미생성 — GPU 학습 필요 |
| outputs/test_metrics.json | 미생성 — 학습 모델 필요 |
| 학습·평가 성능 그래프 | 미생성 |
| outputs/final_5s.mp4 | 미생성 — 모델과 입력 영상 필요 |
| outputs/trajectory.csv | 미생성 |
| 최종 결과보고서 | 미완료 — 현재 문서는 진행 보고서 |

NVIDIA GPU 환경 확보, 빈 라벨 이상 검토, CUDA 학습, test 평가, 실제 추적 영상 제작 순으로 이어가야 한다. `runs/`는 Git에서 제외되므로 완료된 학습·평가 그래프는 `report/assets/` 등 공개 경로로 복사한다. 각 제출 파일이 GitHub 일반 push 제한 100 MiB 이내인지 확인하고 최종 산출물을 커밋·push해야 한다.
