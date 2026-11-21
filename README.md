# YOLOv8 문구류 객체 감지·추적 결과보고서

형광펜, 수정테이프, 지우개를 YOLOv8n으로 학습하고, ByteTrack으로 클래스명·추적 ID·이동 궤적을 표시한 5초 영상을 제작했다.

**GitHub:** [jeongjunseo/yolov8-stationery-tracking](https://github.com/jeongjunseo/yolov8-stationery-tracking)

## 학습 과정

Roboflow 공개 데이터셋을 모아 세 클래스의 라벨 형식을 통일했다. 클래스별 포함 이미지는 형광펜 219장, 수정테이프 129장, 지우개 318장이다. 빈 라벨 21장을 포함한 총 687장을 train 498장, validation 116장, test 73장으로 나누어 사용했다. [데이터 출처 및 이용 조건](data/SOURCES.md)

Google Colab의 Tesla T4에서 YOLOv8n을 100 epoch 학습했다. Batch size는 16, 입력 크기는 640, seed는 42로 설정했으며 학습 시간은 약 2시간 44분이었다.

[학습 곡선](report/assets/train/results.png) · [학습 설정](report/assets/train/args.yaml)

## 성능

기본 모델을 test 이미지 73장, 정답 객체 108개로 평가했다. 수치는 % 단위이다.

| 클래스 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 전체 | 93.70 | 88.77 | 93.58 | 76.68 |
| 형광펜 | 89.07 | 91.30 | 93.68 | 87.09 |
| 수정테이프 | 93.06 | 75.00 | 87.56 | 48.35 |
| 지우개 | 98.98 | 100.00 | 99.50 | 94.60 |

수정테이프는 물체 일부만 잡거나 여러 물체를 하나로 묶는 위치 오류가 있었다. 지우개는 학습·평가 데이터에 유사한 장면이 있어 성능이 높게 측정됐을 가능성이 있다. 일부 누락 라벨도 데이터의 한계로 남아 있다.

## 결과물

| 결과물 | 링크 |
|---|---|
| 기본 학습 모델 | [best.pt](models/best.pt) |
| 기본 모델의 5초 영상 | [final_5s.mp4](outputs/final_5s.mp4) |
| 평가 수치·이동 궤적 | [test_metrics.json](outputs/test_metrics.json) · [trajectory.csv](outputs/trajectory.csv) |
| 추가 학습 후 5초 영상 | [recheck_5s.mp4](outputs/video_finetune/recheck_5s.mp4) · [궤적 CSV](outputs/video_finetune/recheck_trajectory.csv) |

영상은 30 FPS, 150프레임으로 정확히 5초다. ByteTrack으로 ID를 연결하고 최근 60개의 중심 좌표를 궤적으로 표시했다. 궤적은 카메라 움직임을 포함한 화면상의 좌표 변화다.

기본 영상에서는 형광펜을 놓치고 지우개를 형광펜으로 잘못 분류했다. 영상에서 추출한 48장을 추가 학습한 뒤에는 앞 5초에서 세 물체의 클래스와 ID가 유지됐다. 보완 영상에는 97 epoch까지 학습한 결과 중 82 epoch의 최고 모델을 사용했다. 같은 영상으로 학습하고 확인한 결과이며, 별도 test 평가와 보완 가중치 공개는 아직 진행하지 않았다. 위 성능 표는 기본 모델의 결과다.

## 과제 과정에서 경험한 내용

- 로컬 CPU에서는 메모리가 부족해 batch를 줄여도 학습이 어려웠다. Colab의 T4 GPU로 옮긴 뒤 batch 16으로 학습을 마쳤다.
- HDR로 촬영된 영상은 SDR로 변환했다. 이미지 test 점수가 높았지만 실제 영상에서는 오분류가 나와, 영상 속 물체의 이미지를 추가로 학습했다.
- 추가 학습은 Colab 연결이 끊긴 뒤 GPU 사용 한도에 걸려 재개하지 못했다. 매 epoch Drive에 백업해 둔 체크포인트로 97 epoch까지의 결과를 보존했다.

실험 과정과 실패 사례는 [상세 보고서](report/REPORT.md), 설치·학습·영상 처리 명령은 [실행 안내](docs/RUNNING.md)에 정리했다.
