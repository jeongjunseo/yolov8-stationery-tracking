# 실행 안내

명령은 프로젝트 루트에서 실행한다. 데이터 이미지·라벨과 원본 영상은 GitHub에 포함하지 않았다.

## Colab 학습

[학습 노트북](https://colab.research.google.com/github/jeongjunseo/yolov8-stationery-tracking/blob/main/notebooks/train_colab.ipynb)에서 GPU 런타임을 선택하고 Drive를 연결한 뒤 셀을 순서대로 실행한다. 데이터는 Drive의 `MyDrive/yolov8-stationery-tracking/`에 둔다. 노트북이 데이터 해시와 CUDA를 확인하고 학습·평가 및 체크포인트 백업을 수행한다.

중단된 추가 학습의 98–100 epoch는 [복구 노트북](https://colab.research.google.com/github/jeongjunseo/yolov8-stationery-tracking/blob/main/notebooks/resume_video_finetune_colab.ipynb)을 사용한다. 실행 상태와 백업 위치는 [HANDOFF.md](../HANDOFF.md)에 있다.

## 로컬 환경

Windows PowerShell에서 새 가상환경을 만든다.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

NVIDIA GPU에서 실행하려면 드라이버에 맞는 CUDA 지원 PyTorch를 설치하고 아래 결과가 `True`인지 확인한다. Colab 노트북은 자체 환경 설치 절차를 사용한다.

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

데이터는 `data/train`, `data/valid`, `data/test` 각각의 `images`, `labels` 폴더에 배치한다. 클래스 ID는 `0: highlighter`, `1: correction_tape`, `2: eraser`이다.

```powershell
python scripts/check_dataset.py
python scripts/train.py --device 0 --batch 16 --epochs 100
python scripts/evaluate.py --device 0
```

GPU 메모리 부족이 발생할 때만 batch를 8 또는 4로 낮춘다. 학습 결과는 `runs/train/`, 최적 가중치는 `models/best.pt`, 평가 수치는 `outputs/test_metrics.json`에 저장된다.

## 공개된 기본 영상의 생성 명령

원본 `1000030242.mp4`는 프로젝트 상위 폴더에 두었다. HEVC 10-bit HDR 영상을 SDR로 변환한 뒤, 기본 모델로 앞 5초를 추적했다. 실제 실행에는 FFmpeg 9.0.2의 `zscale`, `tonemap`, `libx264`를 사용했다.

다음은 기존 결과를 생성한 명령이다. 재실행할 때는 이미 존재하는 영상과 CSV 대신 새 출력 경로를 지정한다. 추적 스크립트와 FFmpeg의 `-n` 옵션은 기존 파일을 덮어쓰지 않는다.

```powershell
ffmpeg -n -i ../1000030242.mp4 -map 0:v:0 -an -map_metadata -1 -vf "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p" -c:v libx264 -crf 16 -preset fast -pix_fmt yuv420p -fps_mode passthrough -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart inputs/source_sdr.mp4
python scripts/track_video.py --source inputs/source_sdr.mp4 --model models/best.pt --device cpu --imgsz 1280 --rotate-inference 90 --agnostic-nms --conf 0.25 --iou 0.5 --trail-length 60 --tracker bytetrack.yaml --duration 5 --output outputs/final_5s_render.mp4 --trajectory outputs/trajectory.csv
ffmpeg -n -i outputs/final_5s_render.mp4 -map 0:v:0 -an -map_metadata -1 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -fps_mode passthrough -movflags +faststart outputs/final_5s.mp4
```

ByteTrack은 프레임 사이 ID를 연결하며 별도 Re-ID 모델을 사용하지 않는다. 추론용 회전 좌표는 원래 영상 좌표로 복원한다. CSV의 시간과 중심 좌표는 출력 영상 기준이다. 공개 파일은 H.264/yuv420p, 1080 × 1920, 30 FPS, 150프레임이다.

보완 영상은 별도 모델을 CPU·imgsz 640·회전 0·클래스별 NMS로 실행했다. 기준 모델의 위 명령과 구분하며, 설정과 모델 해시는 [재검증 기록](../outputs/video_finetune/video_recheck.json)에 있다.

## 코드 검사

```powershell
python -m unittest discover -s tests -v
```

실행 기록, 데이터 문제와 실제 영상의 실패 사례는 [결과보고서](../report/REPORT.md)를 참고한다.
