# GPU PC handoff

## Goal

Train a YOLOv8 detector for three stationery classes, evaluate it, and create a five-second ByteTrack result video with track IDs and trajectories.

## Repository

https://github.com/jeongjunseo/yolov8-stationery-tracking

## Classes

```yaml
0: highlighter
1: correction_tape
2: eraser
```

## Dataset status

The merged dataset is prepared and passes the structural checks in `scripts/check_dataset.py`. The latest audit below identifies an empty-label anomaly that still needs semantic review.

| Split | Images |
|---|---:|
| train | 498 |
| valid | 116 |
| test | 73 |
| total | 687 |

Images containing each class:

- highlighter: 219
- correction_tape: 129
- eraser: 318

Processing already completed:

- Roboflow ZIP files were merged with unique filename prefixes.
- Highlighter polygon labels were converted to detection bounding boxes.
- Three eraser color classes were merged into `eraser`.
- Six correction-tape boxes that slightly exceeded image boundaries were clipped.
- Dataset sources and CC BY 4.0 attribution are recorded in `data/SOURCES.md`.

The actual images and labels are ignored by Git because of their size. Copy these directories to the GPU PC separately:

```text
data/train/
data/valid/
data/test/
```

`data/raw/` contains the original ZIP files and is not needed for training.

## Previous machine status (historical)

- Python: 3.14.3
- Ultralytics: 8.4.215
- PyTorch: 2.15.0+cpu
- CUDA available: no
- Training has not started.

Do not copy `.venv` to the GPU PC. Create a fresh virtual environment there.

## Next steps on the GPU PC

1. Confirm the GPU and driver with `nvidia-smi`.
2. Create a fresh `.venv`.
3. Install a CUDA-enabled PyTorch build compatible with that PC.
4. Install `requirements.txt`.
5. Confirm `torch.cuda.is_available()` returns `True`.
6. Validate the copied dataset:

   ```powershell
   python scripts/check_dataset.py
   ```

7. Train:

   ```powershell
   python scripts/train.py --device 0 --batch 16 --epochs 100
   ```

8. Evaluate after training:

   ```powershell
   python scripts/evaluate.py --device 0
   ```

The training script copies the best weights to `models/best.pt`. After evaluation, provide a source video of at least five seconds at `inputs/source.mp4`, then run `scripts/track_video.py`.

## Latest handoff — 2026-11-19, Windows workstation

This section is the latest state; the machine details above describe the previous PC.

- Repository cloned to `C:\Users\Administrator\ml_HW\yolov8-stationery-tracking` on branch `main`.
- No root `AGENTS.md` exists in the cloned repository. `HANDOFF.md` and `README.md` were read before making changes.
- The pre-existing `C:\Users\Administrator\ml_HW\data.zip` contains the prepared data. Only `data/train/`, `data/valid/`, and `data/test/` were restored. The original archive was preserved; unnecessary `data/raw/` archives were not extracted.
- All six image/label directories are populated. `scripts/check_dataset.py` passed with 498 / 116 / 73 images. All 687 images decoded successfully; no matching label files are missing.
- The 687 labels contain 868 boxes and 21 empty files. The audit and exact counts are in `outputs/dataset_summary.json`.
- Visual inspection of `outputs/dataset_audit/empty_labels_contact_sheet.jpg` found a visible green-capped highlighter in sample 02 despite an empty label. This is an unresolved annotation issue, not a failed model prediction. Original labels were preserved. Near-duplicate/scene-level leakage has not been checked.
- Hardware: AMD Ryzen 5 5500GT, AMD Radeon(TM) Graphics, approximately 16 GB RAM; no NVIDIA display device was found. `nvidia-smi` is absent from PATH and its standard installation locations.
- Existing system Python is 3.14.0 with PyTorch 2.9.0+cu126, but CUDA availability is `False` and CUDA device count is 0. The existing system installation is not the project environment.
- A new project virtual environment was created. Installation initially failed because C: ran out of space. That newly created incomplete environment was preserved under `D:\CodexEnvironments\yolov8-stationery-tracking-20261119\.venv-incomplete`.
- The project `.venv` is now a Windows junction pointing to a newly created environment at `D:\CodexEnvironments\yolov8-stationery-tracking-20261119\.venv`. This is local to this machine; do not copy it to another PC. Use `.venv\Scripts\python.exe` from the project as usual. Package cache and temporary installation files were placed on D:.
- Installation of `requirements.txt` and `pip check` succeeded in the new environment: Python 3.14.0, Ultralytics 8.4.215, PyTorch 2.15.0+cpu, torchvision 0.29.2, OpenCV 4.14.0.94, PyYAML 6.0.3. New-environment CUDA availability is also `False`, with 0 devices. This is a code-validation environment, not a working CUDA training environment; no compatible NVIDIA device was available to select a CUDA wheel for. See `outputs/environment_check.json`.
- `scripts/train.py` and `data/stationery.yaml` retain their original training configuration. No training was started and no CPU training fallback was authorized.
- `inputs/source.mp4` is absent. No trained model, test metrics, final video, trajectory CSV, or training graphs exist yet.
- Tracking now accepts `--device`, normalizes the output FPS for the requested duration, timestamps CSV rows in output-video time, and refuses to overwrite an existing output video or CSV.
- Evaluation now reports class-specific precision, recall, mAP50, mAP50-95, and target counts, mapped through `ap_class_index`; unavailable class metrics are `null`.
- All five `tests/test_pipeline.py` regression tests passed in the new environment (1.235 s test body, 11.846 s including imports). They use temporary synthetic video and mocked detections with real Ultralytics result rendering/OpenCV encoding. They verify 150 decoded frames at 30 FPS, matching CSV timestamps, IDs/trajectories/device forwarding, existing-output protection, and per-class metric mapping. They do not establish real detection or tracking accuracy.
- `report/REPORT.md` is an evidence-based progress report, not a completed experiment report.

### Required next inputs and execution

1. Continue on an NVIDIA GPU PC (or obtain an explicit change of training device from the user). Confirm the actual GPU, driver, and `torch.cuda.is_available() == True`; installing a CUDA wheel does not supply missing GPU hardware.
2. On another PC, copy the existing `data/train`, `data/valid`, and `data/test` folders or extract them from `data.zip` into the repository's `data` folder. Preserve source archives and do not regenerate the prepared split.
3. Review the identified empty-label anomaly before the definitive training run. Any label correction must be recorded and reflected in refreshed data counts.
4. Create a fresh environment on the GPU PC, install a compatible CUDA PyTorch build and `requirements.txt`, then run the dataset check.
5. Run the original command: `python scripts/train.py --device 0 --batch 16 --epochs 100`. Lower only batch to 8 or 4 if GPU out-of-memory occurs. Keep other settings unchanged.
6. Confirm `models/best.pt`, run `python scripts/evaluate.py --device 0`, inspect actual per-class metrics and failure cases, and copy selected training/evaluation graphs out of ignored `runs/`.
7. Place a real video of at least five seconds at `C:\Users\Administrator\ml_HW\yolov8-stationery-tracking\inputs\source.mp4` on this PC, or at `inputs/source.mp4` in the GPU-PC checkout. Run `python scripts/track_video.py --source inputs/source.mp4 --model models/best.pt --device 0`.
8. Validate five-second duration, class names, track IDs, trajectories, and CSV/video agreement. Complete README/report using measured outcomes and verify every staged file is below 100 MiB before committing and pushing final artifacts.

