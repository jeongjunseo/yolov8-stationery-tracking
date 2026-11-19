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

The merged dataset is ready and passes `scripts/check_dataset.py`.

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

## Current machine status

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

