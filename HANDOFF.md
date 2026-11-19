# Training handoff

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

## GPU execution procedure (alternative; original plan)

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
- Installation of `requirements.txt` and `pip check` succeeded in the new environment: Python 3.14.0, Ultralytics 8.4.215, PyTorch 2.15.0+cpu, torchvision 0.29.2, OpenCV 4.14.0.94, PyYAML 6.0.3. New-environment CUDA availability is also `False`, with 0 devices. The user subsequently explicitly authorized CPU training in this environment. See `outputs/environment_check.json` and `outputs/training_status.json`.
- `scripts/train.py` and `data/stationery.yaml` retain their original configuration. The CPU command was `python scripts/train.py --device cpu --batch 16 --epochs 100`; device was the only requested-setting change. Other settings include imgsz 640, workers 4, seed 42, and deterministic=True. The log resolved `optimizer=auto` to AdamW with initial learning rate 0.001429.
- The original dataset was intentionally kept unchanged for this run, including the known empty-label anomaly. It is a limitation of the training labels, not an unresolved authorization requirement.
- `runs/` is a junction to `D:\CodexEnvironments\yolov8-stationery-tracking-20261119\runs` to avoid C: disk pressure. `outputs/training_status.json` records the latest phase. Do not start a duplicate run or terminate an active training process.
- First CPU attempt: started 2026-11-19 19:17:51 KST; exited at 19:18:27 KST with code 1. The log reports `DefaultCPUAllocator: not enough memory` while trying to allocate 13,107,200 bytes. No epoch result was produced. Its launcher/logs and status snapshot are preserved under `runs/session_cpu_20261119/`; this attempt ended on error rather than being interrupted.
- CPU batch-8 retry: started 2026-11-19 19:20:06 KST; exited at 19:21:02 KST with code 1. It failed at epoch 1, batch 11/63 with another CPU allocation error (16,384,000 bytes requested); no epoch completed. Its launcher/logs and `status.json` are preserved under `runs/session_cpu_20261119_b8/`, and the actual run directory is `runs/train/stationery-2/`.
- CPU batch-4 retry: started 2026-11-19 19:22:27 KST using `python scripts/train.py --device cpu --batch 4 --epochs 100`; it is actively processing training batches. Its launcher/logs are under `runs/session_cpu_20261119_b4/`, and the actual run directory is `runs/train/stationery-3/`. Only batch was reduced after the observed CPU memory errors, applying the existing 8/4 memory fallback to the CPU execution authorized by the user. The log reports 0 dataloader workers because Ultralytics overrides the unchanged workers=4 argument for CPU execution. Other settings and original data remain unchanged.
- The launcher is configured to run `python scripts/evaluate.py --device cpu` automatically only after training succeeds and `models/best.pt` exists. Evaluation has not run yet.
- `inputs/source.mp4` is absent; the user said the source video will be prepared. Training and evaluation should continue while the video is pending. No completed trained model, test metrics, final video, or trajectory CSV exists yet. Setup plots are not completed learning curves.
- Tracking now accepts `--device`, normalizes the output FPS for the requested duration, timestamps CSV rows in output-video time, and refuses to overwrite an existing output video or CSV.
- Evaluation now reports class-specific precision, recall, mAP50, mAP50-95, and target counts, mapped through `ap_class_index`; unavailable class metrics are `null`.
- All five `tests/test_pipeline.py` regression tests passed in the new environment (1.235 s test body, 11.846 s including imports). They use temporary synthetic video and mocked detections with real Ultralytics result rendering/OpenCV encoding. They verify 150 decoded frames at 30 FPS, matching CSV timestamps, IDs/trajectories/device forwarding, existing-output protection, and per-class metric mapping. They do not establish real detection or tracking accuracy.
- `report/REPORT.md` is an evidence-based progress report, not a completed experiment report.

### Earlier CPU execution plan (superseded by the pause and Colab direction below)

1. Monitor the active batch-4 CPU run without stopping it. Preserve failed-run logs and existing files. Record any additional error and necessary memory-related adjustment; do not silently change the remaining training conditions.
2. Keep the existing split and labels intact. Disclose the known empty-label anomaly and unverified scene-level leakage when interpreting results. Any later corrected-data experiment must be distinguished from this baseline run.
3. Confirm successful training and `models/best.pt`, then complete `python scripts/evaluate.py --device cpu`. Inspect actual per-class metrics and failure cases, and copy selected training/evaluation graphs out of ignored `runs/`.
4. Place a real video of at least five seconds at `C:\Users\Administrator\ml_HW\yolov8-stationery-tracking\inputs\source.mp4`. Run `python scripts/track_video.py --source inputs/source.mp4 --model models/best.pt --device cpu` after a trained model and input video exist.
5. Validate five-second duration, class names, track IDs, trajectories, and CSV/video agreement. Complete README/report using measured outcomes and verify every staged file is below 100 MiB before committing and pushing final artifacts.
6. If moving to another PC later, copy only the prepared data and project artifacts, create a fresh environment, and follow the GPU procedure above if NVIDIA hardware is available. Do not copy the local environment or junctions.


## Paused by user - 2026-11-19 19:28 KST

The user explicitly requested a temporary pause. The batch-4 CPU training subprocess tree is suspended at the OS level; CPU time stopped advancing and in-memory state is retained. No training process was terminated. The latest process IDs and creation times are stored in outputs/training_status.json (phase: paused). The launcher is waiting and will not evaluate until training resumes and finishes. The temporary keep-awake helper releases its request after seeing the paused state. Do not resume training without the user asking to resume.

Completed epochs: 1. Last progress: 2/100         0G      1.037      1.502      1.179         28        640: 44% ━━━━━─────── 56/125 1.3s/it 1:32<1:31. Checkpoints under runs/train/stationery-3/weights/ remain intact. To resume, verify the saved process creation times match, resume only the saved suspended processes, restore phase to training, and re-enable the temporary keep-awake guard. Avoid launching a duplicate training run. Resuming from an on-disk checkpoint is only needed if the suspended processes no longer exist.

## Latest direction: Colab GPU training - 2026-11-19

The user confirmed the supplied course references are the professor's assignment instructions and explicitly requested training in Colab. Read docs/ASSIGNMENT_GUIDELINES.md and the new root AGENTS.md. The local CPU run remains paused; this Colab request does not resume it.

- notebooks/train_colab.ipynb is the executable Colab workflow. scripts/colab_support.py calls the unchanged original train.main with GPU 0, batch 16, epochs 100, adding checkpoint backup and a version-guarded callback that disables Ultralytics 8.4.215 internal batch retries. Only the notebook may retry batch 8 or 4 after actual CUDA OOM; unexpected actual batch values stop execution. Model, augmentation, optimizer, and seed settings stay unchanged.
- The prepared dataset archive is 2,451,554,279 bytes, SHA256 1ee9ad1ba42e00a2af6a04de79d44358373837db4448d4efb3d7be66e998acf8. It contains 687 image/label pairs, data/stationery.yaml, data/SOURCES.md, and a per-file SHA256 manifest. No raw-source ZIPs or virtual environment are included.
- The transfer tool has a server-side 100 MiB limit, so transfer/chunks_100mb under the D: environment folder contains 25 raw parts (24 x 100,000,000 bytes and one x 51,554,279 bytes) plus stationery_dataset.parts.json. The original archive, earlier larger transfer parts, and dataset remain intact.
- The Drive destination is MyDrive/yolov8-stationery-tracking. The notebook mounts Drive, reconstructs the archive on local /content storage, verifies all hashes, checks the original split, and refuses CPU fallback. Windows/Linux newline differences are tolerated only for the two metadata files after normalized-content equality verification; image/label mismatches stop extraction.
- Each completed training epoch backs up weights and results.csv to colab_runs/<run-id> in the Drive project. GPU environment and dataset verification JSON are backed up separately for observation. A successful training result requires an actual CUDA trainer and 100 completed epochs.
- Evaluation and failure analysis run after successful training. A missing real source video leaves tracking pending. The final artifact ZIP contains the model, measured metrics, plots, and analysis; report prose still requires reconciliation with actual outcomes before submission.
- No actual Colab runtime has been verified yet. Browser input attempts failed with GetCursorPos access denied; reconnecting the browser controller reported that its access host was unavailable, even after the user confirmed the desktop was unlocked. File-transfer and notebook preparation are separate from actual GPU execution. Verify Drive transfer completeness and actual notebook execution before claiming training started.

### Colab preparation verified - 2026-11-19 20:42 KST

- Code and assignment guidance were published in commit 139abc33ad09d5148bb1ccd9fa9f4c1168c0570a. The repository Colab link in README is ready to open.
- Drive listing verified exactly 25 uniquely named dataset parts with expected sizes totaling 2,451,554,279 bytes. The parts manifest and executable train_colab.ipynb were uploaded into MyDrive/yolov8-stationery-tracking, with metadata readback. Per-part and complete SHA256 validation occurs again when the notebook reconstructs the archive in Colab.
- One upload response timed out after Drive created the file; listing confirmed it existed, so it was not uploaded twice. No existing dataset or local files were removed.
- Ten additional pure unit tests passed (six failure-matching tests and four explicit batch-policy tests). All eight notebook code cells passed AST parsing. These are preparation checks, not evidence of completed model training.
- A final browser-controller retry still reported "The browser access host is unavailable". Actual GPU allocation and training have not started through the agent. The remaining handoff is to open the prepared Colab notebook, choose a GPU runtime, run all cells, and complete the normal Drive connection prompt. Then inspect the new Drive colab_runs folder and its environment/results evidence before updating training status.
- The saved local CPU processes were checked by creation time and remained stopped. Keep them paused while proceeding with Colab.
