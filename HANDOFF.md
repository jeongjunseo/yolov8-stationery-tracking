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

### Colab preparation verified - 2026-11-19 (KST)

- Code and assignment guidance were published in commit 139abc33ad09d5148bb1ccd9fa9f4c1168c0570a. The repository Colab link in README is ready to open.
- Drive listing verified exactly 25 uniquely named dataset parts with expected sizes totaling 2,451,554,279 bytes. The parts manifest and executable train_colab.ipynb were uploaded into MyDrive/yolov8-stationery-tracking, with metadata readback. Per-part and complete SHA256 validation occurs again when the notebook reconstructs the archive in Colab.
- One upload response timed out after Drive created the file; listing confirmed it existed, so it was not uploaded twice. No existing dataset or local files were removed.
- Ten additional pure unit tests passed (six failure-matching tests and four explicit batch-policy tests). All eight notebook code cells passed AST parsing. These are preparation checks, not evidence of completed model training.
- A final browser-controller retry still reported "The browser access host is unavailable". Actual GPU allocation and training have not started through the agent. The remaining handoff is to open the prepared Colab notebook, choose a GPU runtime, run all cells, and complete the normal Drive connection prompt. Then inspect the new Drive colab_runs folder and its environment/results evidence before updating training status.
- The saved local CPU processes were checked by creation time and remained stopped. Keep them paused while proceeding with Colab.

## Colab runtime and data verified - 2026-11-19 (KST)

- The first Colab attempt stopped in drive.mount with ValueError: mount failed before cloning or training. Its exact underlying cause was not established. A normal user retry succeeded; no data deletion or runtime reset was needed.
- Active Colab run ID: 20261119T124150Z_3c8dfaf4. Run backup folder: https://drive.google.com/drive/folders/120AIdt3Wj6mRk84Hm5tY9WxWRaBLqs-v under MyDrive/yolov8-stationery-tracking/colab_runs.
- Downloaded raw outputs/colab_environment.json verifies Tesla T4 (15,637,086,208 bytes GPU memory), CUDA available=True, CUDA 13.0, Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, Ultralytics 8.4.215, and executed Git commit f81977c78490fc59e0905e1cdcc0493765a7b4cd.
- Colab pip check reported a pre-existing IPython dependency warning: ipython 7.34.0 requires jedi, which is not installed. The separate core CUDA/import check passed. Do not describe the entire Colab pip check as clean.
- Downloaded raw outputs/colab_dataset_check.json verifies all 25 parts, the complete archive SHA256, 1,376 manifest files, and check_dataset.py success. Split counts remain train=498, valid=116, test=73. No image or label content was changed.
- At this update, a first completed training epoch and final test metrics have not yet been verified. Monitor the run folder for stationery/results.csv, weights, colab_training_run.json, and the final result ZIP. Do not start another training run or equate successful setup with completed training.
- Browser accessibility output became stale and screenshots were black, while Drive API reads continued to show new files. Prefer the actual Drive backup records for progress. Local CPU training remains paused.

### First GPU epoch verified

- Drive backup stationery/results.csv confirms epoch 1/100 completed, elapsed training time 125.905 seconds. This is an intermediate validation record, not test evaluation: precision=0.70129, recall=0.70148, mAP50=0.74282, mAP50-95=0.56468.
- The backup args.yaml verifies YOLOv8n, GPU device 0, batch 16, epochs 100, imgsz 640, workers 4, seed 42, deterministic=True, with original train.py settings. There is no observed GPU OOM or batch reduction.
- Both Drive weights/best.pt and weights/last.pt exist at 31,775,527 bytes. They are intermediate checkpoints, not a verified completed models/best.pt.
- Local monitoring snapshots are under ignored runs/colab_monitor/20261119T124150Z_3c8dfaf4. Training continues on Colab; do not interrupt it or resume the local CPU process. Wait for all 100 epochs and the completed training record before importing final artifacts.

### Video delivery readiness review

- Existing track_video.py writes mp4v; for reliable browser/GitHub playback, render to a new outputs/final_5s_render.mp4 and transcode a separate outputs/final_5s.mp4 with FFmpeg libx264/yuv420p/faststart, preserving original frame timestamps. README contains the exact non-overwriting command. Do not force a new frame rate or trim to hide an incorrect duration.
- No real video has been provided, and this conversion has not run. ffmpeg/ffprobe were not found on the local PATH during review. Check/install a trusted build only when preparing the actual clip, or use a verified available Colab build.
- The currently running Colab checkout predates this documentation update and will skip video because its source is absent. Keep the final real-video rendering and codec/duration/CSV verification as outstanding work after training.

### Cross-split near-scene audit during GPU training

- A read-only audit of all 687 images compared 102,590 cross-split pairs. Strong heuristic candidates require all of pHash64 distance <=6, horizontal+vertical dHash128 distance <=12, and resized 32x32 RGB mean absolute error <=12. This flagged 494 candidate pairs, all eraser: train-valid=239, train-test=210, valid-test=45.
- Candidate pairs involve 136 distinct train, 54 valid, and 30 test images. Of the test images, 29 have a candidate pair with a train image. These are heuristic participation counts, not counts of confirmed duplicate images.
- Root visually reviewed the 12 strongest image-disjoint pairs in outputs/dataset_audit/near_duplicate_candidates_01.jpg and _02.jpg: nearly the same eraser arrangement, green grid mat, and viewpoint across splits were visible. They use 12 distinct train, 6 valid, and 6 test images. Byte-identical cross-split duplicates remain zero.
- Six additional threshold-near candidates include visible object rotation/viewpoint changes. Do not call all 494 candidates exact duplicates or confirmed frame duplicates. The visual evidence does establish a material scene-level independence concern, particularly for eraser, so final test metrics must not be presented as unbiased performance on entirely new scenes.
- All original image/label SHA256 hashes still match the transfer archive manifest. The running baseline, labels, splits, and training conditions are unchanged. A future corrected experiment should use scene/session groups and a fresh independent test set; it must be distinguished from this baseline and must not interrupt the current authorized run.

## Colab training and test evaluation completed - 2026-11-20 (KST)

This section supersedes the intermediate training status above. The real video is still pending; the overall assignment is not yet complete.

- Run `20261119T124150Z_3c8dfaf4` completed all 100 epochs on Tesla T4 / `cuda:0`, with requested and actual batch 16. `outputs/colab_training_attempts.json` records one successful attempt; `outputs/colab_training_run.json` records zero backup failures. No GPU OOM or batch reduction occurred.
- The 100 rows of `report/assets/train/results.csv` cover epochs 1 through 100. Final elapsed CSV time is 9,834.78 seconds (2 h 43 min 54.78 s), excluding setup and subsequent evaluation. The maximum CSV validation mAP50-95 is 0.82416 at epoch 88; this observation alone is not a claim about checkpoint metadata or its selected epoch.
- Actual training settings were verified against `report/assets/train/args.yaml`: YOLOv8n, imgsz 640, workers argument 4, seed 42, deterministic=True, GPU 0, batch 16, epochs 100. Original data, labels, split, and `scripts/train.py` were preserved.
- The final result archive is `stationery_results_20261119T124150Z_3c8dfaf4.zip` (17,410,670 bytes), SHA-256 `71e290188099aa4ad4067da43ad32a2f210aab40f8a636ea96c1962eea0ae04d`. It remains in the private Drive run folder and in `D:\CodexEnvironments\yolov8-stationery-tracking-20261119\transfer\`. ZIP CRC, paths, provenance, metrics schema, sizes and hashes passed before selective import. Old README/report/code copies from the archive were not imported.
- `models/best.pt` is 6,500,778 bytes, SHA-256 `72eec58f57a0decd694a98606e9759824cb7cb4d75af345e9ae8db735a4d66ea`, matching the Colab training record. It loaded successfully as a CPU detection model in this PC's fresh environment with the expected three class names and 3,138,489 parameters. Loading did not change the file, and no local training was run.
- The completed GPU evaluation on 73 test images / 108 annotated objects is in `outputs/test_metrics.json`: precision=0.9370122590803192, recall=0.8876811594202899, mAP50=0.9357860759530722, mAP50-95=0.7668113026812392. Class mAP50-95 values are highlighter=0.8709275025799792, correction_tape=0.48350110678107205, eraser=0.9460052986826659. These results retain the annotation and scene-independence limitations above.
- Real training/test plots and nine diagnostic failure panels are now in `report/assets/train/`, `report/assets/test/`, and `report/assets/failures/`. All 42 images passed image-integrity checks. Fixed-threshold diagnosis in `outputs/failure_analysis.json` is distinct from Ultralytics AP/P/R reporting and its confusion matrix.
- Local CPU process IDs 15792 and 33548 were checked against their saved creation times after GPU completion and still report `stopped`. Their in-memory state, original checkpoints and logs remain preserved. Do not resume them automatically.
- `inputs/source.mp4` is still absent, and no video was found in the project's Drive folder. Neither `outputs/final_5s.mp4` nor `outputs/trajectory.csv` has been created. Do not substitute synthetic test footage. Once a real clip is provided, use the trained model, render IDs/class names/trails, encode a separate H.264 final file, and validate exact 5-second duration and CSV agreement.
- Final imported model/metrics/plots passed the 100 MiB per-file check; the largest artifact is the approximately 6.50 MB model. README and the report now include the measured results, real plots, failure analysis, verification evidence, and outstanding video requirements. Relative documentation links were checked before publication.

### GitHub publication verified - 2026-11-20 (KST)

- Published the completed model, test metrics, provenance, 42 plots/failure images, README, and report in commit [`4a1c72791f622667c0dff185a1154a6b689f9ce9`](https://github.com/jeongjunseo/yolov8-stationery-tracking/commit/4a1c72791f622667c0dff185a1154a6b689f9ce9). Push succeeded, and `git ls-remote` confirmed remote `main` matched that commit before this publication note was added.
- Git index verification checked all 94 tracked files against the 100 MiB limit, found no excluded raw data/environment/input files in the 54-file result commit, and confirmed the staged model SHA-256 still matched the Colab record. The working tree was clean after the result push.
- Remaining required input: a real video of at least five seconds at `C:\Users\Administrator\ml_HW\yolov8-stationery-tracking\inputs\source.mp4`, or a user-supplied local/Drive path. Do not retrain the completed model or rerun evaluation merely to create the clip. This workstation can use `--device cpu` for video inference; Colab can use `--device 0`. CPU video inference does not resume the suspended CPU training processes.
- After the real clip arrives, follow the non-overwriting render/H.264 procedure in README, verify 5 seconds, class names, IDs, trails, and CSV agreement, then publish the video/CSV and update the report. Until then, the complete assignment remains pending real-video deliverables.
