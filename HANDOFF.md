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

## Real-video result verified - 2026-11-20 (KST)

This section supersedes the earlier missing-video status. All requested artifact files now exist, but successful detection and classification of all three physical objects has not been achieved on the new video. Do not describe file-format verification as tracking accuracy.

- The user uploaded `C:\Users\Administrator\ml_HW\1000030242.mp4` (26,398,062 bytes), SHA-256 `c284b4cc16c9d19368e8c427e81b143c40e5bd4e754cc63b8dfebe22638531a2`. It remains unchanged at that location; `inputs/source.mp4` was not created. Raw input video is not included in Git.
- FFprobe identified HEVC Main 10, BT.2020/PQ HDR, rotation metadata -90 degrees, 287 frames, duration 9.566089 s, displayed at 1080 x 1920. FFmpeg 9.0.2 from the Windows build linked by ffmpeg.org was downloaded to the D: tools directory and checked against the vendor's SHA-256. The tone-mapped input, all intermediate renders/CSVs, diagnostics and review frames are preserved under ignored `runs/tracking_20261120_1000030242/` on D:.
- A new SDR BT.709 input was made with zscale linearization and Mobius tone mapping; all 287 frames were retained. The first 150 frames were used for the result. Default raw-video and SDR-only tracking both performed poorly. A 24-case sample diagnosis and four full five-second runs are summarized in `outputs/tracking_experiments.json`. These are deployment-video explorations, not new independent test evaluation.
- Final inference used the unchanged `models/best.pt`, local CPU, imgsz 1280, clockwise inference rotation 90 degrees, class-agnostic NMS, confidence 0.25, NMS IoU 0.5, default ByteTrack and 60 recent observations per ID. New script options retain the old defaults 640 / 0 degrees / class-aware NMS. Rotated predicted boxes are mapped back before rendering and CSV export; displayed video remains portrait. No boxes or class names were manually invented or corrected.
- The final `outputs/final_5s.mp4` is H.264/yuv420p, 1080 x 1920, 30 FPS, 150 frames, exactly 5.000000 s, without audio; 8,644,494 bytes; SHA-256 `6f78446e677f3b1b6dece44540c410df6b86f8aa8141ad9c87cdfef54f7972ba`. `outputs/trajectory.csv` is 13,363 bytes, SHA-256 `40c92e65450080dc9d9c78fcdaa04c4961753742545f806e27e705646eb6468b`. `.gitattributes` preserves these verified bytes across platform checkouts.
- `outputs/tracking_verification.json` records 20 passed structural checks: whole-video decoding, frame count, FPS, PTS from 0 to 4.966667 s and final-frame end at 5 s, CSV schema/time/coordinate bounds and unique frame-ID pairs, and unchanged inputs. The CSV-derived yellow trail mask matched expected displayed pixels in all 149 frames with a trail (2-pixel tolerance). Six sampled frames were visually reviewed; `report/assets/tracking/preview.jpg` shows them. This rendering-consistency test is not a detection-accuracy measure.
- The CSV has 300 rows and two IDs, each present in all 150 frames. ID 2 follows the correction tape with the correct class in 150 rows. ID 1 follows the physical eraser, but its predicted class is eraser in 82 rows and highlighter in 68 rows, with 16 class changes. The actual yellow highlighter at the top is missed. The highlighter rows must not be reported as successful highlighter tracking. Camera motion also contributes to these image-coordinate trails; no physical-world motion, MOTA or IDF1 was measured.
- The previous 15 tests and three new orientation/integration tests passed. After the class-agnostic NMS option was added, the eight affected pipeline/orientation tests passed again. `lap` 0.5.14 was installed in the existing fresh local environment and added as an explicit tracking dependency. Original training code, model bytes, dataset and official test metrics are unchanged. The two paused CPU training processes were rechecked against creation times and remain stopped.
- `outputs/artifact_verification.json` is the earlier Colab-import snapshot; its video-pending fields describe that timestamp and now link to the later tracking verification. README and the report distinguish completed output files from the unresolved three-object tracking quality goal. A future improvement should use additional labeled real-object images and a separate evaluation video while preserving this baseline; do not quietly overwrite the model or claim the current clip succeeds on all three objects.

### Real-video GitHub publication verified - 2026-11-20 (KST)

- Published the video, CSV, preview, verification/experiment records, inference options and updated documentation in commit [`29d864994d8d86d4957244c71d417a535b39c04d`](https://github.com/jeongjunseo/yolov8-stationery-tracking/commit/29d864994d8d86d4957244c71d417a535b39c04d). Push succeeded and remote `main` matched; the working tree was clean before this publication note.
- Checked all 101 Git-tracked files against the 100 MiB per-file limit. The largest is the 8,644,494-byte final video. Model, video and CSV Git blob SHA-256 values match the recorded verified files; the user's original input hash is also unchanged. Only the explicit 15-file result change was staged, with no raw input/data/environment files. The CSV's intentional CRLF bytes are preserved, and its whitespace attribute treats CR as part of the line ending.
- All requested artifact files are now public. The documented physical highlighter miss and eraser/highlighter confusion remain unresolved performance limitations; publication does not change `tracking_accuracy_goal_met=false`.

## Authorized video-frame fine-tuning - 2026-11-20 (KST)

The user explicitly requested extracting images from the uploaded video and using them for additional training. This supersedes the earlier instruction not to retrain merely to create the first video. Preserve the completed baseline and all previous artifacts.

- Prepare 48 evenly spaced SDR frames (indices 0, 6, ..., 282) from the real video as a separate train-only addition. Review all three actual object boxes; do not treat the baseline detector's incorrect predictions as ground truth. The physical class interpretation follows visual review of the supplied stationery.
- Mix the new frames with the original 498 training images. Keep original valid 116/test 73 unchanged, and disclose their previously documented limitations. Do not randomly split frames from the same short video into training and purported independent evaluation sets.
- Initialize from the verified baseline `models/best.pt`; train a separate YOLOv8n experiment on Colab CUDA 0, batch 16, 100 epochs, imgsz 640, seed 42 and unchanged remaining training settings. Only actual CUDA OOM permits batch 8/4. Use `models/video_finetune/` and `outputs/video_finetune/`; the baseline files and paused local CPU training must remain intact.
- Re-running this video after fine-tuning measures performance on a training scene, not new-scene generalization. Re-evaluate the unchanged original test set separately to check the effect on prior classes. Training has not yet started at this preparation stage.
- Browser control currently reports `The browser access host is unavailable`; Windows Chrome activation reports `GetCursorPos ... access denied (0x80070005)`. Prepare and verify the notebook/data completely before handing off the execution click if this persists. No Colab training start may be claimed from upload or notebook preparation alone.

### Fine-tuning preparation verified - 2026-11-20 (KST)

- Prepared and visually inspected 48 SDR frames and 144 boxes (48 per class), recorded in `outputs/video_finetune_annotations.json`. These are AI-assisted, visually reviewed annotations, not an independent human annotation audit. Frame 216's correction-tape box was adjusted to include its blue cap. Clipped highlighters and motion-blurred frames are explicitly recorded.
- Initial v1 ZIP/labels are preserved. A strict validation failure at frame 234 exposed 8-decimal floating-point box rounding outside the image edge. The v2 serializer uses 12 decimals and tiny inward adjustments only when necessary; original integer boxes and all JPEG bytes remain unchanged. A regression test covers this boundary case.
- Use only `runs/video_finetune_20261120_v2/video_training_addition_v2.zip`: 17,783,081 bytes, SHA-256 `22a7de84e451f3ad9c120773b74f42111c3955864ed21013d7614e83a980f4c9`. Its 97 members contain 48 image/label pairs and a manifest. CRC, per-file hashes, all-image decoding, original strict label checks and the fine-tuning preparation function passed. Mixed train is 546; valid/test remain 116/73. Original model/metrics/train script/YAML hashes match the baseline.
- Uploaded v2 to private `MyDrive/yolov8-stationery-tracking/video_finetune/video_training_addition_v2.zip` and verified Drive metadata name/size/parent. The old v1 upload remains preserved but the notebook pins the v2 filename and SHA. Colab will independently verify full bytes before training.
- `notebooks/finetune_video_colab.ipynb` creates a fresh UUID checkout and requires CUDA. `scripts/finetune_video.py` trains the separate model, backs up every epoch, evaluates original test data and separately compares both models on the added training-scene frames, then collects only the adaptation artifacts. Best checkpoint selection remains based on original validation; checkpoints are preserved in Drive. Six new tests and four existing batch-policy tests passed; eight notebook code cells plus nested environment code passed AST checks.
- `outputs/video_finetune_setup.json` is preparation evidence, not a GPU run record. Do not create or commit `outputs/video_finetune/` before this first run: the notebook protects existing output folders. Actual GPU allocation, training start and completion are still unverified. The pending step is opening the new fine-tuning Colab notebook, selecting a GPU and running all cells with Drive authorization if prompted.

### Additional Colab training started and verified - 2026-11-20 (KST)

- The user ran all cells. Active run is `20261120T123735Z_2e0735a1`, under private Drive `yolov8-stationery-tracking/video_finetune/colab_runs/`. Run folder: `https://drive.google.com/drive/folders/1VLkM-SS3DYj5BQJPkzsheZLAvnhIaNT_`.
- Downloaded actual environment evidence confirms Tesla T4, CUDA available=True, CUDA 13.0, torch 2.14.1+cu130, ultralytics 8.4.215 and Git commit `009f16b7b3ffa0968eb0b604ff87183bf59e9719`. The pre-existing IPython `jedi` pip-check warning remains non-blocking; do not claim pip check was clean.
- Actual Colab checks verified all 1,376 original archive members and the v2 addition SHA. Mixed train=546, valid=116, test=73, added 48 images/144 labels with original valid/test paths unchanged.
- Epoch 1 completed and its four checkpoint/log artifacts were backed up successfully. `args.yaml` confirms baseline best.pt initialization, device 0, batch 16, epochs 100, imgsz 640, workers 4, seed 42 and deterministic=True. First CSV cumulative time is 125.547 seconds. Completion, final test results and new video are still pending; never present validation metrics as final test metrics.
- Raw monitoring copies are under ignored `runs/video_finetune_monitor/20261120T123735Z_2e0735a1/`. The user-started runtime must not be interrupted or restarted. Browser rendering can be stale/black, so use fresh Drive folder listings and actual backup bytes for progress. Existing local CPU training remains separate and paused.

### Fine-tuning midpoint verified - 2026-11-20 23:04 KST

- The same Colab run completed 50/100 epochs, with cumulative CSV time 4,871.29 seconds and 50 successful backups, zero failures. Batch remains 16. Training and final evaluation are not yet complete; do not restart the notebook. Monitoring snapshots remain under the ignored run directory above.
- Validation mAP50-95 reached 0.79802 at epoch 44, exceeding the initial additional epoch's 0.79337. A downloaded interim best checkpoint confirmed selected epoch 44 (zero-based metadata 43). Six added training frames contained 18 correctly matched detections, zero FP/FN, at imgsz 640/conf 0.25/NMS IoU 0.5 with class-aware NMS. This is only a training-scene interim check, not final test or independent video accuracy. Probe bytes and review records are under `probes/best_update_20261120T135710Z/`; do not publish that interim checkpoint as the final model.
- A local probe encountered torch DLL/oneDNN initialization failures while Windows had little commit-memory headroom. A process-only workaround disabled `torch.backends.mkldnn`, set initial torch/OMP/MKL thread requests to 1, and released unused checkpoint training-state references; the six-frame check then succeeded. Actual thread count after first inference was not recorded and must not be claimed to equal 1. No Colab setting, OS-wide setting, user process or paused CPU training was changed. Final local inference should record the actual backend/thread state if this workaround is needed.
- Future reproduction changes preserve already published fine-tune outputs in a checked sibling snapshot immediately after a fresh clone. `scripts/preserve_published_finetune.py`, its eight passing fixture tests, and the corresponding notebook call are not part of the active Colab checkout. They must not be used on an active checkout. Existing overwrite guards and baseline files remain intact; publishing this helper does not change or complete the running experiment.

### Latest observation: runtime status awaiting confirmation - 2026-11-21 00:31 KST

- Drive has not advanced beyond epoch 97: results/events last modified around 2026-11-20 15:19:29 UTC, last.pt at 15:19:35 UTC. Fresh folder and direct file-metadata reads agree. No training_run.json, completed evaluation or result ZIP has been observed. This does not establish whether the runtime stopped or only backup synchronization stalled. The user was asked for Colab's latest displayed epoch/status and told not to rerun all cells or restart the runtime. Do not start a duplicate run while its state is unknown.
- Root's existing native sky/window binding can read Chrome accessibility, but its execution text remains the initial 37-second state with outdated RAM/disk usage. It is not current training evidence. A new UI-tool initialization failed with `failed to write kernel assets ... os error 3`; ordinary filesystem tools and authenticated Drive downloads still work. No browser input or runtime control was performed.
- A separate immutable recovery copy is under `runs/video_finetune_monitor/20261120T123735Z_2e0735a1/recovery_snapshot_20261120T1528Z/` on D:. last.pt is 31,788,199 bytes, SHA-256 `c336147c23c3acef1fe55e39a603cb3afb79e8c82bb4f66bca4be0ef18bb8551`; best.pt is 31,786,279 bytes, SHA-256 `61e35e27335a72841e5ded5075d1e05d76460a0aa4b03590c0c656b6f9121976`. Matching CSV, args and 97 successful backup events are preserved with download metadata.
- Metadata-only `torch.load(map_location="meta")` verified last.pt epoch 96 (zero-based), meaning 97 completed epochs; its internal 97-row training history matches CSV epochs 1-97. Optimizer, EMA and AMP scaler are present. best.pt is selected epoch 82, validation fitness 0.83085. All five input file hashes remained unchanged. Validation details are in checkpoint_metadata_validation.json beside these copies. These are interim checkpoints, not final 100-epoch artifacts.
- If an interruption is actually confirmed, use a separately reviewed resume path from the matching last.pt, retaining total epochs=100, to perform only epochs 98-100. Do not call the existing finetune_video.py train again: it starts from baseline and initializes new backup history. A correct recovery must preserve the 97 CSV rows/events and prior best.pt, restore data/settings/GPU, and record the interruption. Ultralytics resume does not automatically restore CSV and resets its time origin; model/optimizer argument resolution also differs from the initial args.yaml. Existing completion/import checks therefore need explicit resume provenance handling. No recovery execution or code change has been performed.

### Disconnected Colab and recovery preparation - 2026-11-21 (KST)

- The user reported that the runtime appeared stopped. A fresh native accessibility read now shows `Reconnect GPU` on the original fine-tune notebook. A reconnect click failed with Windows `GetCursorPos` access denied; no runtime restart, Run all, stop, or training action was executed. The user was asked to click only Reconnect and report whether the original training continues. A disconnected browser does not by itself establish that the remote training process has terminated.
- Fresh Drive metadata at approximately 13:02 KST still shows last.pt modified at 2026-11-20T15:19:35.115Z, 97 backup events, and no training_run.json or final ZIP. The verified immutable local snapshot remains unchanged. The paused Windows CPU training was not touched.
- A separate `notebooks/resume_video_finetune_colab.ipynb` and `scripts/resume_video_finetune.py` recovery path preserve the original Drive folder and use a unique new checkout/Drive run. The notebook pins the five verified source file hashes, restores the original dataset and 48-frame addition, and rechecks source hashes immediately before recovery. Same-runtime process checks and changed-source guards reject observable duplicate execution; they cannot replace checking an original job running on another VM.
- Recovery retains total epochs=100, CUDA 0 and batch 16, restores CSV epochs 1-97 and the prior best checkpoint, and uses resumable last.pt for epochs 98-100. It preserves optimizer/EMA/scaler state and resolved training settings. CSV time continues from 9,447.17 active seconds, excluding the interruption. Bitwise equivalence to uninterrupted RNG progression is not claimed. Original and recovery backup events/provenance are kept distinct.
- The recovery notebook then uses the existing original-test evaluation, explicitly labeled training-scene recheck, and artifact collection commands. A separate Drive collection_receipt.json will record the actual collected ZIP SHA and confirm the Drive copy matches. Preparation and fixture checks are not evidence of a real GPU resume: actual recovery execution, 100-epoch completion, final test scores and new video are still pending.
- Verification: six recovery guard/state fixtures passed with `python -m unittest discover -s tests -p test_resume_video_finetune.py -v`. All eight notebook code cells and the embedded environment program passed syntax checks; mocked runs checked actual newline serialization, exclusive file staging, rerun/live-process rejection and changed-source rejection before training. The new source validator also accepted the actual pinned last/best checkpoints and matching history while all source hashes remained unchanged. CLI help matches the notebook's prepare/train calls. These checks did not execute GPU training.

### User confirmed interruption; GPU quota unavailable - 2026-11-21 18:38 KST

- The user clicked Reconnect and explicitly confirmed that training remained stopped, then reported that GPU allocation was unavailable because usage quota was exhausted. The native Chrome view showed a connected runtime, but no GPU allocation warning was visible in the captured accessibility text; the quota diagnosis is the user's report. The original reason for the 97-epoch interruption remains unknown.
- Fresh Drive last.pt metadata still matches the 97-epoch backup (31,788,199 bytes, modified 2026-11-20T15:19:35.115Z). No recovery was started, no CPU training was resumed, and no paid compute was purchased. The recovery notebook is published in commit `8f273274a47735f6cdfe51cbd44c36ccadd4ba23`; remote main and the public notebook readback were verified.
- An optional user decision is pending: retain the 100-epoch target and resume the last three when GPU becomes available, or explicitly end this additional experiment at 97 completed epochs and report that limitation. Do not silently claim 100 epochs or relax final completion checks without the user's decision. Existing baseline 100-epoch results remain unaffected.
- Meanwhile a separate, ignored CPU-inference preview uses the preserved epoch-82 best checkpoint (selected from the 97 completed epochs). This advances the actual source-video check without retraining or replacing any published final artifact. It is a training-scene preview, not independent video accuracy and not evidence of completed 100-epoch fine-tuning.
- The reviewed ignored importer now accepts explicitly verified resume provenance and preserves its previous version. `runs/video_finetune_20261120_v2/import_finetune_artifacts.py` passed 10 synthetic schema checks; a real recovered ZIP still requires its new run ID and the separate Colab collection_receipt.json checksum. No final ZIP was imported.
- The interim CPU preview completed in the new ignored `runs/video_finetune_20261121_quota_preview/` directory. H.264 MP4 is 8,597,224 bytes, SHA-256 `c0c125466a4accbc12c46950edf78e207207062d539a4366a20057cf33330d3b`; CSV has 450 rows (150 per class), with correction_tape ID 1, eraser ID 2 and highlighter ID 3 continuously present. All 20 file/time/CSV checks passed; 25 existing labeled frames gave 75/75 class-matched centers inside their GT boxes. This is a centers-only training-scene check, not IoU or independent tracking accuracy. Root and the delegated reviewer visually inspected the six-frame contact sheet. Actual inference threads=8, interop=6, mkldnn=False; initial request=1 is not reported as the measured thread count.
- A byte-verified copy of the interim MP4/CSV/contact sheet and a compact provenance/verification JSON is under `outputs/video_finetune/interim_best82_*` and `report/assets/video_finetune/interim_best82_preview.jpg`. These explicitly labeled interim files do not replace any baseline or final fine-tune output. Snapshot/source hashes stayed unchanged. No final original-test metrics have been measured for this checkpoint yet.

### User decision: keep 100 epochs - 2026-11-21 19:15 KST

- The user explicitly selected **retain 100 epochs and resume the remaining three after GPU reallocation**. The earlier optional decision is resolved. Do not finalize the additional experiment at 97 epochs or promote the interim preview to a completed 100-epoch result.
- Keep the verified epoch-97 last.pt, prior epoch-82 best, CSV and backup history intact. When Colab CUDA is available, use the published recovery notebook for epochs 98-100, then run final original-test evaluation, final-model tracking and report/artifact verification. The user has authorized this continuation; no repeated approval of the 100-epoch plan is needed.
- GPU reallocation is still pending based on the latest user report. No new runtime or recovery execution was started in recording this decision, and the local CPU training remains untouched.

### Requested current-model full-video recheck - 2026-11-21

- The user requested immediate video verification using the additional training completed so far. Executed a NEW 287-frame CPU inference using preserved best82 from the 97 completed epochs. This request did not execute more training or establish 100-epoch completion. Original data, model/source snapshots, paused CPU training and earlier public artifacts were preserved.
- New ignored run: `runs/video_finetune_20261121_recheck_104933Z/`. Full H.264 video is 17,522,497 bytes, SHA `eb4fe8994fe1225977cefc146e923896c7d4efb5aa769cb306504ad92e2f6e7e`; first-150-frame 5s H.264 output is 8,597,224 bytes, SHA `c0c125466a4accbc12c46950edf78e207207062d539a4366a20057cf33330d3b`, reproducing the earlier preview exactly after new inference. Full CSV has 859 rows, first5s CSV 450; full/5s file checks passed 9/9 and 20/20. Raw returned post-tracking boxes and runtime measurements are preserved.
- All 48 reviewed training frames were checked using class-aware IoU>=0.5 one-to-one matching: TP144/FP1/FN0, fixed-threshold micro P=0.9931034483/R=1.0. These are post-tracking training-scene sample metrics, not AP, independent test or full-video accuracy. Three matcher boundary fixtures passed, and the existing project matcher agreed for all 48 frames.
- Full-video inspection found highlighter output absences at frames 212-215 (mostly clipped/outside image) and 219 (visible but no returned tracking box), with IDs 3->10->11. Do not count all five as ground-truth FN or attribute frame219 specifically to the detector: pre-tracking raw detections were not captured. Actual correction_tape ID1 and eraser ID2 persisted across all287 frames. Background floor/mat boundary false positives appeared as correction_tape ID12 at276/277/279; this is not eraser misclassification. Only276 is in the 48 labeled sample, explaining the sample FP1.
- Public copies use `outputs/video_finetune/recheck_5s.mp4`, `recheck_full.mp4`, corresponding CSVs, `recheck_inference.json`, `recheck_labeled_frames.json`, and `video_recheck.json`. Full/ID-gap/background-FP contact sheets are under `report/assets/video_finetune/recheck_*.jpg`. Root and both delegated reviewers inspected the relevant contacts; no independent human audit is claimed. The user's 100-epoch continuation decision is retained, while this immediate video recheck is complete.

### Submission audit - 2026-11-21

- The user requested a check against assignment conditions. The GitHub connector confirmed public repository visibility; local HEAD and remote main both matched `2f9c569dce985290f13480d058d5abe2fe556e94` at audit start. All baseline deliverables are Git tracked: 100-epoch model and CSV, original-test metrics, exact-five-second video and trajectory, training/test graphs, README and report. The baseline model SHA still matches its completed Colab record.
- Direct ffprobe and CSV checks confirmed the public improved clip is H.264/yuv420p, 1080x1920, 30 FPS, 150 frames, exactly 5 seconds, with 450 corresponding rows and three IDs. Its video/CSV hashes match the recheck record. This best82 checkpoint is a different model from public `models/best.pt`; the baseline test numbers must not be attributed to it.
- The additional run remains 97/100. Final `models/video_finetune/best.pt`, `outputs/video_finetune/test_metrics.json`, and additional `training_run.json` are absent. The user's 100-epoch decision remains in force; this audit performed no training or inference. School submission-system receipt/upload evidence is unavailable, so only GitHub publication is confirmed.
- All 129 tracked files are below 100 MiB; largest is `outputs/video_finetune/recheck_full.mp4`, 17,522,497 bytes. README/report were clarified to distinguish baseline completion from additional-experiment completion and to correct the stale maximum-file-size statement. No model, dataset, source video or prior result was changed.
