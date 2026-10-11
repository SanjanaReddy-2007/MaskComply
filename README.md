# MaskComply

**Temporal Face Mask Compliance Scoring Using Lightweight Detection and Tracking**

A rule-based scoring layer on top of a face detector that turns per-face
mask classifications into a continuous, temporally-smoothed compliance
score (0.0–1.0) per frame/session — for facility monitoring dashboards,
not just per-face labels.

---

## Project structure

```
MaskComply/
├── README.md
├── requirements.txt
├── .gitignore
├── validation_config.json      ← made by build_validation_config.py; fill in the human ratings (Day 4)
│
├── data/
│   ├── kaggle_raw/              ← andrewmvd/face-mask-detection (images/ + annotations/); not in git
│   ├── yolo_dataset/            ← made by convert_voc_to_yolo.py (train/val + data.yaml); not in git
│   ├── shiekhburhan_raw/        ← classification-only data for the legacy classifier; not in git
│   │   └── FMD_DATASET/{incorrect_mask,with_mask,without_mask}/
│   └── test_clips/              ← test footage for the pipeline and Day 4 validation
│
├── models/
│   ├── best.pt                  ← fine-tuned YOLOv8 detector (default; train it, see below)
│   ├── mask_classifier.pt       ← legacy CNN classifier weights (--detector haar only)
│   └── classes.json             ← class-name mapping for the legacy classifier
│
├── src/
│   ├── detection.py              ← shared detector: YOLOv8 (default) or legacy Haar+CNN
│   ├── detect.py                 ← single-image detect, draws box/label/confidence
│   ├── train_classifier.py       ← fine-tunes classifier on real labeled data
│   ├── verify_classifier.py      ← per-class accuracy + spot-check tool
│   ├── tracker.py                ← centroid tracker with track confirmation
│   ├── scorer.py                 ← EMA scoring formula + threshold bands
│   ├── nms.py                    ← duplicate-detection fix (see Day 2 bug below)
│   ├── pipeline_video.py         ← full detect→track→score pipeline for video
│   ├── dashboard.py               ← Streamlit live dashboard + alert logging
│   ├── validate.py                ← Day 4: correlate automated vs human ratings
│   ├── build_validation_config.py ← scaffolds validation_config.json
│   ├── build_synthetic_test.py    ← synthetic dropout stress-test generator
│   ├── train_yolo.py              ← YOLOv8n 3-class fine-tuning
│   └── convert_voc_to_yolo.py     ← Kaggle VOC XML → YOLO format + train/val split + data.yaml
│
├── outputs/                      ← generated results: scored_video.avi, scored_video_frames/,
│                                   compliance_log.csv, alert_log.csv, validation_result.json
└── docs/
    ├── DAY1_README.md
    └── DAY2_README.md
```

---

## Pipeline

```
video/image → YOLOv8 detector (finds each face + mask / no_mask / incorrect)
    → tracker (persistent IDs, filters phantom detections)
    → per-person EMA scorer → frame/session compliance score
    → dashboard + alert log
```

**Scoring formula** (exact, from the original blueprint):

```
class_weight = 1.0 (mask) / 0.5 (incorrect) / 0.0 (no_mask)
raw_score(i) = class_weight(i) × confidence(i)
EMA_score(p, t) = α·raw_score(p, t) + (1-α)·EMA_score(p, t-1)
Frame_Compliance_Score = mean(EMA_score(p, t) for all tracked p in frame)

Bands:  Green ≥ 0.8   |   Yellow 0.5–0.8   |   Red < 0.5
```

**When nobody is in the frame there is no score.** The frame score is only
computed over people currently tracked, so an empty scene shows "No people
detected" (and a blank score in the CSV) instead of red. A person who leaves
keeps their last score for `--max_missed` frames (default 10) before it
clears; that hold is what smooths over brief misdetections. A red score on a
scene with no real people means the detector produced a false box (see
limitation 2).

---

## Setup

```bash
pip install -r requirements.txt
```

This installs everything the YOLO pipeline needs (including `ultralytics`).
Training the detector needs a GPU, so it is easiest on Google Colab; running
the pipeline on a CPU works but is slow (add `--imgsz 640` for a quicker look).

You need a Kaggle API key (kaggle.com → Settings → API → create a legacy
token, saved as `~/.kaggle/kaggle.json`; on Windows `C:\Users\<you>\.kaggle\`)
to download the datasets:

- **`andrewmvd/face-mask-detection`** (bounding boxes) — trains the YOLO
  detector. Used in Quick start step 1. This is the one you need.
- `shiekhburhan/face-mask-dataset` (cropped faces, no boxes) — only for the
  legacy classifier (`--detector haar`):
  ```bash
  kaggle datasets download shiekhburhan/face-mask-dataset
  unzip face-mask-dataset.zip -d data/shiekhburhan_raw
  ```

`unzip` does not exist in Windows PowerShell; use
`Expand-Archive -Path file.zip -DestinationPath folder` there.

---

## Quick start

```bash
# 1. Get the bounding-box dataset and convert it to YOLO format
kaggle datasets download andrewmvd/face-mask-detection
unzip face-mask-detection.zip -d data/kaggle_raw        # Windows: Expand-Archive
python src/convert_voc_to_yolo.py --raw_dir data/kaggle_raw --out_dir data/yolo_dataset
#    (optional) --negatives_dir data/negatives   <- frames with no faces: windows, lights, signs

# 2. Train the detector (use a GPU, e.g. Colab). The path of the finished
#    best.pt is printed at the end of training (recent Ultralytics versions use
#    runs/detect/...; the Colab `yolo detect train` command uses
#    runs/detect/train/weights/best.pt). Copy it to models/best.pt, which is
#    where the other scripts look by default.
python src/train_yolo.py --data data/yolo_dataset/data.yaml --epochs 50 --imgsz 960

# 3. Run on a single image
python src/detect.py --image path/to/photo.jpg --yolo_weights models/best.pt

# 4. Run the full detect+track+score pipeline on a video
python src/pipeline_video.py --video path/to/video.mp4 --yolo_weights models/best.pt

# 5. Live dashboard (keep the -- before your own args; Streamlit needs it)
streamlit run src/dashboard.py -- --video path/to/video.mp4 --yolo_weights models/best.pt

# Useful options on every script above:
#   --conf 0.4      minimum detection confidence (raise it to cut false boxes)
#   --imgsz 960     YOLO inference size
#   --save_frames 30  (pipeline_video.py) also save every 30th annotated frame as a JPG
#   --max_missed 10   (pipeline_video.py) frames a person's score is held after they vanish
#   --detector haar use the old Haar cascade + CNN classifier instead
#                   (needs opencv-python<5 and models/mask_classifier.pt)

# Legacy classifier (only needed for --detector haar)
python src/train_classifier.py --data_dir data/shiekhburhan_raw/FMD_DATASET
python src/verify_classifier.py --data_dir data/shiekhburhan_raw/FMD_DATASET
```

---

## Day 4 — Validation

This is how you turn "a working pipeline" into a validated result for the
paper.

1. **Get 3–5 real, independent short video clips.** Not the training
   data, not synthetic — genuine footage. Recording your own short
   phone/webcam clips is the recommended approach: it is cheap, has no
   licensing questions, and matches the deployment scenario directly. Put
   them in `data/test_clips/`.

2. **Scaffold the config:**
   ```bash
   python src/build_validation_config.py --clips_dir data/test_clips
   ```
   This writes `validation_config.json` with a `null` placeholder for
   each clip's human rating.

3. **Watch each clip yourself** (or with 1–2 others) and fill in a real
   0.0–1.0 compliance rating in place of each `null`.

4. **Run validation:**
   ```bash
   python src/validate.py --config validation_config.json --yolo_weights models/best.pt
   ```
   This prints Pearson and Spearman correlation between your automated
   scores and human ratings, and saves the full result to
   `outputs/validation_result.json`.

**Important:** a mechanism test was run during development using 3
deliberately distinct synthetic clips built from training images
(Pearson r=0.998) — this number is real but scientifically meaningless
for the report, since it's circular (clips built from training data) and
the sample is trivially separable. Do not report that number. Run it
again on your own real, independent clips before writing up results.

---

## Known limitations — read before writing the report

**1. Detector is a first fine-tune on a small dataset.** The default
detector is YOLOv8n fine-tuned on the Kaggle `andrewmvd/face-mask-detection`
set (853 images, 3 classes). Held-out validation (171 images, imgsz 960):
overall mAP50 0.860; mask 0.982, no_mask 0.879, incorrect 0.720. Two caveats
for the report: (a) `incorrect` was measured on only 19 examples, so that
figure is rough; (b) these images are mostly clear, close-up faces, so
accuracy on small, distant, blurry faces in street footage is expected to be
lower and has not been measured -- that needs labeled frames from your own
clips. `no_mask` recall is 0.793, so about 1 in 5 bare faces can be missed,
which makes the compliance score read slightly high.

**2. The earlier Haar + CNN path had real false-positive problems** (boxes on
windows, lights and clothing, each scored as `no_mask`) and misses masked,
turned or small faces. It is kept behind `--detector haar` for comparison
only. Adding background (no-face) frames from your clips via
`convert_voc_to_yolo.py --negatives_dir` and retraining is the planned fix
for any remaining false boxes in the YOLO detector.

**3. `mask` vs `incorrect` confusion.** This was measured on the legacy CNN
classifier (`--detector haar`): per-class accuracy mask=89.7%, no_mask=98.9%,
incorrect=92.2%. The YOLO detector's own per-class numbers are in item 1,
and its weakest class is also `incorrect`. The errors that do occur are concentrated between `mask` and `incorrect`
specifically (confirmed via spot-check), which makes sense — a mask worn
slightly wrong looks visually similar to one worn correctly, while a bare
face looks nothing like either. Worth stating directly in the report's
limitations section as an expected, explainable failure mode.

**4. A real bug was found and fixed during Day 2 testing**, worth
mentioning in the report as a legitimate engineering finding: Haar
cascade emitted duplicate overlapping boxes for the same face on noisy
frames, causing the tracker to spawn phantom "people" that diluted the
frame-level compliance average. Fixed via IoU-based deduplication
(`nms.py`) + track confirmation (a track only counts once matched
consistently for 3 frames, sticky across brief dropouts). Verified via a
synthetic stress test: the target person's EMA held rock-steady through
a simulated 4-frame misdetection after the fix.

**5. OpenCV video codec gotcha.** OpenCV's `mp4v` output often appears
blank in standard players, and `XVID` `.avi` files also failed to decode on
at least one Windows machine (VLC showed only an icon). `pipeline_video.py`
now writes `MJPG` in an `.avi`, which uses OpenCV's built-in encoder and
needs no extra codecs. If a video still will not play, run with
`--save_frames 30` to get JPG snapshots of the annotated frames instead.
(`build_synthetic_test.py` still uses `XVID`.)

**6. Footage licensing.** The clips in `data/test_clips/` look like stock
footage. Check each one's licence before the repository is shared or the
clips are redistributed, and cite their sources in the report. Your own
recordings avoid the question.

---

## Day-by-day build log

See `docs/DAY1_README.md` and `docs/DAY2_README.md` for the detailed,
honest account of what was built each day, what didn't work on the first
try, and how it was fixed. Those two files describe the earlier Haar + CNN
version, with `scripts/` paths that are now `src/`; the current default is
the YOLOv8 detector described above.

**Detector upgrade (after Day 4 setup).** Replaced the Haar cascade with a
YOLOv8n detector fine-tuned on `andrewmvd/face-mask-detection`
(`convert_voc_to_yolo.py` → `train_yolo.py`), behind a shared interface in
`detection.py`. Also added: `--conf`/`--imgsz`/`--max_missed`/`--save_frames`
options, a "No people detected" state, an `MJPG` video writer, and a
dashboard fix for current Streamlit versions.