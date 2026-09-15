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
├── validation_config.json      ← you create this (see Day 4 below)
│
├── data/
│   ├── shiekhburhan_raw/
│   │   └── FMD_DATASET/
│   │       ├── incorrect_mask/
│   │       ├── with_mask/
│   │       └── without_mask/
│   └── test_clips/              ← your real Day 4 test footage goes here
│
├── models/
│   ├── mask_classifier.pt       ← trained weights
│   ├── classes.json             ← auto-generated class-name mapping
│   └── yolov8n.pt                ← pretrained base (not fine-tuned, see limitations)
│
├── src/
│   ├── detect.py                 ← face detect + classify, draws box/label/confidence
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
│   ├── train_yolo.py              ← YOLOv8n 3-class fine-tuning (needs bbox data — not yet run)
│   └── convert_voc_to_yolo.py     ← VOC XML → YOLO format converter (for train_yolo.py)
│
├── outputs/                      ← generated results (annotated media, CSV logs)
└── docs/
    ├── DAY1_README.md
    └── DAY2_README.md
```

---

## Pipeline

```
video/image → face detector (Haar cascade) → classifier (CNN)
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

---

## Setup

```bash
pip install -r requirements.txt
```

Get a Kaggle API key (kaggle.com → Settings → Create New Token → save as
`~/.kaggle/kaggle.json`), then:

```bash
kaggle datasets download shiekhburhan/face-mask-dataset
unzip face-mask-dataset.zip -d data/shiekhburhan_raw
```

---

## Quick start

```bash
# Train the classifier (real 3-class data, ~93% validation accuracy achieved)
python src/train_classifier.py --data_dir data/shiekhburhan_raw/FMD_DATASET

# Sanity-check it (per-class accuracy + spot-check predictions)
python src/verify_classifier.py --data_dir data/shiekhburhan_raw/FMD_DATASET

# Run on a single image
python src/detect.py --image path/to/photo.jpg --weights models/mask_classifier.pt

# Run the full detect+track+score pipeline on a video
python src/pipeline_video.py --video path/to/video.mp4 --weights models/mask_classifier.pt

# Live dashboard (note the -- before your own args)
streamlit run src/dashboard.py -- --video path/to/video.mp4 --weights models/mask_classifier.pt
```

---

## Day 4 — Validation

This is how you turn "a working pipeline" into a validated result for the
paper.

1. **Get 3–5 real, independent short video clips.** Not the training
   data, not synthetic — genuine footage. Recording your own short
   phone/webcam clips is the recommended approach (see rationale in the
   project chat log) — cheap, no licensing issues, and matches the
   deployment scenario directly. Put them in `data/test_clips/`.

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
   python src/validate.py --config validation_config.json --weights models/mask_classifier.pt
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

**1. Detector is not fine-tuned.** The blueprint specifies a single-stage
YOLOv8n detector fine-tuned on 3 classes. What's actually running is
OpenCV's off-the-shelf Haar cascade (face location, not fine-tuned) +
a separately fine-tuned CNN classifier (mask/no_mask/incorrect). This
happened because every dataset actually reachable during development
(Kaggle's `andrewmvd/face-mask-detection` was inaccessible; the GitHub
substitute and `shiekhburhan/face-mask-dataset` used instead) is
classification-only — folders of cropped face images, no bounding-box
annotations. `train_yolo.py` and `convert_voc_to_yolo.py` are written
and ready; they just need a bbox-annotated dataset to actually run.

**2. Haar cascade misses faces sometimes.** It's not trained for masked
faces specifically and can miss detections at odd angles or when a mask
obscures the geometry it looks for. This is the real motivation for
item 1 above — a fine-tuned YOLOv8n would be materially more robust.

**3. `mask` vs `incorrect` confusion.** The trained classifier's
per-class accuracy is mask=89.7%, no_mask=98.9%, incorrect=92.2%. The
errors that do occur are concentrated between `mask` and `incorrect`
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

**5. OpenCV video codec gotcha.** Videos written by earlier versions of
these scripts used the `mp4v` fourcc, which produces `.mp4` files that
often appear blank/unplayable in standard players (a well-known OpenCV
issue) despite containing valid frame data. Fixed by switching to `XVID`
fourcc with `.avi` containers throughout.

---

## Day-by-day build log

See `docs/DAY1_README.md` and `docs/DAY2_README.md` for the detailed,
honest account of what was built each day, what didn't work on the first
try, and how it was fixed.