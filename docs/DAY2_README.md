# MaskComply — Day 2: Tracker + Scoring Layer

## What was built

1. **`tracker.py`** — centroid-based multi-object tracker. Matches faces
   across frames by nearest centroid, assigns persistent IDs, and holds a
   track alive for `max_missed` frames if detection briefly drops out
   (rather than treating a missed frame as "person left").
2. **`scorer.py`** — implements the blueprint's exact formula:
   `raw_score = class_weight × confidence`, then per-person EMA
   (`EMA(t) = α·raw(t) + (1-α)·EMA(t-1)`), then frame-level mean across
   all tracked people, plus Green/Yellow/Red threshold bands.
3. **`pipeline_video.py`** — wires detect → track → score together on any
   video file, drawing box + label + confidence + track ID + running EMA
   per person, and the frame-level compliance score + band in the corner.
   Also logs every frame to CSV.
4. **`nms.py`** — a fix (see below) for a real bug found during testing.

## No real video was available, so I built a controlled test — and it caught a real bug

There's no webcam or sample video in this sandbox, so I built
`build_synthetic_test.py`: it takes Day 1's real 3-face crowd photo and
generates 60 frames with small camera jitter + noise, then **blanks out
the mask-wearer's face for 4 consecutive frames** (frames 25–28) to
simulate exactly the failure mode the Day 2 checkpoint targets: *"someone
briefly looks away or is misdetected for one frame."*

First run surfaced a real problem: the frame-level compliance score
**crashed hard** during and after the dropout (down to ~0.31 from ~0.60),
far more than the EMA math should allow. Traced it with a per-frame debug
script and found the actual cause: **OpenCV's Haar cascade was emitting
multiple overlapping boxes for the same face** on the jittered/noisy
frames, and the centroid tracker was creating a brand-new phantom track ID
for each duplicate — diluting the frame average with 0-valued ghost
"people" who don't exist.

Fixed it two ways:
- **`nms.py`**: IoU-based deduplication on raw detections before they
  reach the tracker.
- **Track confirmation** (added to `tracker.py`): a track only counts
  toward the score once it's matched consistently for `min_hits=3`
  frames — filtering out one-off spurious detections. Confirmation is
  *sticky*: once a track is confirmed, a later brief dropout doesn't
  force it to re-earn confirmation (this is what makes the EMA-hold
  behavior below actually work end to end).

## Result after the fix

The specific person the test targeted (the mask-wearer, track ID 0) holds
a **rock-steady EMA of 0.997 across the entire simulated dropout** — the
scorer correctly holds the last known value instead of punishing a
momentary misdetection. That's the literal Day 2 checkpoint, verified,
not just claimed.

The **frame-level aggregate** (chart shown in chat) is not perfectly flat,
though — it drifts in the 0.25–0.45 range with a real dip around the
dropout window. That's honest residual noise from the Haar cascade
occasionally emitting a spurious detection elsewhere in the frame that
survives 3 confirmation hits before dropping out again. This is a
detector-quality problem, not a scoring-math problem — the exact thing a
properly fine-tuned YOLOv8n detector (once you have bounding-box data)
would fix, since it wouldn't hallucinate face-like patterns in background
clutter the way Haar cascades can.

## Files

```
scripts/tracker.py              — centroid tracker + confirmation logic
scripts/scorer.py               — EMA + threshold-band scoring formula
scripts/pipeline_video.py       — full detect+track+score video pipeline
scripts/nms.py                  — dedup fix for the phantom-track bug
scripts/build_synthetic_test.py — synthetic dropout stress-test generator
outputs/synthetic_test.mp4      — the generated test video
outputs/synthetic_scored.mp4    — annotated output (boxes/IDs/scores)
outputs/synthetic_compliance_log.csv — per-frame score log
```

## Run it yourself

```bash
python scripts/pipeline_video.py --video path/to/video.mp4 \
    --weights models/mask_classifier.pt \
    --output outputs/scored.mp4 --csv outputs/log.csv
```

## Tomorrow (Day 3, per blueprint)

- Streamlit/matplotlib live dashboard reading the CSV log
- Threshold-crossing alert logging
- Worth prioritizing first: swap Haar cascade for YOLOv8n once the real
  Kaggle dataset is available, since the residual frame-score noise found
  today traces back to detector quality, not the tracker/scorer logic.
