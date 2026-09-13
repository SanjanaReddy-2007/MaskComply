# MaskComply — Day 1

## What actually ran today (all real, not simulated)

1. **Environment**: fixed a corrupted torch install eating disk space, then
   installed a clean CPU-only stack — `torch 2.14`, `torchvision`,
   `ultralytics 8.4` (YOLOv8), `opencv-python`.
2. **YOLOv8n**: pretrained COCO weights downloaded and load correctly
   (`yolov8n.pt`). Kept for Day 2+ once the full 3-class dataset is available.
3. **Dataset**: Kaggle's "Face Mask Detection" (the one named in the
   blueprint) needs a Kaggle API key — not reachable from this sandbox.
   Substituted `chandrikadeb7/Face-Mask-Detection`'s bundled dataset
   (github.com, no auth needed): **4,095 real face-crop images**,
   `with_mask` / `without_mask` only — **no "incorrect" class**.
4. **Face localization**: OpenCV Haar Cascade (pretrained, ships with
   opencv, no training needed).
5. **Mask classifier**: small CNN, **trained from scratch today** on the
   4,095 real images, 4 epochs, CPU. Result: **95.1% validation accuracy**.
   Weights: `models/mask_classifier.pt`.
6. **End-to-end test**: ran face-detect → classify → draw box + label +
   confidence on two real street-photo test images.
   - `outputs/pic1_annotated.jpg` — 1 face detected, correctly labeled
     `mask 1.00`.
   - `outputs/pic2_annotated.jpg` — **3 faces** detected in a crowd shot,
     correctly labeled `mask 1.00` (center) and `no_mask 0.99` / `0.97`
     (sides).

This hits the Day 1 checkpoint from the blueprint: **detector correctly
draws boxes and labels on a test image/video**, with real confidence
scores — ready for Day 2's tracker + EMA scoring layer to consume.

## Known gaps to fix honestly, not hide

- **Only 2 classes, not 3.** No "incorrect-wear" data was reachable today.
  Get the Kaggle dataset locally (needs a free Kaggle account + API key)
  to add the third class — see `scripts/train_yolo.py` docstring for exact
  steps.
- **Haar cascade misses some faces**, especially at odd angles or when a
  mask obscures more of the face geometry it looks for (missed 1 of 2 faces
  in the first test image). A YOLOv8n face detector fine-tuned on the real
  dataset will be materially more robust — that's what `train_yolo.py` is
  for, once you have the annotated data.
- **CNN classifier is basic** (4 conv blocks, 4 epochs) — good enough to
  prove the pipeline today; swap in MobileNetV3 fine-tuning for the real
  report numbers, per the blueprint's tech stack.

## Files

```
scripts/detect.py           — face detect + classify + draw (run today)
scripts/train_classifier.py — trained the classifier on real data (run today)
scripts/train_yolo.py       — YOLOv8n 3-class fine-tuning (ready, needs Kaggle data)
models/mask_classifier.pt   — trained weights (95.1% val acc)
outputs/*.jpg                — annotated test results
```

## Run it yourself

```bash
python scripts/detect.py --image path/to/photo.jpg \
    --weights models/mask_classifier.pt --output outputs/result.jpg
```

## Tomorrow (Day 2, per blueprint)

- Add centroid/IOU tracker so each face gets a persistent ID across frames.
- Implement `class_weight(i) × confidence(i)` → EMA smoothing per tracked
  person → frame-level mean. The classifier here already outputs exactly
  the (class, confidence) pair the formula needs.
