"""
Verifies the trained mask classifier: confirms classes.json matches the
model's output layer, reports per-class validation accuracy (not just
overall accuracy -- overall can look fine while one class, e.g.
"incorrect", is being predicted badly), and runs it on a few real sample
images per class so you can eyeball actual predictions, not just numbers.

Usage (run from repo root):
    python src/verify_classifier.py --data_dir data/raw/shiekhburhan_raw/FMD_DATASET \
        --weights models/mask_classifier.pt
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from detect import MaskClassifier, load_class_names, classify_face
from train_classifier import TRANSFORM
import cv2


def main(data_dir, weights_path):
    classes_path = Path(weights_path).with_name("classes.json")
    if not classes_path.exists():
        print(f"ERROR: {classes_path} not found. Was this model trained with "
              f"the updated train_classifier.py?")
        return

    saved_class_names = load_class_names(weights_path)
    print(f"classes.json says (index order): {saved_class_names}")

    full_dataset = datasets.ImageFolder(data_dir, transform=TRANSFORM)
    print(f"ImageFolder on disk finds (index order): {full_dataset.classes}")

    if len(saved_class_names) != len(full_dataset.classes):
        print(f"MISMATCH: model has {len(saved_class_names)} classes, "
              f"dataset on disk has {len(full_dataset.classes)}. "
              f"Did the data_dir change since training?")
        return

    model = MaskClassifier(num_classes=len(saved_class_names))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # --- per-class accuracy on the full dataset (not just an overall number) ---
    loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=2)
    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = outputs.argmax(1)
            for label, pred in zip(labels, preds):
                total_per_class[label.item()] += 1
                if label.item() == pred.item():
                    correct_per_class[label.item()] += 1

    print("\nPer-class accuracy (full dataset, includes training data --")
    print("indicative only, not a true held-out validation number):")
    for idx, folder_name in enumerate(full_dataset.classes):
        canon = saved_class_names[idx]
        total = total_per_class[idx]
        correct = correct_per_class[idx]
        acc = correct / total if total else 0
        print(f"  [{idx}] {folder_name:20s} -> {canon:10s}  "
              f"{correct}/{total} = {acc:.3f}")

    # --- spot-check a few real random samples per class ---
    print("\nSpot-check (3 random real images per class, model's actual prediction):")
    samples_by_class = defaultdict(list)
    for path, label in full_dataset.samples:
        samples_by_class[label].append(path)

    for idx, folder_name in enumerate(full_dataset.classes):
        canon = saved_class_names[idx]
        paths = random.sample(samples_by_class[idx], min(3, len(samples_by_class[idx])))
        print(f"\n  Class [{idx}] {folder_name} (true label: {canon}):")
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            pred_label, conf = classify_face(model, saved_class_names, img)
            flag = "OK" if pred_label == canon else "MISCLASSIFIED"
            print(f"    {Path(p).name:30s} -> predicted {pred_label:10s} "
                  f"({conf:.3f})  [{flag}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/shiekhburhan_raw/FMD_DATASET")
    parser.add_argument("--weights", default="models/mask_classifier.pt")
    args = parser.parse_args()
    main(args.data_dir, args.weights)