"""
Day 2/3 update — Fine-tune the mask classifier on real data.

Dataset (updated): shiekhburhan/face-mask-dataset (Kaggle) — 14,535 real
images across 3 classes: with_mask, without_mask, incorrect (mask-on-chin /
mask-on-chin-mouth). This replaces the earlier 2-class GitHub substitute
and finally supplies the "incorrect" class the blueprint calls for.

NOTE: like the earlier substitute, this is still a classification dataset
(folders of face-crop images) -- it does NOT have bounding-box annotations,
so it can train this classifier but cannot be used with train_yolo.py /
convert_voc_to_yolo.py. If you need the true single-stage 3-class YOLOv8n
detector from the blueprint, you still need a bbox-annotated dataset
(e.g. andrewmvd/face-mask-detection, if/when accessible).

Download (on your own machine, not this sandbox):
    kaggle datasets download shiekhburhan/face-mask-dataset
    unzip face-mask-dataset.zip -d data/shiekhburhan_raw

Folder-name handling: the exact top-level folder names/casing in this
dataset weren't confirmed ahead of time, so this script does NOT hardcode
them. It reads whatever class folders ImageFolder finds, prints them for
you to sanity-check, and saves a classes.json next to the trained weights
mapping each discovered folder name to a canonical label (mask/no_mask/
incorrect) via CLASS_NAME_ALIASES below. If a folder name isn't recognized,
the script stops and asks you to add an alias rather than silently
mislabeling a class.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from detect import MaskClassifier

TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Maps whatever the dataset's actual folder names turn out to be -> our
# canonical 3 labels. Add entries here if the real folder names differ
# from these guesses (case-insensitive, underscores/spaces normalized).
CLASS_NAME_ALIASES = {
    "with_mask": "mask", "withmask": "mask", "mask": "mask",
    "without_mask": "no_mask", "withoutmask": "no_mask", "no_mask": "no_mask", "nomask": "no_mask",
    "incorrect": "incorrect", "incorrect_mask": "incorrect", "incorrectmask": "incorrect",
    "mask_weared_incorrect": "incorrect",
}


def normalize(folder_name):
    key = folder_name.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in CLASS_NAME_ALIASES:
        raise ValueError(
            f"Unrecognized class folder name: '{folder_name}'. "
            f"Add it to CLASS_NAME_ALIASES in train_classifier.py, mapped to "
            f"'mask', 'no_mask', or 'incorrect'."
        )
    return CLASS_NAME_ALIASES[key]


def main(data_dir, epochs, batch_size, output_path):
    full_dataset = datasets.ImageFolder(data_dir, transform=TRANSFORM)
    print(f"Classes found on disk: {full_dataset.classes}")

    canonical_names = [normalize(c) for c in full_dataset.classes]
    print(f"Mapped to canonical labels (index order): {canonical_names}")

    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MaskClassifier(num_classes=len(full_dataset.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/total:.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)

    classes_path = Path(output_path).with_name("classes.json")
    classes_path.write_text(json.dumps(canonical_names, indent=2))

    print(f"\nSaved fine-tuned weights to: {output_path}")
    print(f"Saved class mapping to: {classes_path}")


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/shiekhburhan_raw/FMD_DATASET")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="models/mask_classifier.pt")
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.batch_size, args.output)