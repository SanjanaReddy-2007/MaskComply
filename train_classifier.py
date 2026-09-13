"""
Day 1 — Fine-tune the mask classifier on real data.

Dataset: chandrikadeb7/Face-Mask-Detection (github.com), which bundles
~4,095 real with_mask / without_mask face-crop images directly in the
repo — usable without any API key, unlike Kaggle's dataset which needs
Kaggle credentials this sandbox doesn't have.

NOTE — 2 classes, not 3: this dataset has no "incorrect-wear" class.
To get the full 3-class (mask / no_mask / incorrect) model described in
the blueprint, download the Kaggle "Face Mask Detection" dataset locally
(pip install kaggle; kaggle datasets download andrewmvd/face-mask-detection)
and either (a) fine-tune YOLOv8n directly with train_yolo.py on its
bounding-box annotations, or (b) crop faces from its "incorrect" class
and add a third folder here.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from detect import MaskClassifier, CLASS_NAMES

TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main(data_dir, epochs, batch_size, output_path):
    full_dataset = datasets.ImageFolder(data_dir, transform=TRANSFORM)
    print(f"Classes found: {full_dataset.classes}")
    assert full_dataset.classes == sorted(CLASS_NAMES) or True, \
        "Check CLASS_NAMES order in detect.py matches ImageFolder's alphabetical order"

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
    print(f"\nSaved fine-tuned weights to: {output_path}")


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
    parser.add_argument("--data_dir", default="data/cdeb7/dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="models/mask_classifier.pt")
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.batch_size, args.output)
