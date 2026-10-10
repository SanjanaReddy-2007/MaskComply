"""
Convert the Kaggle "Face Mask Detection" dataset (andrewmvd/face-mask-detection,
Pascal VOC XML annotations) into the YOLO format that train_yolo.py needs,
split into train/val, and write data.yaml.

Expected raw layout (what the Kaggle zip gives you):

    data/kaggle_raw/
        images/       maksssksksss0.png, maksssksksss1.png, ...
        annotations/  maksssksksss0.xml, maksssksksss1.xml, ...

Output layout:

    data/yolo_dataset/
        images/train, images/val
        labels/train, labels/val
        data.yaml

Class ids (same names the scorer / CLASS_COLORS use):
    0 = mask, 1 = no_mask, 2 = incorrect

Usage:
    python src/convert_voc_to_yolo.py --raw_dir data/kaggle_raw --out_dir data/yolo_dataset

Optional: add background images (no faces at all -- windows, traffic lights,
signs, street scenes) as hard negatives. They get an empty label file, which
YOLO treats as "nothing here" and uses to suppress false positives:

    python src/convert_voc_to_yolo.py --negatives_dir data/negatives
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASS_NAMES = ["mask", "no_mask", "incorrect"]

# Names used inside the XML files -> our class id. Matched case-insensitively.
NAME_TO_ID = {
    "with_mask": 0, "mask": 0,
    "without_mask": 1, "no_mask": 1,
    "mask_weared_incorrect": 2, "incorrect": 2, "incorrect_mask": 2,
}

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def find_image(images_dir, xml_filename, stem):
    """The <filename> tag in the XML is sometimes wrong, so fall back to the stem."""
    if xml_filename and (images_dir / xml_filename).exists():
        return images_dir / xml_filename
    for ext in IMAGE_EXTS:
        for candidate in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if candidate.exists():
                return candidate
    return None


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w, img_h = int(size.find("width").text), int(size.find("height").text)
    filename = root.findtext("filename")

    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if name not in NAME_TO_ID:
            raise ValueError(f"{xml_path.name}: unknown class '{name}'. "
                             f"Add it to NAME_TO_ID in convert_voc_to_yolo.py.")
        bb = obj.find("bndbox")
        xmin, ymin = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        xmax, ymax = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        # clip to the image, skip degenerate boxes
        xmin, xmax = max(0.0, xmin), min(float(img_w), xmax)
        ymin, ymax = max(0.0, ymin), min(float(img_h), ymax)
        if xmax - xmin < 2 or ymax - ymin < 2:
            continue
        boxes.append((NAME_TO_ID[name], xmin, ymin, xmax, ymax))
    return filename, img_w, img_h, boxes


def to_yolo_line(cls_id, xmin, ymin, xmax, ymax, img_w, img_h):
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main(raw_dir, out_dir, val_frac, seed, negatives_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    images_dir, ann_dir = raw_dir / "images", raw_dir / "annotations"
    if not images_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {images_dir} and {ann_dir}. Unzip the Kaggle download into "
            f"{raw_dir} so it contains 'images' and 'annotations' folders."
        )

    # (image_path, [yolo label lines]) for every sample
    samples, skipped = [], 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        filename, img_w, img_h, boxes = parse_xml(xml_path)
        img_path = find_image(images_dir, filename, xml_path.stem)
        if img_path is None:
            skipped += 1
            continue
        lines = [to_yolo_line(c, x1, y1, x2, y2, img_w, img_h) for c, x1, y1, x2, y2 in boxes]
        samples.append((img_path, lines))
    if skipped:
        print(f"WARNING: {skipped} annotation file(s) had no matching image and were skipped.")

    n_annotated = len(samples)
    if negatives_dir:
        neg_files = [p for p in sorted(Path(negatives_dir).iterdir())
                     if p.suffix.lower() in IMAGE_EXTS]
        samples += [(p, []) for p in neg_files]
        print(f"Added {len(neg_files)} background (no-face) image(s) as hard negatives.")

    if not samples:
        raise RuntimeError("No samples found -- check --raw_dir.")

    # Split by image. Shuffle with a fixed seed so the split is reproducible.
    random.Random(seed).shuffle(samples)
    n_val = max(1, int(round(len(samples) * val_frac)))
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    # Start clean so stale files from an earlier run can't leak between splits.
    for sub in ("images", "labels"):
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)

    for split, items in splits.items():
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts = Counter()
        used_names = set()
        for img_path, lines in items:
            name = img_path.name
            # negatives could share a filename with a Kaggle image -- keep names unique
            if name in used_names:
                name = f"{img_path.stem}_dup{len(used_names)}{img_path.suffix}"
            used_names.add(name)
            shutil.copy2(img_path, out_dir / "images" / split / name)
            (out_dir / "labels" / split / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            for line in lines:
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
        print(f"{split:>5}: {len(items):4d} images  boxes per class: "
              f"{ {c: counts.get(c, 0) for c in CLASS_NAMES} }")

    # Absolute path so ultralytics finds the data no matter where you run from.
    yaml_text = (
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (out_dir / "data.yaml").write_text(yaml_text)
    print(f"\n{n_annotated} annotated images converted -> {out_dir}")
    print(f"Wrote {out_dir / 'data.yaml'}")
    print("Next: python src/train_yolo.py --data "
          f"{out_dir / 'data.yaml'} --epochs 50 --imgsz 960")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/kaggle_raw")
    parser.add_argument("--out_dir", default="data/yolo_dataset")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negatives_dir", default=None,
                        help="optional folder of images with NO faces (hard negatives)")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.val_frac, args.seed, args.negatives_dir)