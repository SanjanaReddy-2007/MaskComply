"""
YOLOv8n fine-tuning for the full 3-class (mask / no_mask / incorrect)
detector, as specified in the project blueprint.

Not run in this sandbox because it needs the Kaggle "Face Mask
Detection" dataset (andrewmvd/face-mask-detection), which requires a
Kaggle API key/account — this sandbox has no route to kaggle.com.

TO RUN THIS YOURSELF:

1. Get a Kaggle API token (kaggle.com/settings -> Create New Token),
   drop kaggle.json into ~/.kaggle/

2. Download + unzip the dataset:
     pip install kaggle
     kaggle datasets download andrewmvd/face-mask-detection
     unzip face-mask-detection.zip -d data/kaggle_raw

   This gives you Pascal VOC XML annotations (images/ + annotations/).

3. Convert VOC XML -> YOLO txt format (one txt per image: class cx cy w h,
   normalized 0-1). A converter script is straightforward to write with
   xml.etree.ElementTree, or use an existing VOC->YOLO converter.
   Split into data/yolo_dataset/{images,labels}/{train,val}/.

4. Fill in data.yaml (see below) and run this script:
     python scripts/train_yolo.py --data data/yolo_dataset/data.yaml --epochs 50

Expected data.yaml format:

    path: data/yolo_dataset
    train: images/train
    val: images/val
    names:
      0: mask
      1: no_mask
      2: incorrect
"""

import argparse
from ultralytics import YOLO


def main(data_yaml, epochs, imgsz, batch):
    model = YOLO("yolov8n.pt")  # start from COCO-pretrained weights
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        project="models",
        name="mask_yolov8n",
    )
    metrics = model.val()
    print("Validation mAP50-95:", metrics.box.map)
    # Where Ultralytics puts the run depends on its version (recent versions nest
    # it under runs/detect/), so print the real path instead of guessing.
    best = getattr(getattr(model, "trainer", None), "best", None)
    print(f"Best weights: {best}" if best else "Best weights: see the 'weights' folder of this run")
    print("Copy that file to models/best.pt -- that is where the other scripts look by default.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    main(args.data, args.epochs, args.imgsz, args.batch)
