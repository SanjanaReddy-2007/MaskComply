"""
Day 1 — Base Detector
======================
Detects faces in an image/video frame and classifies mask compliance,
drawing bounding boxes + class label + confidence for each face.

The default detector is now the fine-tuned YOLOv8 model (models/best.pt, see
train_yolo.py), a single stage that finds each face and classifies it. The
original two-stage path (Haar cascade + the CNN classifier defined below) is
still available with --detector haar, and MaskClassifier / classify_face are
still used by train_classifier.py and verify_classifier.py. The detector
implementations live in detection.py. The scoring layer consumes the same
(box, class, confidence) triples either way.
"""

import argparse
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path

CLASS_COLORS = {
    "mask": (0, 200, 0),        # green (BGR)
    "no_mask": (0, 0, 220),     # red
    "incorrect": (0, 200, 220), # yellow
}


def load_class_names(weights_path):
    """Reads classes.json saved alongside the weights by train_classifier.py.
    Falls back to the old 2-class default only if it's genuinely missing,
    so older weight files from Day 1 still work."""
    classes_path = Path(weights_path).with_name("classes.json")
    if classes_path.exists():
        return json.loads(classes_path.read_text())
    return ["mask", "no_mask"]  # legacy fallback (Day 1 2-class model)


class MaskClassifier(nn.Module):
    """Small CNN — enough capacity for face-crop mask/no_mask classification
    on CPU in a few epochs. Swap in MobileNetV3 later for better accuracy."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    return detector


def load_classifier(weights_path):
    class_names = load_class_names(weights_path)
    model = MaskClassifier(num_classes=len(class_names))
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, class_names


PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_face(model, class_names, face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = PREPROCESS(face_rgb).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    return class_names[pred_idx.item()], float(conf.item())


def run_on_image(image_path, detector, output_path):
    """detector: a callable from detection.py (YoloDetector or HaarClassifierDetector)."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    results = []
    for det in detector(img):
        x, y, w, h = det.box
        label, confidence = det.label, det.confidence
        results.append({"box": det.box, "class": label, "confidence": confidence})

        color = CLASS_COLORS.get(label, (255, 255, 255))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - th - 10), (x + tw + 6, y), color, -1)
        cv2.putText(img, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(output_path), img)
    return results


if __name__ == "__main__":
    from detection import add_detector_args, detector_from_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/annotated.jpg")
    add_detector_args(parser)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_on_image(args.image, detector_from_args(args), args.output)

    print(f"\nDetected {len(results)} face(s):")
    for r in results:
        print(f"  box={r['box']}  class={r['class']}  confidence={r['confidence']:.3f}")
    print(f"\nAnnotated image saved to: {args.output}")
