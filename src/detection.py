"""
Shared detection step for pipeline_video.py, dashboard.py, validate.py and
detect.py, so all of them behave identically.

Two interchangeable detectors, both returning a list of Detection(box, label,
confidence) for one frame, where box is (x, y, w, h) in pixels:

  YoloDetector                 -- the fine-tuned YOLOv8 model (models/best.pt).
                                  Single stage: it finds the face AND says
                                  mask / no_mask / incorrect. Because it was
                                  trained with background images, it should
                                  not fire on windows, lights or clothing the
                                  way the Haar cascade did.
  HaarClassifierDetector       -- the old two-stage path (Haar cascade + CNN
                                  classifier). Kept as a fallback.

classify_tracks() then attaches a class to each confirmed tracker ID.
"""

from collections import namedtuple
from pathlib import Path

import cv2

from nms import _iou, deduplicate_boxes

Detection = namedtuple("Detection", ["box", "label", "confidence"])

# Whatever names the YOLO model was trained with -> the names scorer.py and
# CLASS_COLORS use.
CANONICAL = {
    "mask": "mask", "with_mask": "mask",
    "no_mask": "no_mask", "without_mask": "no_mask",
    "incorrect": "incorrect", "mask_weared_incorrect": "incorrect",
    "incorrect_mask": "incorrect",
}


class YoloDetector:
    def __init__(self, weights, conf=0.4, imgsz=960, device=None):
        from ultralytics import YOLO  # imported here so the Haar path doesn't need it

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.names = {
            int(i): CANONICAL.get(str(n).strip().lower(), str(n))
            for i, n in self.model.names.items()
        }

    def __call__(self, frame):
        # agnostic_nms=True: one box per face even if two classes both fire on
        # it (otherwise the same face could show up as "mask" AND "incorrect").
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
            agnostic_nms=True, verbose=False,
        )[0]

        h, w = frame.shape[:2]
        detections = []
        for b in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
            bw, bh = x2 - x1, y2 - y1
            if bw < 2 or bh < 2:
                continue
            label = self.names.get(int(b.cls[0]), "no_mask")
            detections.append(Detection((x1, y1, bw, bh), label, float(b.conf[0])))
        return detections


class HaarClassifierDetector:
    """Legacy two-stage detector: Haar cascade for location, CNN for class."""

    def __init__(self, classifier_weights):
        from detect import load_face_detector, load_classifier, classify_face

        self._classify = classify_face
        self.face_detector = load_face_detector()
        self.classifier, self.class_names = load_classifier(classifier_weights)

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50))
        boxes = deduplicate_boxes([tuple(int(v) for v in b) for b in boxes], iou_thresh=0.3)

        detections = []
        for (x, y, bw, bh) in boxes:
            crop = frame[y:y + bh, x:x + bw]
            if crop.size == 0:
                continue
            label, conf = self._classify(self.classifier, self.class_names, crop)
            detections.append(Detection((x, y, bw, bh), label, conf))
        return detections


def make_detector(kind="yolo", yolo_weights="models/best.pt",
                  classifier_weights="models/mask_classifier.pt",
                  conf=0.4, imgsz=960, device=None):
    if kind == "yolo":
        if not Path(yolo_weights).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at '{yolo_weights}'. Put your trained best.pt "
                f"there, or pass --yolo_weights <path>."
            )
        return YoloDetector(yolo_weights, conf=conf, imgsz=imgsz, device=device)
    if kind == "haar":
        return HaarClassifierDetector(classifier_weights)
    raise ValueError(f"Unknown detector '{kind}' (use 'yolo' or 'haar')")


def add_detector_args(parser):
    """Adds the shared command-line options to an argparse parser."""
    parser.add_argument("--detector", choices=["yolo", "haar"], default="yolo",
                        help="yolo = fine-tuned YOLOv8 (default); haar = old Haar+CNN path")
    parser.add_argument("--yolo_weights", default="models/best.pt")
    parser.add_argument("--classifier_weights", default="models/mask_classifier.pt",
                        help="only used with --detector haar")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="minimum YOLO detection confidence (raise it to cut false boxes)")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    return parser


def detector_from_args(args):
    return make_detector(args.detector, args.yolo_weights, args.classifier_weights,
                         conf=args.conf, imgsz=args.imgsz)


def classify_tracks(tracker, active_boxes, detections, min_iou=0.5):
    """
    Attach this frame's class + confidence to each confirmed tracker ID.

    Returns (per_person_results, missed_ids) in exactly the shape
    ComplianceAggregator.update_frame() expects:
      per_person_results = {track_id: (label, confidence)}
      missed_ids         = set of confirmed IDs with no detection this frame
    Unconfirmed (possibly phantom) tracks are left out entirely.
    """
    per_person, missed = {}, set()
    for track_id, box in active_boxes.items():
        if not tracker.is_confirmed(track_id):
            continue
        if tracker.is_missed_this_frame(track_id):
            missed.add(track_id)
            continue
        best, best_iou = None, 0.0
        for d in detections:
            iou = _iou(box, d.box)
            if iou > best_iou:
                best, best_iou = d, iou
        if best is None or best_iou < min_iou:
            missed.add(track_id)
        else:
            per_person[track_id] = (best.label, best.confidence)
    return per_person, missed
