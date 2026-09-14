"""
Day 2 — Full pipeline: detect -> track -> EMA score, frame by frame.

Works on any video file or webcam (cv2.VideoCapture(0)). Draws per-person
box + label + confidence + persistent track ID, plus the running frame-level
compliance score and threshold band in the corner.
"""

import argparse
import csv
from pathlib import Path

import cv2

from src.detect import load_face_detector, load_classifier, classify_face, CLASS_COLORS
from src.tracker import CentroidTracker
from src.scorer import ComplianceAggregator, threshold_band
from src.nms import deduplicate_boxes

BAND_COLOR = {"GREEN": (0, 200, 0), "YELLOW": (0, 200, 220), "RED": (0, 0, 220)}


def process_video(video_path, weights_path, output_path, csv_path,
                   alpha=0.25, max_missed=10):
    face_detector = load_face_detector()
    classifier = load_classifier(weights_path)
    tracker = CentroidTracker(max_missed=max_missed, max_distance=100)
    aggregator = ComplianceAggregator(alpha=alpha, session_window=90)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp_s", "frame_score", "band", "session_score"])

    frame_scores = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_boxes = face_detector.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(50, 50)
        )
        raw_boxes = deduplicate_boxes([tuple(b) for b in raw_boxes], iou_thresh=0.3)

        active_boxes = tracker.update(raw_boxes)

        per_person_results, missed_ids = {}, set()
        for track_id, box in active_boxes.items():
            if not tracker.is_confirmed(track_id):
                continue  # phantom/unconfirmed track -- don't let it into the score yet
            x, y, bw, bh = box
            if tracker.is_missed_this_frame(track_id):
                missed_ids.add(track_id)
                continue
            face_crop = frame[y:y + bh, x:x + bw]
            if face_crop.size == 0:
                missed_ids.add(track_id)
                continue
            label, confidence = classify_face(classifier, face_crop)
            per_person_results[track_id] = (label, confidence)

        frame_result = aggregator.update_frame(per_person_results, missed_ids)

        # --- draw overlays ---
        for track_id, box in active_boxes.items():
            x, y, bw, bh = box
            if track_id in per_person_results:
                label, confidence = per_person_results[track_id]
                color = CLASS_COLORS.get(label, (255, 255, 255))
                ema = frame_result["per_person_ema"].get(track_id, 0.0)
                text = f"ID{track_id} {label} {confidence:.2f} ema={ema:.2f}"
            else:
                color = (150, 150, 150)
                text = f"ID{track_id} (missed)"
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, text, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        fscore = frame_result["frame_score"]
        band = frame_result["frame_band"]
        if fscore is not None:
            band_color = BAND_COLOR[band]
            cv2.putText(frame, f"Compliance: {fscore:.2f} [{band}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, band_color, 2)
            frame_scores.append(fscore)

        writer.write(frame)
        csv_writer.writerow([
            frame_idx, round(frame_idx / fps, 2),
            round(fscore, 4) if fscore is not None else "",
            band or "",
            round(frame_result["session_score"], 4) if frame_result["session_score"] else "",
        ])
        frame_idx += 1

    cap.release()
    writer.release()
    csv_file.close()
    return frame_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default="models/mask_classifier.pt")
    parser.add_argument("--output", default="outputs/scored_video.mp4")
    parser.add_argument("--csv", default="outputs/compliance_log.csv")
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()

    scores = process_video(args.video, args.weights, args.output, args.csv, args.alpha)
    print(f"Processed {len(scores)} scored frames.")
    print(f"Annotated video: {args.output}")
    print(f"CSV log: {args.csv}")
