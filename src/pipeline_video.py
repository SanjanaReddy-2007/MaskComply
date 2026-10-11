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

from detect import CLASS_COLORS
from detection import add_detector_args, detector_from_args, classify_tracks
from tracker import CentroidTracker
from scorer import ComplianceAggregator, threshold_band

BAND_COLOR = {"GREEN": (0, 200, 0), "YELLOW": (0, 200, 220), "RED": (0, 0, 220)}


def make_writer(output_path, fps, size):
    """MJPG in an .avi (or mp4v in an .mp4) uses OpenCV's own encoders, so it
    does not depend on extra codecs being installed. The earlier XVID choice
    produced files some players could not decode."""
    ext = Path(output_path).suffix.lower()
    fourcc = "mp4v" if ext == ".mp4" else "MJPG"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open a video writer for {output_path} ({fourcc}). "
            f"Try --output outputs/scored_video.avi, or use --save_frames to get JPG snapshots instead."
        )
    return writer


def process_video(video_path, detector, output_path, csv_path,
                   alpha=0.25, max_missed=10, save_frames_every=0):
    """detector: any callable from detection.py (YoloDetector or the legacy
    HaarClassifierDetector) that maps a frame to a list of Detection tuples."""
    tracker = CentroidTracker(max_missed=max_missed, max_distance=100)
    aggregator = ComplianceAggregator(alpha=alpha, session_window=90)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = make_writer(output_path, fps, (w, h))

    frames_dir = None
    if save_frames_every > 0:
        # JPG snapshots of the annotated frames: viewable in any image viewer,
        # even if a video player can't open the video.
        frames_dir = Path(output_path).with_name(Path(output_path).stem + "_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)

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

        detections = detector(frame)
        active_boxes = tracker.update([d.box for d in detections])

        # Unconfirmed (possibly phantom) tracks are skipped inside classify_tracks.
        per_person_results, missed_ids = classify_tracks(tracker, active_boxes, detections)

        frame_result = aggregator.update_frame(per_person_results, missed_ids)

        # --- draw overlays ---
        for track_id, box in active_boxes.items():
            if not tracker.is_confirmed(track_id):
                continue  # don't draw unconfirmed boxes -- they're often false positives
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
        else:
            # Nobody is being tracked: no score exists, so don't show one. (A grey
            # label makes this obvious instead of leaving the corner blank.)
            cv2.putText(frame, "No people detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

        writer.write(frame)
        if frames_dir is not None and frame_idx % save_frames_every == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.jpg"), frame)
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
    parser.add_argument("--output", default="outputs/scored_video.avi")
    parser.add_argument("--csv", default="outputs/compliance_log.csv")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--max_missed", type=int, default=10,
                        help="frames a person's last score is held after they stop being detected "
                             "(smooths brief misdetections; lower = score clears sooner after "
                             "someone leaves the frame)")
    parser.add_argument("--save_frames", type=int, default=0, metavar="N",
                        help="also save every Nth annotated frame as a JPG next to the video "
                             "(e.g. 30 = about one image per second)")
    add_detector_args(parser)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detector = detector_from_args(args)
    scores = process_video(args.video, detector, args.output, args.csv, args.alpha,
                           max_missed=args.max_missed, save_frames_every=args.save_frames)
    print(f"Processed {len(scores)} scored frames.")
    print(f"Annotated video: {args.output}")
    print(f"CSV log: {args.csv}")