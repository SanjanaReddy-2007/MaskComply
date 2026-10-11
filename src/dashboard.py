"""
Day 3 — Dashboard + Alerting
=============================
Live Streamlit dashboard: runs the detect -> track -> score pipeline on a
video file (or webcam) and shows the current compliance score, a rolling
trend line, and Green/Yellow/Red threshold bands in real time.

Also logs every threshold *crossing* (not every frame) to a separate CSV
as an audit trail, per the blueprint's "Alert Logger" block -- e.g. "score
dropped from GREEN to RED at 00:01:23" -- since a full per-frame log
already exists (compliance_log.csv from pipeline_video.py) and an alert
log should only record events, not every frame.

Usage:
    streamlit run src/dashboard.py -- --video path/to/video.mp4 --yolo_weights models/best.pt

(the -- before script args is required by Streamlit's CLI to separate its
own flags from your script's flags)
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from detect import CLASS_COLORS
from detection import add_detector_args, detector_from_args, classify_tracks
from tracker import CentroidTracker
from scorer import ComplianceAggregator

BAND_EMOJI = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}


def parse_args():
    # Streamlit hands the script everything after the script name, but strips the
    # "--" separator you type on the command line. Older versions kept it, so
    # accept both.
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--alert_log", default="outputs/alert_log.csv")
    add_detector_args(parser)
    return parser.parse_args(argv)


def init_alert_log(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "frame", "from_band", "to_band", "score"])


def log_alert(path, frame_idx, from_band, to_band, score):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"), frame_idx, from_band, to_band, round(score, 4)
        ])


def main():
    args = parse_args()

    st.set_page_config(page_title="MaskComply Dashboard", layout="wide")
    st.title("MaskComply — Live Compliance Dashboard")

    col1, col2 = st.columns([2, 1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        score_placeholder = st.empty()
        band_placeholder = st.empty()
        chart_placeholder = st.empty()
        alert_placeholder = st.empty()

    detector = detector_from_args(args)
    tracker = CentroidTracker(max_missed=10, max_distance=100, min_hits=3)
    aggregator = ComplianceAggregator(alpha=args.alpha, session_window=90)

    init_alert_log(args.alert_log)
    last_band = None
    score_history = []
    recent_alerts = []

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        st.error(f"Could not open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector(frame)
        active_boxes = tracker.update([d.box for d in detections])
        per_person_results, missed_ids = classify_tracks(tracker, active_boxes, detections)

        result = aggregator.update_frame(per_person_results, missed_ids)

        # --- draw overlays on frame ---
        for track_id, box in active_boxes.items():
            if not tracker.is_confirmed(track_id):
                continue
            x, y, bw, bh = box
            if track_id in per_person_results:
                label, conf = per_person_results[track_id]
                color = CLASS_COLORS.get(label, (255, 255, 255))
                text = f"ID{track_id} {label} {conf:.2f}"
            else:
                color, text = (150, 150, 150), f"ID{track_id} (missed)"
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, text, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame_score, band = result["frame_score"], result["frame_band"]

        # --- alert logging: only on threshold CROSSINGS, not every frame ---
        if band is not None and band != last_band:
            if last_band is not None:  # skip the very first frame's "crossing"
                log_alert(args.alert_log, frame_idx, last_band, band, frame_score)
                recent_alerts.insert(0, f"Frame {frame_idx} ({frame_idx/fps:.1f}s): "
                                         f"{last_band} → {band} (score={frame_score:.2f})")
            last_band = band

        if frame_score is not None:
            score_history.append(frame_score)

        # --- update dashboard ---
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if frame_score is not None:
            score_placeholder.metric("Compliance Score", f"{frame_score:.2f}")
            band_placeholder.markdown(f"### {BAND_EMOJI.get(band, '')} {band}")
            chart_placeholder.line_chart(pd.DataFrame({"score": score_history[-90:]}))
        else:
            score_placeholder.metric("Compliance Score", "-")
            band_placeholder.markdown("### No people detected")
        if recent_alerts:
            alert_placeholder.markdown("**Recent alerts:**\n" + "\n".join(
                f"- {a}" for a in recent_alerts[:5]
            ))

        frame_idx += 1
        time.sleep(1 / fps)

    cap.release()
    st.success(f"Video finished. {frame_idx} frames processed. "
               f"Alert log saved to {args.alert_log}")


if __name__ == "__main__":
    main()