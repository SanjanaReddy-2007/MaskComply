"""
Day 4 — Validation
===================
Runs the full pipeline on a set of test clips, computes each clip's mean
session-level compliance score, and correlates it against your own manual
human ratings (0-1 scale) using Pearson and Spearman correlation -- the
blueprint's actual validation methodology and the paper's core novelty
claim ("human-validated compliance score", not just detector accuracy).

You provide the human ratings yourself (or with 1-2 others) by watching
each clip and rating overall mask compliance 0.0-1.0 by eye. There's no
way to automate this step -- it's the whole point of the validation.

Usage:
    python src/validate.py --config validation_config.json \
        --weights models/mask_classifier.pt

Where validation_config.json looks like:
{
  "clips": [
    {"path": "data/test_clips/clip1.mp4", "human_rating": 0.9},
    {"path": "data/test_clips/clip2.mp4", "human_rating": 0.4},
    {"path": "data/test_clips/clip3.mp4", "human_rating": 0.1},
    {"path": "data/test_clips/clip4.mp4", "human_rating": 0.7},
    {"path": "data/test_clips/clip5.mp4", "human_rating": 0.55}
  ]
}

Fill in real clip paths and your own honest human ratings before running.
See build_validation_config.py to generate a template automatically from
a folder of clips.
"""

import argparse
import json
from pathlib import Path

import cv2
from scipy.stats import pearsonr, spearmanr

from detection import add_detector_args, detector_from_args, classify_tracks
from tracker import CentroidTracker
from scorer import ComplianceAggregator


def score_clip(video_path, detector, alpha=0.25):
    """Runs the full pipeline on one clip, returns its mean session score."""
    tracker = CentroidTracker(max_missed=10, max_distance=100, min_hits=3)
    aggregator = ComplianceAggregator(alpha=alpha, session_window=10_000)  # no window cap for a full-clip average

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open: {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector(frame)
        active_boxes = tracker.update([d.box for d in detections])
        per_person_results, missed_ids = classify_tracks(tracker, active_boxes, detections)

        aggregator.update_frame(per_person_results, missed_ids)

    cap.release()

    if not aggregator.frame_history:
        return None  # no faces ever detected in this clip
    return sum(aggregator.frame_history) / len(aggregator.frame_history)


def main(config_path, args):
    config = json.loads(Path(config_path).read_text())
    clips = config["clips"]

    unrated = [c["path"] for c in clips if c.get("human_rating") is None]
    if unrated:
        print("These clips still have human_rating = null in the config:")
        for path in unrated:
            print(f"  {path}")
        print("Watch each clip, enter your own 0.0-1.0 compliance rating, then run again.")
        return

    detector = detector_from_args(args)  # built after the checks so config mistakes show first

    automated_scores, human_ratings, labels = [], [], []
    print(f"Scoring {len(clips)} clip(s)...\n")

    for clip in clips:
        path = clip["path"]
        human = clip["human_rating"]
        print(f"  {path} ... ", end="", flush=True)

        auto = score_clip(path, detector)
        if auto is None:
            print("SKIPPED (no faces detected)")
            continue

        print(f"automated={auto:.3f}  human={human:.3f}")
        automated_scores.append(auto)
        human_ratings.append(human)
        labels.append(Path(path).name)

    if len(automated_scores) < 3:
        print("\nNeed at least 3 valid clips to compute a meaningful correlation.")
        print(f"Only got {len(automated_scores)}. Add more clips to validation_config.json.")
        return

    pearson_r, pearson_p = pearsonr(automated_scores, human_ratings)
    spearman_r, spearman_p = spearmanr(automated_scores, human_ratings)

    print("\n" + "=" * 50)
    print("VALIDATION RESULT")
    print("=" * 50)
    print(f"Clips scored: {len(automated_scores)}")
    print(f"Pearson r  = {pearson_r:.3f}  (p={pearson_p:.4f})")
    print(f"Spearman r = {spearman_r:.3f}  (p={spearman_p:.4f})")
    print()
    print(f"{'Clip':30s} {'Automated':>10s} {'Human':>10s}")
    for label, a, h in zip(labels, automated_scores, human_ratings):
        print(f"{label:30s} {a:10.3f} {h:10.3f}")

    result_path = Path("outputs/validation_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "pearson_r": pearson_r, "pearson_p": pearson_p,
        "spearman_r": spearman_r, "spearman_p": spearman_p,
        "clips": [{"name": l, "automated": a, "human": h}
                  for l, a, h in zip(labels, automated_scores, human_ratings)],
    }, indent=2))
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="validation_config.json")
    add_detector_args(parser)
    args = parser.parse_args()
    main(args.config, args)