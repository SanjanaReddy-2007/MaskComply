"""
Scans a folder of video clips and writes a validation_config.json
template with human_rating placeholders for you to fill in by hand
after watching each clip.

Usage:
    python src/build_validation_config.py --clips_dir data/test_clips
"""

import argparse
import json
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main(clips_dir, output_path):
    clips_dir = Path(clips_dir)
    clip_paths = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)

    if not clip_paths:
        print(f"No video files found in {clips_dir}")
        return

    config = {"clips": [
        {"path": str(p), "human_rating": None}  # <- fill in after watching each clip
        for p in clip_paths
    ]}

    Path(output_path).write_text(json.dumps(config, indent=2))
    print(f"Found {len(clip_paths)} clip(s). Wrote template to {output_path}")
    print("\nNext: open it, watch each clip, and replace each null with your")
    print("own 0.0-1.0 compliance rating before running validate.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", default="data/test_clips")
    parser.add_argument("--output", default="validation_config.json")
    args = parser.parse_args()
    main(args.clips_dir, args.output)