"""
Builds a synthetic test video from the real pic2.jpg (3 faces: mask,
no_mask, no_mask) to stress-test the tracker + EMA scorer against exactly
the failure mode the Day 2 checkpoint cares about:

    "score updates smoothly frame-to-frame without wild jumps when someone
     briefly looks away or is misdetected for one frame"

No real webcam/video feed is available in this sandbox, so this is a
controlled synthetic substitute -- flagged clearly, not passed off as a
real video test. It:
  - jitters the frame slightly each step (simulates natural camera/subject
    motion, keeps the tracker's centroid-matching honest rather than
    matching identical static frames trivially)
  - blacks out the mask-wearing face's region for 4 consecutive frames
    partway through, simulating that person turning away / a momentary
    misdetection -- this is the exact event the EMA smoothing must survive
    without the compliance score cratering to 0 for those frames
"""

import cv2
import numpy as np

SRC_IMAGE = "data/raw/day1_substitute_cdeb7/images/pic2.jpg"
OUT_VIDEO = "outputs/synthetic_test.avi"
N_FRAMES = 60
DROPOUT_START = 25
DROPOUT_LEN = 4
# Region of the mask-wearing (center) face, from Day 1's detection run
MASK_FACE_BOX = (343, 58, 212, 212)  # x, y, w, h


def build_video():
    img = cv2.imread(SRC_IMAGE)
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"XVID"), 10, (w, h))

    rng = np.random.default_rng(42)
    for i in range(N_FRAMES):
        frame = img.copy()

        # small jitter: random translation, capped so faces stay near their spot
        dx, dy = rng.integers(-4, 5), rng.integers(-4, 5)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # mild gaussian noise so it's not a frozen static repeat
        noise = rng.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # simulated misdetection: blank out the mask-wearer's face region
        if DROPOUT_START <= i < DROPOUT_START + DROPOUT_LEN:
            x, y, bw, bh = MASK_FACE_BOX
            frame[y:y + bh, x:x + bw] = (120, 120, 120)  # flat gray, no face features

        writer.write(frame)

    writer.release()
    print(f"Built {N_FRAMES}-frame synthetic test video: {OUT_VIDEO}")
    print(f"Simulated misdetection on frames {DROPOUT_START}-{DROPOUT_START + DROPOUT_LEN - 1}")


if __name__ == "__main__":
    build_video()