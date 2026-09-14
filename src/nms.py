"""
Fix for a real bug found during Day 2 testing: OpenCV's Haar cascade
detectMultiScale() can return multiple overlapping boxes for a single
true face -- especially on noisy or slightly perturbed frames, since its
internal grouping (minNeighbors) doesn't fully collapse near-duplicates.

Without deduplication, the centroid tracker spawns a separate "phantom"
track ID for each duplicate box, and the frame-level average gets diluted
by these phantom identities -- discovered because the synthetic-video test
showed the frame compliance score dropping far more than the EMA math
alone could explain (see debug_trace.py output).

Simple IoU-based NMS, keeping the largest box in each overlapping cluster.
"""


def _iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a, area_b = aw * ah, bw * bh
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def deduplicate_boxes(boxes, iou_thresh=0.3):
    """boxes: list of (x, y, w, h). Returns a de-duplicated list, keeping
    the largest box among any cluster of overlapping detections."""
    if len(boxes) <= 1:
        return list(boxes)

    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes_sorted:
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept
