"""
Day 2 — Tracker
================
Simple centroid tracker: matches detections across frames by nearest
centroid distance, giving each person a persistent ID. No re-ID/appearance
model — good enough for a single-camera, low-crowd-density monitoring feed
per the blueprint's "keep it lightweight" tech choice (no DeepSORT needed).

Handles the two failure modes that matter for the scoring layer downstream:
  - A person briefly not detected for a frame or two (e.g. looked away) ->
    tracker keeps their ID alive for `max_missed` frames before dropping it,
    so the EMA scorer isn't fed a "new person" and doesn't reset to 0.
  - A new face appears -> gets a new ID once, not a new ID every frame.
"""

from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_missed=10, max_distance=80, min_hits=3):
        self.next_id = 0
        self.objects = OrderedDict()   # id -> (centroid, box)
        self.missed = OrderedDict()    # id -> consecutive frames missed
        self.hits = OrderedDict()      # id -> consecutive frames matched (for initial confirmation)
        self.confirmed = OrderedDict() # id -> True once ever confirmed (sticky, survives brief drops)
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.min_hits = min_hits

    @staticmethod
    def _centroid(box):
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def update(self, boxes):
        """
        boxes: list of (x, y, w, h) detections for the current frame.
        Returns: dict {track_id: box} for all currently-active tracks
                 (includes tracks temporarily missed this frame, using
                 their last known box, so the scorer can still discount them).
        """
        input_centroids = [self._centroid(b) for b in boxes]

        if len(self.objects) == 0:
            for c, b in zip(input_centroids, boxes):
                self._register(c, b)
            return self._active_boxes()

        if len(input_centroids) == 0:
            for oid in list(self.missed.keys()):
                self.missed[oid] += 1
                if self.missed[oid] > self.max_missed:
                    self._deregister(oid)
            return self._active_boxes()

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid][0] for oid in object_ids]

        D = np.linalg.norm(
            np.array(object_centroids)[:, None, :] - np.array(input_centroids)[None, :, :],
            axis=2,
        )

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid] = (input_centroids[col], boxes[col])
            self.missed[oid] = 0
            self.hits[oid] = self.hits.get(oid, 0) + 1
            if self.hits[oid] >= self.min_hits:
                self.confirmed[oid] = True
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.missed[oid] += 1
            if self.missed[oid] > self.max_missed:
                self._deregister(oid)

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(input_centroids[col], boxes[col])

        return self._active_boxes()

    def _register(self, centroid, box):
        self.objects[self.next_id] = (centroid, box)
        self.missed[self.next_id] = 0
        self.hits[self.next_id] = 1
        self.confirmed[self.next_id] = self.min_hits <= 1
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.missed[oid]
        del self.hits[oid]
        del self.confirmed[oid]

    def _active_boxes(self):
        return {oid: box for oid, (centroid, box) in self.objects.items()}

    def is_missed_this_frame(self, oid):
        return self.missed.get(oid, 0) > 0

    def is_confirmed(self, oid):
        """A track only 'counts' toward scoring once it's been matched
        min_hits times in a row at some point. Sticky once True -- a track
        that later has a brief dropout stays confirmed (its EMA is held,
        not reset), so recovering from a misdetection doesn't force it to
        re-earn confirmation. Filters out one-off false-positive detections
        spawning short-lived phantom tracks that would otherwise dilute the
        frame-level average at full weight from their very first frame."""
        return self.confirmed.get(oid, False)
