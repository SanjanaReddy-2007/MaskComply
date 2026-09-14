"""
Day 2 — Scoring Layer
======================
Implements the blueprint's exact formulas:

    class_weight = 1.0 (mask) / 0.5 (incorrect) / 0.0 (no_mask)
    raw_score(i) = class_weight(i) * confidence(i)
    EMA_score(p, t) = alpha * raw_score(p, t) + (1 - alpha) * EMA_score(p, t-1)
    Frame_Compliance_Score = mean(EMA_score(p, t) for all tracked p in frame)

Threshold bands: Green >= 0.8, Yellow 0.5-0.8, Red < 0.5.
"""

CLASS_WEIGHTS = {
    "mask": 1.0,
    "incorrect": 0.5,
    "no_mask": 0.0,
}


def raw_score(class_name, confidence):
    weight = CLASS_WEIGHTS.get(class_name, 0.0)
    return weight * confidence


def threshold_band(score):
    if score >= 0.8:
        return "GREEN"
    elif score >= 0.5:
        return "YELLOW"
    return "RED"


class PersonScorer:
    """Tracks one person's EMA compliance score across frames."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.ema_score = None

    def update(self, class_name, confidence, missed_this_frame=False):
        if missed_this_frame:
            # No new detection this frame (tracker carried the box forward).
            # Don't feed a raw_score of 0 -- that would incorrectly punish
            # someone for a momentary misdetection. Just hold the EMA steady.
            return self.ema_score if self.ema_score is not None else 0.0

        r = raw_score(class_name, confidence)
        if self.ema_score is None:
            self.ema_score = r  # first observation, no history to blend with
        else:
            self.ema_score = self.alpha * r + (1 - self.alpha) * self.ema_score
        return self.ema_score


class ComplianceAggregator:
    """Owns one PersonScorer per tracked ID and computes frame/session scores."""

    def __init__(self, alpha=0.25, session_window=30):
        self.alpha = alpha
        self.person_scorers = {}   # track_id -> PersonScorer
        self.session_window = session_window
        self.frame_history = []    # rolling list of frame-level scores

    def update_frame(self, per_person_results, missed_ids=None):
        """
        per_person_results: dict {track_id: (class_name, confidence)}
                             for people detected THIS frame.
        missed_ids: set of track_ids that are still active (tracker didn't
                    drop them) but had no detection this frame.
        """
        missed_ids = missed_ids or set()
        current_scores = {}

        for track_id, (class_name, confidence) in per_person_results.items():
            scorer = self.person_scorers.setdefault(track_id, PersonScorer(self.alpha))
            current_scores[track_id] = scorer.update(class_name, confidence)

        for track_id in missed_ids:
            if track_id in self.person_scorers:
                current_scores[track_id] = self.person_scorers[track_id].update(
                    None, 0.0, missed_this_frame=True
                )

        frame_score = (
            sum(current_scores.values()) / len(current_scores)
            if current_scores else None
        )

        if frame_score is not None:
            self.frame_history.append(frame_score)
            if len(self.frame_history) > self.session_window:
                self.frame_history.pop(0)

        return {
            "per_person_ema": current_scores,
            "frame_score": frame_score,
            "frame_band": threshold_band(frame_score) if frame_score is not None else None,
            "session_score": (
                sum(self.frame_history) / len(self.frame_history)
                if self.frame_history else None
            ),
        }

    def drop_track(self, track_id):
        self.person_scorers.pop(track_id, None)
