"""
gesture_detector.py — Classify hand poses into gestures.

Gestures detected:
  • "open_hand"  — all fingers extended (interaction-ready)
  • "pull"       — open hand moving forward (depth decreasing over time)
  • "grab"       — thumb tip and index tip pinched together
  • "none"       — no recognisable gesture

The detector maintains a short rolling buffer of HandData frames so it can
reason about temporal changes (pull requires consistent motion over N frames).
Hysteresis prevents jitter: a gesture must be stable for GESTURE_HYSTERESIS
consecutive frames before it is reported.
"""

import collections
from hand_tracker import HandData
from utils import (
    distance,
    MovingAverage,
    GRAB_THRESHOLD,
    PULL_Z_DELTA_THRESHOLD,
    PULL_FRAMES_REQUIRED,
    GESTURE_HYSTERESIS,
    DEPTH_SMOOTH_WINDOW,
)


class GestureDetector:
    """Temporal gesture classifier operating on a stream of HandData frames."""

    def __init__(self):
        # Rolling buffer of recent HandData (oldest → newest)
        self._buffer: collections.deque[HandData] = collections.deque(maxlen=12)

        # Depth smoothers to reduce noise
        self._z_smoother = MovingAverage(window=6)
        self._w_smoother = MovingAverage(window=6)

        # Hysteresis counter: (gesture_name, consecutive_count)
        self._last_gesture: str = "none"
        self._stable_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, hand_data: HandData) -> str:
        """Classify the current gesture from *hand_data*."""
        self._buffer.append(hand_data)
        self._z_smoother.update(hand_data.avg_z)
        self._w_smoother.update(hand_data.palm_width)

        raw = self._classify(hand_data)

        # ── Hysteresis ────────────────────────────────────────────────
        if raw == self._last_gesture:
            self._stable_count += 1
        else:
            self._last_gesture = raw
            self._stable_count = 1

        # Only report once the gesture has been stable long enough
        if self._stable_count >= GESTURE_HYSTERESIS:
            return raw
        
        return self._last_gesture if self._stable_count > 1 else "none"

    def reset(self):
        """Clear internal state (e.g. when tracking is lost)."""
        self._buffer.clear()
        self._z_smoother.reset()
        self._w_smoother.reset()
        self._last_gesture = "none"
        self._stable_count = 0

    # ── Internal classification ───────────────────────────────────────────

    def _classify(self, hd: HandData) -> str:
        """Raw (un-hysteresised) gesture classification."""
        # 1. Check GRAB first
        if self._is_grab(hd):
            return "grab"

        # 2. Check if hand is open
        fingers_open = self._fingers_extended(hd)
        if not fingers_open:
            return "none"

        # 3. Check PULL — open hand + forward motion
        if self._is_pull():
            return "pull"

        # 4. Fallback: open hand
        return "open_hand"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _is_grab(hd: HandData) -> bool:
        """True when thumb tip and index tip are pinched together."""
        return distance(hd.thumb_tip, hd.index_tip) < GRAB_THRESHOLD

    @staticmethod
    def _fingers_extended(hd: HandData) -> bool:
        """True when the four fingers (index through pinky) are extended."""
        lm = hd.all_landmarks
        # Indices: Index(8,5), Middle(12,9), Ring(16,13), Pinky(20,17)
        tips = [8, 12, 16, 20]
        mcps = [5,  9, 13, 17]

        for tip_id, mcp_id in zip(tips, mcps):
            if lm[tip_id].y >= lm[mcp_id].y:
                return False
        return True

    def _is_pull(self) -> bool:
        """True if either Z is decreasing OR palm-width is increasing."""
        from utils import PULL_Z_DELTA_THRESHOLD, PULL_W_DELTA_THRESHOLD, PULL_FRAMES_REQUIRED

        z_vals = self._z_smoother.values
        w_vals = self._w_smoother.values

        if len(z_vals) < PULL_FRAMES_REQUIRED + 1:
            return False

        # Compute deltas over consecutive frames
        z_recent = z_vals[-(PULL_FRAMES_REQUIRED + 1):]
        w_recent = w_vals[-(PULL_FRAMES_REQUIRED + 1):]
        
        z_deltas = [z_recent[i+1] - z_recent[i] for i in range(len(z_recent)-1)]
        w_deltas = [w_recent[i+1] - w_recent[i] for i in range(len(w_recent)-1)]

        # Check for consistent trends
        # Z decreasing = approaching
        z_consistent = all(d < PULL_Z_DELTA_THRESHOLD for d in z_deltas)
        # Width increasing = approaching
        w_consistent = all(d > PULL_W_DELTA_THRESHOLD for d in w_deltas)

        return z_consistent or w_consistent
