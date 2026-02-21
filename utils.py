"""
utils.py — Shared constants, math helpers, and smoothing utilities.

This module is the foundation layer: every other module imports from here.
"""

import collections
import math

# ─── Window names (used by cv2.imshow / cv2.namedWindow) ──────────────────────
CAMERA_WINDOW = "Camera Window"
APPLE_WINDOW  = "Apple Window"

# ─── Window Position & Movement ──────────────────────────────────────────────
WIN_START_X = 50                 # Initial X for Camera Window
WIN_GAP_FAR = 600                # Starting gap between windows
WIN_GAP_MIN = 20                 # Min gap when side-by-side
WIN_MOVE_THRESHOLD = 0.85        # Hand X (0-1) to start moving window
WIN_MOVE_SPEED = 20              # Pixels per frame move speed
WIN_DRAG_SENSITIVITY = 1.5        # Multiplier for dragging the window

# ─── Performance ──────────────────────────────────────────────────────────────
TARGET_FPS = 30

# ─── Apple scale limits ──────────────────────────────────────────────────────
MIN_SCALE = 0.3      # Smallest the apple can shrink to
MAX_SCALE = 2.5      # Largest the apple can grow to
DEFAULT_SCALE = 0.6  # Starting scale

# ─── Gesture thresholds ──────────────────────────────────────────────────────
GRAB_THRESHOLD = 0.06            # Normalized distance between thumb & index to count as grab
PULL_Z_DELTA_THRESHOLD = -0.002  # Loosened
PULL_W_DELTA_THRESHOLD = 0.002   # Palm width increase (hand approaching)
PULL_FRAMES_REQUIRED = 3         # Reduced from 4 for responsiveness
GESTURE_HYSTERESIS = 2           # Reduced from 3 to reduce lag

# ─── Smoothing ────────────────────────────────────────────────────────────────
DEPTH_SMOOTH_WINDOW = 8          # Rolling window size for depth moving-average
POSITION_LERP  = 0.15            # How fast apple position tracks the hand (0 = frozen, 1 = instant)
SCALE_LERP     = 0.10            # How fast apple scale changes
PULL_SCALE_STEP = 0.04           # Increased from 0.02 for more visible feedback

# ─── Apple default position (normalized 0-1) ─────────────────────────────────
DEFAULT_APPLE_X = 0.5
DEFAULT_APPLE_Y = 0.5
DEFAULT_APPLE_Z = 1.0            # Far away initially

# ─── Idle timeout ─────────────────────────────────────────────────────────────
IDLE_TIMEOUT_FRAMES = 15         # Frames without a hand before resetting to IDLE


# ═══════════════════════════════════════════════════════════════════════════════
#  Math helpers
# ═══════════════════════════════════════════════════════════════════════════════

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from *a* to *b* by factor *t* ∈ [0, 1]."""
    return a + (b - a) * t


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp *val* to the range [lo, hi]."""
    return max(lo, min(hi, val))


def distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two (x, y) or (x, y, z) points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Moving average filter
# ═══════════════════════════════════════════════════════════════════════════════

class MovingAverage:
    """Simple moving-average filter backed by a fixed-size deque.

    Usage:
        smoother = MovingAverage(window=8)
        smooth_val = smoother.update(raw_val)
    """

    def __init__(self, window: int = DEPTH_SMOOTH_WINDOW):
        self._buf: collections.deque = collections.deque(maxlen=window)

    def update(self, value: float) -> float:
        """Push *value* and return the current average."""
        self._buf.append(value)
        return sum(self._buf) / len(self._buf)

    def reset(self):
        """Clear the buffer."""
        self._buf.clear()

    @property
    def values(self):
        """Return a list copy of the current buffer (oldest → newest)."""
        return list(self._buf)

    @property
    def ready(self) -> bool:
        """True once the buffer has at least 2 values."""
        return len(self._buf) >= 2
