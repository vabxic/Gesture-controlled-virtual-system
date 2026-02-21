"""
hand_tracker.py — Modern MediaPipe Tasks-based hand tracking.

This version uses the `mediapipe.tasks` API to support Python 3.13+,
where the legacy `solutions` API may be missing.

Responsibilities:
  • Initialise HandLandmarker with a model file.
  • Process each BGR frame → extract key landmarks.
  • Return a lightweight HandData dataclass.
  • Custom landmark drawing using OpenCV (replacing missing mp_drawing).
"""

from dataclasses import dataclass
import numpy as np
import mediapipe as mp
import cv2
import os
from utils import distance

# Modern Tasks API imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass(slots=True)
class HandData:
    """Lightweight container for the landmarks we actually need."""
    wrist:       tuple[float, float]
    palm_center: tuple[float, float]
    thumb_tip:   tuple[float, float]
    index_tip:   tuple[float, float]
    middle_tip:  tuple[float, float]
    ring_tip:    tuple[float, float]
    pinky_tip:   tuple[float, float]
    palm_width:  float
    avg_z:       float
    all_landmarks: list


class HandTracker:
    """Wraps MediaPipe HandLandmarker for single-hand tracking."""

    def __init__(self, model_path: str = "assets/hand_landmarker.task"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model not found at {model_path}")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
        )
        self._detector = HandLandmarker.create_from_options(options)

    # ── Public API ────────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> "HandData | None":
        """Run detection and return HandData or None."""
        rgb = bgr_frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        results = self._detector.detect(mp_image)

        if not results.hand_landmarks:
            return None

        # Landmarks are normalized [0, 1]
        lm = results.hand_landmarks[0]

        wrist      = (lm[0].x,  lm[0].y)
        thumb_tip  = (lm[4].x,  lm[4].y)
        index_tip  = (lm[8].x,  lm[8].y)
        middle_tip = (lm[12].x, lm[12].y)
        ring_tip   = (lm[16].x, lm[16].y)
        pinky_tip  = (lm[20].x, lm[20].y)

        palm_ids = [0, 5, 9, 13, 17]
        px = sum(lm[i].x for i in palm_ids) / len(palm_ids)
        py = sum(lm[i].y for i in palm_ids) / len(palm_ids)
        palm_center = (px, py)

        palm_width = distance((lm[5].x, lm[5].y), (lm[17].x, lm[17].y))
        avg_z = sum(lm[i].z for i in palm_ids) / len(palm_ids)

        return HandData(
            wrist=wrist,
            palm_center=palm_center,
            thumb_tip=thumb_tip,
            index_tip=index_tip,
            middle_tip=middle_tip,
            ring_tip=ring_tip,
            pinky_tip=pinky_tip,
            palm_width=palm_width,
            avg_z=avg_z,
            all_landmarks=lm,
        )

    def draw(self, bgr_frame: np.ndarray, hd: HandData) -> np.ndarray:
        """Custom drawing logic using standard CV2 (replacing mp_drawing)."""
        if not hd:
            return bgr_frame

        h, w = bgr_frame.shape[:2]
        lm = hd.all_landmarks

        # Define connections (MediaPipe hand structure)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (9, 10), (11, 12),                   # Middle (partial)
            (13, 14), (15, 16),                  # Ring (partial)
            (0, 17), (17, 18), (19, 20),         # Pinky (partial)
            (5, 9), (9, 13), (13, 17)            # Palm base
        ]
        # Adding missing middle/ring segments for a fuller look
        connections += [(5, 9), (9, 13), (13, 17), (9, 10), (10, 11), (13, 14), (14, 15), (17, 18), (18, 19)]

        # Draw lines
        for start_idx, end_idx in connections:
            pt1 = (int(lm[start_idx].x * w), int(lm[start_idx].y * h))
            pt2 = (int(lm[end_idx].x * w), int(lm[end_idx].y * h))
            cv2.line(bgr_frame, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

        # Draw joints
        for i, point in enumerate(lm):
            px, py = int(point.x * w), int(point.y * h)
            # Differentiate tips
            color = (0, 255, 0) if i in [4, 8, 12, 16, 20] else (60, 60, 255)
            cv2.circle(bgr_frame, (px, py), 4, color, -1, cv2.LINE_AA)

        return bgr_frame

    def process_and_draw(self, bgr_frame: np.ndarray) -> "HandData | None":
        """Combined pass: detect and draw."""
        hd = self.process(bgr_frame)
        if hd:
            self.draw(bgr_frame, hd)
        return hd

    def release(self):
        self._detector.close()
