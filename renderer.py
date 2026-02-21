"""
renderer.py — Renders the virtual apple onto the Apple Window canvas.

Responsibilities:
  • Load an RGBA apple image (transparent PNG).
  • Resize the apple according to the controller's current scale.
  • Alpha-blend the apple onto a blank canvas at the controller's position.
  • Draw an optional debug overlay (state name, depth, FPS).

No camera feed is ever drawn by this module — it only operates on
the Apple Window canvas.
"""

import cv2
import numpy as np
import os
from apple_controller import AppleController


class AppleRenderer:
    """Loads an apple sprite and composites it onto a canvas each frame."""

    def __init__(self, apple_path: str, canvas_size: tuple[int, int] = (640, 480)):
        """
        Parameters
        ----------
        apple_path : str
            Path to the RGBA apple PNG file.
        canvas_size : tuple[int, int]
            (width, height) of the Apple Window canvas.
        """
        self.canvas_w, self.canvas_h = canvas_size

        # Load with alpha channel (BGRA)
        raw = cv2.imread(apple_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot load apple image: {apple_path}")

        # Ensure 4 channels
        if raw.shape[2] == 3:
            # No alpha — add a fully opaque channel
            alpha = np.full((*raw.shape[:2], 1), 255, dtype=raw.dtype)
            raw = np.concatenate([raw, alpha], axis=2)

        self._original = raw  # Keep original for clean rescales
        self._base_h, self._base_w = raw.shape[:2]

    # ── Main rendering method ─────────────────────────────────────────────

    def render(
        self,
        canvas: np.ndarray,
        ctrl: AppleController,
        debug: bool = True,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw the apple on *canvas* according to *ctrl* state.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image (the Apple Window background).
        ctrl : AppleController
            Current apple state (position, scale, FSM state).
        debug : bool
            Whether to draw state/debug text.
        fps : float
            Current FPS for the debug overlay.

        Returns
        -------
        np.ndarray
            The canvas with the apple composited onto it.
        """
        # ── Resize apple to current scale ─────────────────────────────
        new_w = max(1, int(self._base_w * ctrl.scale))
        new_h = max(1, int(self._base_h * ctrl.scale))

        interp = cv2.INTER_AREA if ctrl.scale < 1.0 else cv2.INTER_LINEAR
        sprite = cv2.resize(self._original, (new_w, new_h), interpolation=interp)

        # ── Compute top-left position (centre the sprite on (x, y)) ───
        cx = int(ctrl.x * self.canvas_w)
        cy = int(ctrl.y * self.canvas_h)
        x1 = cx - new_w // 2
        y1 = cy - new_h // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        # ── Clamp to canvas bounds ────────────────────────────────────
        # Source region in sprite that actually overlaps the canvas
        sx1 = max(0, -x1)
        sy1 = max(0, -y1)
        sx2 = new_w - max(0, x2 - self.canvas_w)
        sy2 = new_h - max(0, y2 - self.canvas_h)

        # Destination region on canvas
        dx1 = max(0, x1)
        dy1 = max(0, y1)
        dx2 = min(self.canvas_w, x2)
        dy2 = min(self.canvas_h, y2)

        # Only draw if there is an overlapping region
        if dx1 < dx2 and dy1 < dy2 and sx1 < sx2 and sy1 < sy2:
            patch = sprite[sy1:sy2, sx1:sx2]
            self._alpha_blend(canvas, patch, dy1, dy2, dx1, dx2)

        # ── Debug overlay ─────────────────────────────────────────────
        if debug:
            self._draw_debug(canvas, ctrl, fps)

        return canvas

    # ── Alpha blending ────────────────────────────────────────────────────

    @staticmethod
    def _alpha_blend(
        canvas: np.ndarray,
        patch: np.ndarray,
        y1: int, y2: int,
        x1: int, x2: int,
    ):
        """Composite *patch* (BGRA) onto *canvas* (BGR) at [y1:y2, x1:x2].

        Uses the alpha channel of the patch for per-pixel transparency.
        """
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        fg = patch[:, :, :3].astype(np.float32)
        bg = canvas[y1:y2, x1:x2].astype(np.float32)

        blended = fg * alpha + bg * (1.0 - alpha)
        canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    # ── Debug text ────────────────────────────────────────────────────────

    @staticmethod
    def _draw_debug(canvas: np.ndarray, ctrl: AppleController, fps: float):
        """Draw state information at the top-left of the canvas."""
        font  = cv2.FONT_HERSHEY_SIMPLEX
        color = (220, 220, 220)
        lines = [
            f"State: {ctrl.state.name}",
            f"Pos:   ({ctrl.x:.2f}, {ctrl.y:.2f})",
            f"Depth: {ctrl.z:.2f}   Scale: {ctrl.scale:.2f}",
            f"FPS:   {fps:.1f}",
        ]
        for i, line in enumerate(lines):
            y = 25 + i * 25
            # Shadow for readability
            cv2.putText(canvas, line, (11, y + 1), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), font, 0.55, color, 1, cv2.LINE_AA)
