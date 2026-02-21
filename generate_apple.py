"""
generate_apple.py — Creates a procedural apple sprite (RGBA PNG) using OpenCV.

Run this script once to generate assets/apple.png if you don't have one.
The generated apple is a simple but clean vector-style illustration with
full alpha transparency, suitable for the gesture-controlled system.
"""

import cv2
import numpy as np
import os


def generate_apple(output_path: str, size: int = 256):
    """Draw a stylised apple with alpha channel and save as PNG."""
    img = np.zeros((size, size, 4), dtype=np.uint8)  # BGRA, fully transparent

    cx, cy = size // 2, size // 2 + 10  # Apple body centre (shifted down for stem room)
    radius = size // 2 - 30

    # ── Apple body (gradient-like effect with overlapping circles) ─────
    # Base dark red
    cv2.circle(img, (cx, cy), radius, (30, 30, 180, 255), -1, cv2.LINE_AA)
    # Lighter red overlay, shifted slightly
    cv2.circle(img, (cx - 8, cy - 5), radius - 5, (40, 50, 210, 255), -1, cv2.LINE_AA)
    # Highlight (brighter red, upper-left)
    cv2.circle(img, (cx - 20, cy - 20), radius // 2, (60, 80, 240, 255), -1, cv2.LINE_AA)
    # Specular highlight (white-ish spot)
    cv2.circle(img, (cx - 30, cy - 35), radius // 5, (140, 160, 255, 200), -1, cv2.LINE_AA)

    # ── Small indentation at top (two overlapping circles for apple shape) ─
    indent_y = cy - radius + 5
    cv2.circle(img, (cx - 12, indent_y), 18, (0, 0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (cx + 12, indent_y), 18, (0, 0, 0, 0), -1, cv2.LINE_AA)

    # ── Stem ──────────────────────────────────────────────────────────
    stem_top = cy - radius - 15
    stem_bot = cy - radius + 12
    cv2.line(img, (cx + 2, stem_top), (cx + 2, stem_bot), (30, 80, 100, 255), 4, cv2.LINE_AA)

    # ── Leaf ──────────────────────────────────────────────────────────
    leaf_pts = np.array([
        [cx + 4,  stem_top + 5],
        [cx + 30, stem_top - 5],
        [cx + 35, stem_top + 10],
        [cx + 15, stem_top + 15],
    ], dtype=np.int32)
    cv2.fillPoly(img, [leaf_pts], (50, 180, 60, 255), cv2.LINE_AA)
    # Leaf vein
    cv2.line(img, (cx + 8, stem_top + 10), (cx + 30, stem_top + 3), (40, 140, 45, 200), 1, cv2.LINE_AA)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"[generate_apple] Saved {size}x{size} RGBA apple → {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(script_dir, "assets", "apple.png")
    generate_apple(out)
