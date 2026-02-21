"""
main.py — Entry point for the gesture-controlled virtual apple system.

Two-hand interaction:
  • RIGHT hand: grab gesture to drag the Apple Window around the desktop.
  • LEFT hand: open palm triggers apple transfer (closes Apple Window,
    renders apple on right hand's palm in Camera Window).

Press ESC or close the Camera Window to quit.
"""

import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    CAMERA_WINDOW, APPLE_WINDOW, TARGET_FPS,
    WIN_START_X, WIN_GAP_FAR, WIN_GAP_MIN, WIN_MOVE_SPEED, WIN_DRAG_SENSITIVITY,
    distance, CAMERA_WIN_W, CAMERA_WIN_H,
)
from hand_tracker import HandTracker
from gesture_detector import GestureDetector
from apple_controller import AppleController
from renderer import AppleRenderer


def _draw_camera_hud(frame: np.ndarray, gesture: str, hand_data):
    """Draw gesture label and hand info on the camera frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_map = {
        "none":      (128, 128, 128),
        "open_hand": (0, 255, 128),
        "pull":      (0, 200, 255),
        "grab":      (0, 100, 255),
    }
    color = color_map.get(gesture, (255, 255, 255))
    label = f"Gesture: {gesture.upper()}"

    cv2.putText(frame, label, (11, 31), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 30), font, 0.7, color, 2, cv2.LINE_AA)

    if hand_data is not None:
        depth_text = f"Depth Z: {hand_data.avg_z:.4f}  Palm W: {hand_data.palm_width:.3f}"
        cv2.putText(frame, depth_text, (11, 61), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, depth_text, (10, 60), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    # ── Locate apple sprite ───────────────────────────────────────────
    project_dir = os.path.dirname(os.path.abspath(__file__))
    apple_path = os.path.join(project_dir, "assets", "apple.png")
    if not os.path.isfile(apple_path):
        print(f"[ERROR] Apple sprite not found at {apple_path}")
        sys.exit(1)

    # ── Open webcam ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        sys.exit(1)

    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        cap.release()
        sys.exit(1)

    # Determine target camera size: use configured size if provided,
    # otherwise fall back to the native frame size.
    native_h, native_w = test_frame.shape[:2]
    cam_w = CAMERA_WIN_W if CAMERA_WIN_W is not None else native_w
    cam_h = CAMERA_WIN_H if CAMERA_WIN_H is not None else native_h

    # Resize the initial frame to the target camera size so UI elements
    # (windows, renderer) use the desired aspect ratio.
    if (cam_w, cam_h) != (native_w, native_h):
        test_frame = cv2.resize(test_frame, (int(cam_w), int(cam_h)), interpolation=cv2.INTER_LINEAR)

    canvas_size = (cam_w, cam_h)

    # ── Initialise components ─────────────────────────────────────────
    tracker    = HandTracker()
    detector   = GestureDetector()
    controller = AppleController()
    renderer   = AppleRenderer(apple_path, canvas_size)

    # ── Create and Position Windows ───────────────────────────────────
    current_cam_x = WIN_START_X
    current_cam_y = 100.0
    apple_win_x   = float(current_cam_x + cam_w + WIN_GAP_FAR)
    apple_win_y   = 100.0

    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WINDOW, int(cam_w), int(cam_h))
    cv2.moveWindow(CAMERA_WINDOW, int(current_cam_x), int(current_cam_y))

    apple_window_visible = True
    cv2.namedWindow(APPLE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(APPLE_WINDOW, int(apple_win_x), int(apple_win_y))

    print("[INFO] System ready.")
    print("       RIGHT hand: grab to drag the Apple Window.")
    print("       LEFT hand: open palm to transfer the apple.")
    print("       Press ESC to quit.")

    # ── FPS tracking ──────────────────────────────────────────────────
    fps = 0.0
    prev_time = time.perf_counter()
    frame_times: list[float] = []

    # Track hand position for dragging
    prev_hand_x = 0.5
    prev_hand_y = 0.5

    # ══════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # ── 1. Two-hand tracking ──────────────────────────────────────
        hands = tracker.process_and_draw_two(frame)
        right_hand = hands.get('Right')
        left_hand  = hands.get('Left')

        # ── 2. Gesture detection (RIGHT hand) ─────────────────────────
        if right_hand is not None:
            gesture = detector.detect(right_hand)
        else:
            gesture = "none"
            detector.reset()

        # ── 3. Left hand open-palm detection ──────────────────────────
        left_palm_open = False
        if left_hand is not None:
            lm = left_hand.all_landmarks
            tips = [8, 12, 16, 20]
            mcps = [5, 9, 13, 17]
            left_palm_open = all(lm[t].y < lm[m].y for t, m in zip(tips, mcps))

            # If left hand is shown close to the hand that grabbed, treat as transfer
            # (user shows left open palm near grabbing hand to trigger transfer).
            HAND_TRANSFER_DIST = 0.15
            left_close_to_right = False
            if right_hand is not None:
                inter_hand_dist = distance(right_hand.palm_center, left_hand.palm_center)
                left_close_to_right = inter_hand_dist < HAND_TRANSFER_DIST
            else:
                left_close_to_right = False

        # ── 4. State & proximity ──────────────────────────────────────
        from state_machine import AppleState
        is_following = controller.state in (AppleState.GRABBED, AppleState.FOLLOWING)

        current_gap = apple_win_x - (current_cam_x + cam_w)
        is_snapped = current_gap < 100

        # ── 5. Window dragging (RIGHT hand grab) ─────────────────────
        if gesture == "grab" and right_hand and not is_following:
            dx = right_hand.palm_center[0] - prev_hand_x
            dy = right_hand.palm_center[1] - prev_hand_y

            apple_win_x += dx * cam_w * WIN_DRAG_SENSITIVITY
            apple_win_y += dy * cam_h * WIN_DRAG_SENSITIVITY

            apple_win_x = max(-cam_w // 2, apple_win_x)
            apple_win_y = max(0, apple_win_y)

            if apple_window_visible:
                cv2.moveWindow(APPLE_WINDOW, int(apple_win_x), int(apple_win_y))

        # ── Camera move: thumb-right closed-palm gesture moves the Camera
        # window towards the Apple Window for easier interaction.
        if gesture == "thumb_right" and right_hand and not is_following:
            # Move the Camera Window towards the Apple Window in 2D (not only
            # linear X). Compute a target so the apple window sits just after
            # the camera and align vertically with the apple window.
            target_cam_x = apple_win_x - cam_w - WIN_GAP_MIN
            target_cam_y = apple_win_y

            dx = target_cam_x - current_cam_x
            dy = target_cam_y - current_cam_y
            dist = (dx * dx + dy * dy) ** 0.5

            if dist <= WIN_MOVE_SPEED or dist == 0:
                current_cam_x = target_cam_x
                current_cam_y = target_cam_y
            else:
                # Move along the direction vector, capped by WIN_MOVE_SPEED
                nx = dx / dist
                ny = dy / dist
                current_cam_x += nx * WIN_MOVE_SPEED
                current_cam_y += ny * WIN_MOVE_SPEED

            cv2.moveWindow(CAMERA_WINDOW, int(current_cam_x), int(current_cam_y))

        if right_hand:
            prev_hand_x = right_hand.palm_center[0]
            prev_hand_y = right_hand.palm_center[1]

        # Recalculate snap after potential movement
        current_gap = apple_win_x - (current_cam_x + cam_w)
        is_snapped = current_gap < 100

        # ── 6. Apple transfer logic ───────────────────────────────────
        # LEFT palm open + snapped = transfer apple to right hand
        effective_gesture = gesture

        # If the left open palm is shown and we have a right hand, prioritize
        # an immediate transfer to the right hand (close Apple Window and
        # render the apple on the right-hand palm). This shortcut triggers
        # even if the apple window isn't snapped.
        if left_palm_open and right_hand and not is_following:
            effective_gesture = "transfer"
        else:
            # If the right hand is currently grabbing (dragging) the window and the
            # user shows an open left palm nearby, trigger an immediate transfer
            # command that gives the apple to the grabbing hand.
            if right_hand and gesture == "grab" and left_palm_open and left_close_to_right and not is_following:
                effective_gesture = "transfer"
            else:
                # Transfer either when left palm is open near the apple window (snapped)
                # or when left palm is open close to the grabbing hand.
                if (left_palm_open and is_snapped and not is_following) or (left_palm_open and left_close_to_right and not is_following):
                    effective_gesture = "grab"

        # If the user grabs but the apple window isn't snapped to the camera,
        # suppress the gesture for the controller (used for window dragging).
        if gesture == "grab" and not is_snapped:
            effective_gesture = "none"

        controller.update(effective_gesture, right_hand)

        # ── 7. Re-read state after controller update ──────────────────
        is_following = controller.state in (AppleState.GRABBED, AppleState.FOLLOWING)

        # ── 8. Window lifecycle ───────────────────────────────────────
        if is_following:
            if apple_window_visible:
                cv2.destroyWindow(APPLE_WINDOW)
                apple_window_visible = False
        else:
            if not apple_window_visible:
                cv2.namedWindow(APPLE_WINDOW, cv2.WINDOW_NORMAL)
                cv2.moveWindow(APPLE_WINDOW, int(apple_win_x), int(apple_win_y))
                apple_window_visible = True

        # ── 9. Render ─────────────────────────────────────────────────
        if is_following:
            frame = renderer.render(frame, controller, debug=False)
            canvas = None
        else:
            canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            canvas = renderer.render(canvas, controller, debug=True, fps=fps)

        # ── 10. HUD & display ─────────────────────────────────────────
        # Prefer showing the right-hand gesture; if no right hand is
        # present but the left open palm is detected, show that instead.
        if right_hand is not None:
            gesture_display = gesture
            display_hand = right_hand
        elif left_palm_open:
            gesture_display = "open_hand"
            display_hand = left_hand
        else:
            gesture_display = "none"
            display_hand = None

        _draw_camera_hud(frame, gesture_display, display_hand)

        cv2.imshow(CAMERA_WINDOW, frame)
        if apple_window_visible and canvas is not None:
            cv2.imshow(APPLE_WINDOW, canvas)

        # ── 11. FPS ───────────────────────────────────────────────────
        now = time.perf_counter()
        frame_times.append(now - prev_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
        prev_time = now

        # ── 12. Exit conditions ───────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if cv2.getWindowProperty(CAMERA_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if apple_window_visible and cv2.getWindowProperty(APPLE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    print("[INFO] Shutting down...")
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
