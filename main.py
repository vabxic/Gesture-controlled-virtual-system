"""
main.py — Entry point for the gesture-controlled virtual apple system.

Creates two OpenCV windows:
  1. "Camera Window" — live webcam feed with hand landmarks drawn on top.
  2. "Apple Window"  — blank canvas with the virtual apple rendered on it.

The main loop:
  • Captures a frame from the webcam.
  • Processes it through HandTracker (MediaPipe) to get hand landmarks.
  • Feeds landmarks into GestureDetector to classify the current gesture.
  • Passes the gesture + hand data to AppleController, which updates the FSM
    and smoothly adjusts the apple's position / scale / depth.
  • Renders the apple on a separate canvas in the Apple Window.
  • Repeats at ~30 FPS until ESC is pressed or either window is closed.
"""

import sys
import os
import time
import cv2
import numpy as np

# Ensure the project root is on the path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import CAMERA_WINDOW, APPLE_WINDOW, TARGET_FPS
from hand_tracker import HandTracker
from gesture_detector import GestureDetector
from apple_controller import AppleController
from renderer import AppleRenderer


def main():
    # ── Locate apple sprite ───────────────────────────────────────────
    project_dir = os.path.dirname(os.path.abspath(__file__))
    apple_path  = os.path.join(project_dir, "assets", "apple.png")
    if not os.path.isfile(apple_path):
        print(f"[ERROR] Apple sprite not found at {apple_path}")
        print("        Please place an RGBA PNG at that path.")
        sys.exit(1)

    # ── Open webcam ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        sys.exit(1)

    # Read one frame to determine camera resolution
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read from webcam.")
        cap.release()
        sys.exit(1)

    cam_h, cam_w = test_frame.shape[:2]
    canvas_size = (cam_w, cam_h)  # Apple Window same size as camera

    # ── Initialise components ─────────────────────────────────────────
    tracker    = HandTracker()
    detector   = GestureDetector()
    controller = AppleController()
    renderer   = AppleRenderer(apple_path, canvas_size)

    # ── Create and Position Windows ───────────────────────────────────
    from utils import WIN_START_X, WIN_GAP_FAR, WIN_GAP_MIN, WIN_MOVE_THRESHOLD, WIN_MOVE_SPEED
    
    current_cam_x = WIN_START_X
    apple_win_x   = current_cam_x + cam_w + WIN_GAP_FAR
    
    cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(CAMERA_WINDOW, current_cam_x, 100)
    
    # Track the visibility state of the Apple Window
    apple_window_visible = True
    cv2.namedWindow(APPLE_WINDOW,  cv2.WINDOW_NORMAL)
    cv2.moveWindow(APPLE_WINDOW,  apple_win_x,   100)

    print("[INFO] System ready. Reach towards the apple window to interaction.")
    print("       Press ESC or close either window to quit.")

    # ── FPS tracking ──────────────────────────────────────────────────
    fps        = 0.0
    prev_time  = time.perf_counter()
    frame_times: list[float] = []
    
    # Track hand position for dragging (both X and Y)
    prev_hand_x = 0.5
    prev_hand_y = 0.5
    apple_win_y = 100  # Track Y position of Apple Window

    # ══════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════
    while True:
        # 1. Capture frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 2. Hand tracking + draw landmarks
        hand_data = tracker.process_and_draw(frame)

        # 3. Gesture detection
        if hand_data is not None:
            gesture = detector.detect(hand_data)
        else:
            gesture = "none"
            detector.reset()

        # 4. Compute state FIRST (needed for dragging logic)
        from state_machine import AppleState
        is_following = controller.state in (AppleState.GRABBED, AppleState.FOLLOWING)
        
        # Check if windows are close enough for apple transfer
        current_gap = apple_win_x - (current_cam_x + cam_w)
        is_snapped = current_gap < 100

        # 5. Handle Window Dragging — move Apple Window in ANY direction
        if gesture == "grab" and hand_data and not is_following:
            from utils import WIN_DRAG_SENSITIVITY
            dx = hand_data.palm_center[0] - prev_hand_x
            dy = hand_data.palm_center[1] - prev_hand_y
            
            # Move window proportional to hand movement (scaled by frame size)
            apple_win_x += dx * cam_w * WIN_DRAG_SENSITIVITY
            apple_win_y += dy * cam_h * WIN_DRAG_SENSITIVITY
            
            # Keep window on screen (basic bounds)
            apple_win_x = max(-cam_w // 2, apple_win_x)
            apple_win_y = max(0, apple_win_y)
            
            if apple_window_visible:
                cv2.moveWindow(APPLE_WINDOW, int(apple_win_x), int(apple_win_y))
        
        if hand_data:
            prev_hand_x = hand_data.palm_center[0]
            prev_hand_y = hand_data.palm_center[1]

        # Recalculate snap after potential movement
        current_gap = apple_win_x - (current_cam_x + cam_w)
        is_snapped = current_gap < 100

        # 6. Update apple controller
        # Only allow TRANSFER (grabbed/following) if windows are snapped
        effective_gesture = gesture
        is_reaching = hand_data and hand_data.palm_center[0] > WIN_MOVE_THRESHOLD
        
        if gesture == "grab" and not is_snapped:
            effective_gesture = "none"
            
        controller.update(effective_gesture, hand_data)

        # 7. Re-read state after controller update
        is_following = controller.state in (AppleState.GRABBED, AppleState.FOLLOWING)
        
        # ── Window Lifecycle Management ──────────────────────────────
        if is_following:
            if apple_window_visible:
                cv2.destroyWindow(APPLE_WINDOW)
                apple_window_visible = False
        else:
            if not apple_window_visible:
                cv2.namedWindow(APPLE_WINDOW, cv2.WINDOW_NORMAL)
                cv2.moveWindow(APPLE_WINDOW, int(apple_win_x), int(apple_win_y))
                apple_window_visible = True
        
        # 8. Render Apple Transfer
        if is_following:
            frame = renderer.render(frame, controller, debug=False)
            canvas = None
        else:
            canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            canvas = renderer.render(canvas, controller, debug=True, fps=fps)

        # 9. Camera HUD
        _draw_camera_hud(frame, gesture, hand_data)

        # 10. Show windows
        cv2.imshow(CAMERA_WINDOW, frame)
        if apple_window_visible and canvas is not None:
            cv2.imshow(APPLE_WINDOW, canvas)

        # 8. FPS calculation (rolling average of last 30 frames)
        now = time.perf_counter()
        frame_times.append(now - prev_time)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
        prev_time = now

        # 9. Exit conditions
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        # Also exit if either window has been closed by the user
        if cv2.getWindowProperty(CAMERA_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if apple_window_visible and cv2.getWindowProperty(APPLE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    print("[INFO] Shutting down…")
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()


# ── Helper: draw gesture info on the camera feed ──────────────────────────

def _draw_camera_hud(frame: np.ndarray, gesture: str, hand_data):
    """Draw gesture label and hand info on the camera frame."""
    font  = cv2.FONT_HERSHEY_SIMPLEX
    color_map = {
        "none":      (128, 128, 128),
        "open_hand": (0, 255, 128),
        "pull":      (0, 200, 255),
        "grab":      (0, 100, 255),
    }
    color = color_map.get(gesture, (255, 255, 255))
    label = f"Gesture: {gesture.upper()}"

    # Shadow + text
    cv2.putText(frame, label, (11, 31), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 30), font, 0.7, color, 2, cv2.LINE_AA)

    if hand_data is not None:
        depth_text = f"Depth Z: {hand_data.avg_z:.4f}  Palm W: {hand_data.palm_width:.3f}"
        cv2.putText(frame, depth_text, (11, 61), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, depth_text, (10, 60), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# Make _draw_camera_hud accessible as a module-level function
# (it was mistakenly referenced as self._draw_camera_hud in the loop;
#  we fix that by making it a plain function call)


if __name__ == "__main__":
    main()
