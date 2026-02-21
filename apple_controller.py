"""
apple_controller.py — High-level controller that ties the state machine,
gesture input, and apple properties together.

The controller owns:
  • The AppleStateMachine
  • Apple position  (x, y)  in normalised [0, 1] coordinates
  • Apple depth     (z)     virtual depth value
  • Apple scale             rendering multiplier

Each frame the caller invokes ``update(gesture, hand_data)`` and the
controller advances the FSM and smoothly adjusts the apple's properties.
"""

from state_machine import AppleStateMachine, AppleState
from hand_tracker import HandData
from utils import (
    lerp, clamp,
    DEFAULT_APPLE_X, DEFAULT_APPLE_Y, DEFAULT_APPLE_Z,
    DEFAULT_SCALE, MIN_SCALE, MAX_SCALE,
    POSITION_LERP, SCALE_LERP, PULL_SCALE_STEP,
    IDLE_TIMEOUT_FRAMES,
)


class AppleController:
    """Manages all properties of the virtual apple and drives its FSM."""

    def __init__(self):
        self.fsm = AppleStateMachine()

        # Position in normalised [0, 1] space
        self.x: float = DEFAULT_APPLE_X
        self.y: float = DEFAULT_APPLE_Y

        # Virtual depth (1.0 = far, 0.0 = very close)
        self.z: float = DEFAULT_APPLE_Z

        # Visual scale multiplier
        self.scale: float = DEFAULT_SCALE

        # Frames since we last saw a hand — used for idle timeout
        self._no_hand_frames: int = 0

    # ── Convenience ───────────────────────────────────────────────────────

    @property
    def state(self) -> AppleState:
        return self.fsm.state

    # ── Main per-frame update ─────────────────────────────────────────────

    def update(self, gesture: str, hand_data: "HandData | None"):
        """Called once per frame with the detected gesture and hand data.

        Drives the state machine and updates position / scale / depth.
        """
        # ── Handle hand loss ──────────────────────────────────────────
        if hand_data is None:
            self._no_hand_frames += 1
            if self._no_hand_frames >= IDLE_TIMEOUT_FRAMES:
                self._reset_to_idle()
            return

        self._no_hand_frames = 0

        # ── State transitions based on gesture ────────────────────────
        current = self.fsm.state

        if current == AppleState.IDLE:
            if gesture == "open_hand":
                self.fsm.transition(AppleState.RESPONDING)

        elif current == AppleState.RESPONDING:
            if gesture == "pull":
                self.fsm.transition(AppleState.PULLING)
            elif gesture == "none":
                # Lost the open hand — go back to idle
                self.fsm.transition(AppleState.IDLE)

        elif current == AppleState.PULLING:
            if gesture == "grab":
                if self.fsm.transition(AppleState.GRABBED):
                    # SNAP POSITION INSTEAD OF LERPING
                    if hand_data:
                        self.x = hand_data.palm_center[0]
                        self.y = hand_data.palm_center[1]
                # Immediately move to FOLLOWING
                self.fsm.transition(AppleState.FOLLOWING)
            elif gesture not in ("pull", "open_hand"):
                # Lost pull gesture — reset
                self.fsm.transition(AppleState.IDLE)

        elif current == AppleState.GRABBED:
            # GRABBED is a transient state that auto-advances to FOLLOWING
            self.fsm.transition(AppleState.FOLLOWING)

        elif current == AppleState.FOLLOWING:
            if gesture == "grab":
                # Still holding — keep following
                pass
            else:
                # Released — drop back to idle
                self.fsm.transition(AppleState.IDLE)

        # ── Update apple properties based on *new* state ──────────────
        self._update_properties(gesture, hand_data)

    # ── Property updates ──────────────────────────────────────────────────

    def _update_properties(self, gesture: str, hd: HandData):
        """Smooth position / scale / depth adjustments per state."""
        state = self.fsm.state

        if state == AppleState.RESPONDING:
            # Gently nudge scale up to indicate responsiveness
            target_scale = DEFAULT_SCALE * 1.1
            self.scale = lerp(self.scale, target_scale, SCALE_LERP)

        elif state == AppleState.PULLING:
            # Apple comes closer: depth decreases, scale increases
            self.z = max(0.0, self.z - 0.01)
            self.scale = clamp(self.scale + PULL_SCALE_STEP, MIN_SCALE, MAX_SCALE)

        elif state == AppleState.FOLLOWING:
            # Apple tracks the hand position INSTANTLY once grabbed
            target_x = hd.palm_center[0]
            target_y = hd.palm_center[1]

            self.x = lerp(self.x, target_x, 1.0) # Instant tracking
            self.y = lerp(self.y, target_y, 1.0)

        elif state == AppleState.IDLE:
            # Smoothly drift back to default position / scale
            self.x = lerp(self.x, DEFAULT_APPLE_X, POSITION_LERP * 0.5)
            self.y = lerp(self.y, DEFAULT_APPLE_Y, POSITION_LERP * 0.5)
            self.scale = lerp(self.scale, DEFAULT_SCALE, SCALE_LERP)
            self.z = lerp(self.z, DEFAULT_APPLE_Z, SCALE_LERP)

    # ── Reset ─────────────────────────────────────────────────────────────

    def _reset_to_idle(self):
        """Force FSM to IDLE and begin drifting properties back to defaults."""
        self.fsm.reset()
