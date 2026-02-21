"""
state_machine.py — Finite State Machine for the virtual apple.

States:
    IDLE        → Apple is resting; no interaction.
    RESPONDING  → An open hand has been detected; system is listening.
    PULLING     → Forward-pull gesture detected; apple is being drawn closer.
    GRABBED     → Pinch/grab confirmed; apple attaches to hand.
    FOLLOWING   → Apple is actively following the hand's position.

Valid transitions are enforced — any invalid transition is silently
blocked (and logged to stdout for debugging).
"""

from enum import Enum, auto


class AppleState(Enum):
    IDLE       = auto()
    RESPONDING = auto()
    PULLING    = auto()
    GRABBED    = auto()
    FOLLOWING  = auto()


# ─── Transition table ─────────────────────────────────────────────────────────
# Key = current state, Value = set of states we may legally move to.
_TRANSITIONS: dict[AppleState, set[AppleState]] = {
    AppleState.IDLE:       {AppleState.RESPONDING},
    AppleState.RESPONDING: {AppleState.PULLING, AppleState.IDLE},
    AppleState.PULLING:    {AppleState.GRABBED, AppleState.IDLE},
    AppleState.GRABBED:    {AppleState.FOLLOWING, AppleState.IDLE},
    AppleState.FOLLOWING:  {AppleState.IDLE},
}


class AppleStateMachine:
    """Manages the apple's interaction state with strict transition rules."""

    def __init__(self):
        self._state: AppleState = AppleState.IDLE

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def state(self) -> AppleState:
        return self._state

    def transition(self, new_state: AppleState) -> bool:
        """Attempt to transition to *new_state*.

        Returns True if the transition was valid and applied,
        False if it was blocked.
        """
        if new_state == self._state:
            return True  # No-op, staying in the same state is always OK

        allowed = _TRANSITIONS.get(self._state, set())
        if new_state in allowed:
            # print(f"[FSM] {self._state.name} → {new_state.name}")  # Debug
            self._state = new_state
            return True

        # Invalid transition — block it
        # print(f"[FSM] BLOCKED {self._state.name} → {new_state.name}")  # Debug
        return False

    def reset(self):
        """Force-reset to IDLE (used on hand-loss timeout)."""
        self._state = AppleState.IDLE

    def __repr__(self) -> str:
        return f"AppleStateMachine(state={self._state.name})"
