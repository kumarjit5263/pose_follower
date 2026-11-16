"""Tunable parameters for the follow / gesture logic (no cv2 / mediapipe)."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Settings:
    # --- the three two-handed gestures (Gesture names; one hand shows each) ---
    stop_gesture: Tuple[str, str] = ("PALM", "FIST")          # while following: stop
    resume_gesture: Tuple[str, str] = ("PALM", "SCISSORS")    # while stopped: resume,
    #                               locking onto whoever made the gesture
    draw_toggle_gesture: Tuple[str, str] = ("FIST", "SCISSORS")  # while following: show/
    #                               hide the pose diagram on the body (tracking keeps running)

    hold_seconds: float = 3.0     # s a gesture pair must be held to take effect
    hold_confirm: float = 0.4     # s the pair must be seen before the hold is shown/logged
    #                               (momentary mis-detections never start a hold)
    release_seconds: float = 0.5  # s the hands must be down (pair not shown) before the
    #                               gesture is accepted again after a toggle
    dropout_grace: float = 0.3    # s the pair may drop out (detector flicker)
    #                               without restarting the hold timer

    # --- per-hand label smoothing ---
    label_confirm_frames: int = 2     # a hand's shown gesture changes only after the new
    label_confirm_seconds: float = 0.1  # one is seen on >= this many frames and seconds
    track_timeout: float = 0.5    # s a lost hand keeps its identity (and label)

    # --- whose hands count while following ---
    hand_near_body: float = 3.0   # max hand-to-person-wrist distance, in shoulder widths
    shoulders_per_hand: float = 4.0  # shoulder width ~= 4 x hand size (wrist -> middle
    #                                  knuckle); used when the shoulders aren't visible
    min_visibility: float = 0.5   # pose landmark visibility needed to use it

    # --- presentation / reporting ---
    flash_seconds: float = 1.5    # s the "FOLLOWING STOPPED/RESUMED" banner shows
    event_debounce: float = 0.3   # s a person must be found/lost before it is logged


DEFAULT_SETTINGS = Settings()
