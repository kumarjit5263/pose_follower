"""Pure-Python detection & decision logic. Must NOT import cv2 or mediapipe
(enforced by tests/test_layering.py)."""
from .actions import Action, ActionKind, ActionLog, format_report
from .config import DEFAULT_SETTINGS, Settings
from .gestures import Gesture, GesturePhase, GestureToggle, HandTracker, classify_hand, \
    gesture_pair, pair_name
from .landmark_file import FrameRecord, Landmark, LandmarkFile, LandmarkFileWriter, \
    is_landmark_file
from .paths import OUTPUTS_DIR, PROJECT_ROOT, SCENARIO_DIR
from .scenarios import DEFAULT_SCENARIO_DIR, VIDEO_EXTENSIONS, find_scenarios, \
    is_video_file, scenario_stem
from .tracking import FrameResult, GestureOption, PoseFollower, Region, ResumeTarget, Status, \
    gesturer_target, hands_of_person, pose_matches_target, shoulder_width

__all__ = [
    "Action", "ActionKind", "ActionLog", "format_report",
    "DEFAULT_SETTINGS", "Settings",
    "Gesture", "GesturePhase", "GestureToggle", "HandTracker", "classify_hand", "gesture_pair",
    "pair_name",
    "FrameRecord", "Landmark", "LandmarkFile", "LandmarkFileWriter", "is_landmark_file",
    "OUTPUTS_DIR", "PROJECT_ROOT", "SCENARIO_DIR",
    "DEFAULT_SCENARIO_DIR", "VIDEO_EXTENSIONS", "find_scenarios", "is_video_file",
    "scenario_stem",
    "FrameResult", "GestureOption", "PoseFollower", "Region", "ResumeTarget", "Status",
    "gesturer_target", "hands_of_person", "pose_matches_target", "shoulder_width",
]
