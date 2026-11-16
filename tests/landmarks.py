"""Test-only helpers: synthetic MediaPipe-shaped hand and pose landmarks.

Coordinates are normalised image coordinates (0..1, y pointing down), like
MediaPipe's. ``aspect`` (= width / height) keeps shapes correct in *pixel*
space on non-square frames: x offsets are divided by it.
No cv2 / mediapipe needed.
"""
import math

from follower.logic import Gesture, Landmark

LM = Landmark  # short alias used throughout the tests

# Hand shapes in "hand units", wrist at the origin, fingers pointing up (-y).
# Index layout follows MediaPipe: 0 wrist, 1-4 thumb, then 4 joints
# (MCP, PIP, DIP, TIP) for each of index 5-8, middle 9-12, ring 13-16, pinky 17-20.
_FINGER_X = [-0.15, -0.05, 0.05, 0.15]          # index, middle, ring, pinky
_EXTENDED = [-0.40, -0.60, -0.72, -0.85]        # MCP, PIP, DIP, TIP  (y)
_CURLED = [-0.40, -0.55, -0.42, -0.30]          # tip folds back toward the wrist
_THUMB_OUT = [(-0.12, -0.10), (-0.25, -0.18), (-0.38, -0.24), (-0.50, -0.28)]
_THUMB_IN = [(-0.10, -0.10), (-0.15, -0.20), (-0.15, -0.30), (-0.12, -0.38)]   # over the index
_THUMB_ACROSS = [(-0.10, -0.10), (-0.08, -0.22), (-0.02, -0.32), (0.05, -0.40)]  # on the ring


def _hand_shape(gesture, thumb=None):
    """Finger layout for a gesture; ``thumb`` ("in" / "out") overrides the thumb."""
    thumbs = {"in": _THUMB_IN, "out": _THUMB_OUT}
    if gesture is Gesture.PALM:
        fingers, th = [_EXTENDED] * 4, _THUMB_OUT
    elif gesture is Gesture.FIST:
        fingers, th = [_CURLED] * 4, _THUMB_IN
    elif gesture is Gesture.SCISSORS:
        fingers, th = [_EXTENDED, _EXTENDED, _CURLED, _CURLED], _THUMB_ACROSS
    else:  # pointing: index out, the rest curled -> classifies as NONE
        fingers, th = [_EXTENDED, _CURLED, _CURLED, _CURLED], _THUMB_IN
    if thumb is not None:
        th = thumbs[thumb]
    pts = [(0.0, 0.0)] + list(th)
    for fx, ys in zip(_FINGER_X, fingers):
        pts += [(fx, y) for y in ys]
    return pts


def make_hand(gesture, x=0.5, y=0.6, size=0.15, angle_deg=0.0, aspect=1.0, thumb=None):
    """21 hand landmarks showing ``gesture`` with the wrist at (x, y).
    ``thumb="in"/"out"`` overrides the gesture's usual thumb position."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [LM(x + size * (px * ca - py * sa) / aspect, y + size * (px * sa + py * ca))
            for px, py in _hand_shape(gesture, thumb)]


# Pose skeleton in shoulder-width units, origin between the shoulders, +x = the
# person's left (image right when not mirrored), +y down. (dx, dy) per landmark.
_HEAD = {0: (0, -.55), 1: (.06, -.62), 2: (.10, -.63), 3: (.14, -.62), 4: (-.06, -.62),
         5: (-.10, -.63), 6: (-.14, -.62), 7: (.22, -.58), 8: (-.22, -.58), 9: (.07, -.45),
         10: (-.07, -.45)}
_ARMS_DOWN = {13: (.6, .75), 15: (.5, 1.5)}      # elbow, wrist (left side; right mirrored)
_ARMS_UP = {13: (.8, .55), 15: (.75, -.2)}       # hands raised in front, as when gesturing
_LEGS = {23: (.35, 1.6), 25: (.35, 2.5), 27: (.35, 3.4), 29: (.35, 3.5), 31: (.45, 3.55)}


def make_pose(x=0.5, y=0.5, shoulder_w=0.2, visibility=1.0, arms_up=False, aspect=1.0,
              shoulder_visibility=None):
    """33 pose landmarks: shoulders centred on (x, y).

    ``arms_up`` raises the wrists to chest height (where the hands are when
    showing the gesture). ``shoulder_visibility`` overrides the shoulders'
    visibility (to simulate them being hidden).
    """
    pts = dict(_HEAD)
    pts[11], pts[12] = (.5, 0), (-.5, 0)
    arm = _ARMS_UP if arms_up else _ARMS_DOWN
    for left, (dx, dy) in {**arm, **_LEGS}.items():
        pts[left], pts[left + 1] = (dx, dy), (-dx, dy)
    for left in (17, 19, 21):                     # pinky / index / thumb near the wrist
        wx, wy = pts[15]
        pts[left], pts[left + 1] = (wx + .05, wy - .1), (-wx - .05, wy - .1)
    lm = [LM(x + dx * shoulder_w / aspect, y + dy * shoulder_w, visibility=visibility)
          for dx, dy in (pts[i] for i in range(33))]
    if shoulder_visibility is not None:
        lm[11].visibility = lm[12].visibility = shoulder_visibility
    return lm


def wrists_of(pose):
    """(left_wrist, right_wrist) positions, handy for placing hands on a person."""
    return (pose[15].x, pose[15].y), (pose[16].x, pose[16].y)


def hands_at_wrists(pose, left=Gesture.FIST, right=Gesture.PALM, size=0.12, aspect=1.0):
    """A hand on each of the person's wrists (fingers up) - by default the stop
    gesture (open palm + fist); pass e.g. ``Gesture.SCISSORS`` for resume / the
    pose-drawing toggle."""
    (lx, ly), (rx, ry) = wrists_of(pose)
    return [make_hand(left, lx, ly, size=size, aspect=aspect),
            make_hand(right, rx, ry, size=size, aspect=aspect)]
