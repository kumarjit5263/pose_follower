"""Hand-gesture classification, per-hand label smoothing, and the hold-to-toggle
detector.

Pure Python. Landmarks are duck-typed: anything with ``.x`` / ``.y`` / ``.z``
(and ``.visibility`` for pose landmarks) works - MediaPipe's NormalizedLandmark
in production, ``follower.logic.landmark_file.Landmark`` when replaying a file.

Coordinates are MediaPipe's normalised image coordinates: x is a fraction of
the image *width*, y of the *height*, and a hand's z (depth) is on roughly the
same scale as x. On a non-square frame those units differ, so every distance
takes ``aspect = width / height`` and scales x (and z) by it, making distances
proportional to real pixel distances.
"""
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .config import DEFAULT_SETTINGS, Settings

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9
# (MCP, PIP, TIP) for index, middle, ring, pinky
FINGERS = [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]
# joints the thumb tip rests near when it is tucked in (fist: across the index /
# middle fingers; scissors: holding the ring / pinky down)
TUCK_POINTS = (6, 7, 9, 10, 11, 13, 14, 15, 18, 19)

# Thresholds, measured on four recorded live sessions (finger bends are strongly
# bimodal: straight < 30 deg, curled > 100 deg; a tucked thumb tip is < 0.25 hand
# sizes from the fingers, an extended one ~0.7).
EXTENDED_MAX_BEND = 45.0   # deg: finger counts as straight below this
CURLED_MIN_BEND = 80.0     # deg: finger counts as curled above this
THUMB_TUCKED_MAX = 0.35    # thumb tip to nearest finger joint, in hand sizes

# A pending gesture must be confirmed within this window (s), NONE frames allowed in between.
CANDIDATE_WINDOW = 0.5

# Tolerance for time comparisons, so e.g. 84/30 - 54/30 (= 0.99999...) counts as 1 s.
TIME_EPS = 1e-6


class Gesture(Enum):
    NONE = 0
    FIST = 1       # all four fingers curled, thumb tucked in
    PALM = 2       # all four fingers straight, thumb out
    SCISSORS = 3   # index + middle straight, ring + pinky curled, thumb tucked in


GESTURE_NAMES = {Gesture.NONE: "none", Gesture.FIST: "fist", Gesture.PALM: "open palm",
                 Gesture.SCISSORS: "scissors"}


def gesture_pair(settings: Settings, attr: str) -> Tuple[Gesture, Gesture]:
    """Resolve a ``Settings`` field (e.g. ``"stop_gesture"``) to a pair of Gesture."""
    a, b = getattr(settings, attr)
    return Gesture[a], Gesture[b]


def pair_name(pair: Tuple[Gesture, Gesture]) -> str:
    """e.g. 'open palm + fist'."""
    return " + ".join(GESTURE_NAMES[g] for g in pair)


# ---- geometry ------------------------------------------------------------------
def dist(a, b, aspect=1.0):
    """2-D distance between two landmarks, in units of the image height."""
    return math.hypot((a.x - b.x) * aspect, a.y - b.y)


def _p3(p, aspect):
    return (p.x * aspect, p.y, getattr(p, "z", 0.0) * aspect)


def _dist3(a, b):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def _angle(u, v):
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = sum(x * y for x, y in zip(u, v)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def hand_size(lm, aspect=1.0):
    """Wrist -> middle-finger knuckle distance (2-D): a stable measure of hand scale."""
    return dist(lm[WRIST], lm[MIDDLE_MCP], aspect)


def finger_bends(lm, aspect=1.0) -> List[float]:
    """Bend of index, middle, ring, pinky in degrees (0 = straight, 180 = folded
    back): the angle between knuckle->middle joint and middle joint->tip, in 3-D
    so fingers pointing at the camera still read correctly."""
    p = [_p3(q, aspect) for q in lm]
    return [_angle([b - a for a, b in zip(p[m], p[j])], [b - a for a, b in zip(p[j], p[t])])
            for m, j, t in FINGERS]


def thumb_tuck(lm, aspect=1.0) -> float:
    """Thumb-tip distance to the nearest finger joint, in hand sizes (3-D).
    Small = thumb tucked in (fist / scissors); large = thumb out (open palm)."""
    p = [_p3(q, aspect) for q in lm]
    size = _dist3(p[WRIST], p[MIDDLE_MCP]) or 1e-6
    return min(_dist3(p[THUMB_TIP], p[i]) for i in TUCK_POINTS) / size


def classify_hand(lm, aspect=1.0) -> Gesture:
    """Classify one hand (21 landmarks) as PALM / FIST / SCISSORS / NONE.

    Rotation- and scale-invariant (angles and ratios only)."""
    bends = finger_bends(lm, aspect)
    straight = [b < EXTENDED_MAX_BEND for b in bends]
    curled = [b > CURLED_MIN_BEND for b in bends]
    tucked = thumb_tuck(lm, aspect) < THUMB_TUCKED_MAX
    if all(straight) and not tucked:
        return Gesture.PALM
    if all(curled) and tucked:
        return Gesture.FIST
    if straight[0] and straight[1] and curled[2] and curled[3] and tucked:
        return Gesture.SCISSORS
    return Gesture.NONE


# ---- per-hand tracking + label smoothing ------------------------------------------
@dataclass
class _Track:
    wrist: Tuple[float, float]
    size: float
    last_seen: float
    label: Gesture = Gesture.NONE           # the smoothed (shown) gesture
    candidate: Optional[Gesture] = None     # a different gesture waiting for confirmation
    candidate_since: float = 0.0
    candidate_frames: int = 0
    mismatch_since: float = 0.0             # raw gesture has differed from label since...
    mismatch_frames: int = 0                # ...for this many consecutive frames


def _same_hand(a, b, aspect):
    """True if two detections are the same physical hand (duplicate detection)."""
    size = max(hand_size(a, aspect), hand_size(b, aspect), 1e-6)
    return (dist(a[WRIST], b[WRIST], aspect) < 0.5 * size and
            dist(a[MIDDLE_MCP], b[MIDDLE_MCP], aspect) < 0.5 * size)


class HandTracker:
    """Follows hands from frame to frame and smooths their gesture labels.

    - Duplicate detections of one hand are dropped (the first one is kept).
    - Each hand is matched to the nearest hand of the previous frames.
    - A hand's shown gesture only changes once the new gesture has been seen on
      ``label_confirm_frames`` consecutive frames spanning ``label_confirm_seconds``
      - so single-frame mis-classifications (a fist flickering to NONE, a hand
      that briefly looks open while closing) never show up.
    - If the hand stops looking like its shown gesture for that long but keeps
      changing (e.g. mid-way through closing), the shown gesture drops to NONE.
    - ``update`` also reports the *confirmed* gesture: the shown one if the
      current frame agrees with it, else NONE. Only confirmed gestures count
      towards the hold, so a stale label can never combine with another hand's
      new one into a false gesture pair.
    """

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        self._tracks: List[_Track] = []

    def update(self, hands: Sequence, now: float, aspect: float = 1.0
               ) -> List[Tuple[int, Gesture, Gesture]]:
        """Returns (index into ``hands``, shown gesture, confirmed gesture) for every
        distinct hand."""
        s = self.settings
        keep = []
        for i, h in enumerate(hands):
            if not any(_same_hand(h, hands[j], aspect) for j in keep):
                keep.append(i)

        self._tracks = [t for t in self._tracks if now - t.last_seen <= s.track_timeout]
        pairs = sorted(
            (math.hypot((hands[i][WRIST].x * aspect - t.wrist[0]), hands[i][WRIST].y - t.wrist[1]), i, k)
            for i in keep for k, t in enumerate(self._tracks))
        matched, used = {}, set()
        for d, i, k in pairs:
            if i in matched or k in used:
                continue
            if d <= 1.5 * max(self._tracks[k].size, hand_size(hands[i], aspect)):
                matched[i] = k
                used.add(k)

        out = []
        for i in keep:
            h = hands[i]
            raw = classify_hand(h, aspect)
            if i in matched:
                t = self._tracks[matched[i]]
            else:
                t = _Track(wrist=(0, 0), size=0, last_seen=now)
                self._tracks.append(t)
            t.wrist, t.size, t.last_seen = (h[WRIST].x * aspect, h[WRIST].y), hand_size(h, aspect), now
            self._smooth(t, raw, now)
            out.append((i, t.label, t.label if raw is t.label else Gesture.NONE))
        return out

    def _confirmed(self, since, frames, now):
        s = self.settings
        return frames >= s.label_confirm_frames and \
            now - since >= s.label_confirm_seconds - TIME_EPS

    def _smooth(self, t: _Track, raw: Gesture, now: float):
        """NONE means "unsure", so NONE frames neither confirm nor cancel a pending
        gesture; a different concrete gesture does cancel it."""
        if raw is t.label:
            t.mismatch_frames = 0
            if raw is not Gesture.NONE:
                t.candidate = None
            return
        if t.mismatch_frames == 0:
            t.mismatch_since = now
        t.mismatch_frames += 1
        if raw is not Gesture.NONE:
            if raw is t.candidate and now - t.candidate_since <= CANDIDATE_WINDOW + TIME_EPS:
                t.candidate_frames += 1
            else:
                t.candidate, t.candidate_since, t.candidate_frames = raw, now, 1
        if t.candidate is not None and t.candidate is not t.label and \
                now - t.candidate_since <= CANDIDATE_WINDOW + TIME_EPS and \
                self._confirmed(t.candidate_since, t.candidate_frames, now):
            t.label, t.candidate, t.mismatch_frames = t.candidate, None, 0
        elif t.label is not Gesture.NONE and \
                self._confirmed(t.mismatch_since, t.mismatch_frames, now):
            t.label, t.mismatch_frames = Gesture.NONE, 0   # no longer the old gesture


# ---- the toggle ------------------------------------------------------------------------
class GesturePhase(Enum):
    ARMED = "armed"                 # waiting for the gesture pair
    HOLDING = "holding"             # pair being held; toggles after hold_seconds
    WAIT_RELEASE = "wait_release"   # just toggled; waiting for the hands to go down


class GestureToggle:
    """Fires once when a given pair of gestures (e.g. an open palm and a fist, one
    per hand) is held together for ``hold_seconds``.

    - Both gestures must be visible at the same time (one per hand).
    - The hold is only shown/reported (phase HOLDING) once the pair has been seen
      for ``hold_confirm`` s; the 3 s still count from when it first appeared.
    - Short drop-outs (<= ``dropout_grace``) don't restart the hold.
    - After firing, nothing more happens until the pair has been gone for
      ``release_seconds`` ("hands down"); holding longer never toggles twice.

    A caller that uses several ``GestureToggle``\\ s for different purposes (e.g.
    stop / resume / toggle-drawing) should ``reset()`` the ones that aren't
    currently meaningful (e.g. the stop gesture while already stopped), so an old,
    irrelevant hold can never carry over.
    """

    def __init__(self, pair: Tuple[Gesture, Gesture], settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        self.pair = pair
        self.phase = GesturePhase.ARMED
        self.hold_started = None   # time the pair first appeared for the current hold
        self.last_combo = None     # last time the pair was seen

    def reset(self):
        self.phase, self.hold_started, self.last_combo = GesturePhase.ARMED, None, None

    def hold_elapsed(self, now):
        """Seconds the current hold has lasted (0 unless HOLDING)."""
        return now - self.hold_started if self.phase is GesturePhase.HOLDING else 0.0

    def update(self, gestures, now):
        """gestures: list of Gesture seen this frame; now: seconds. Returns True on toggle."""
        s = self.settings
        combo = self.pair[0] in gestures and self.pair[1] in gestures
        gone_for = math.inf if self.last_combo is None else now - self.last_combo
        if combo:
            gone_for, self.last_combo = 0.0, now

        if self.phase is GesturePhase.WAIT_RELEASE:
            if gone_for >= s.release_seconds - TIME_EPS:
                self.phase, self.hold_started = GesturePhase.ARMED, None
            else:
                return False

        # ARMED (possibly with an unconfirmed hold in progress) or HOLDING
        if gone_for > s.dropout_grace + TIME_EPS:
            self.phase, self.hold_started = GesturePhase.ARMED, None   # let go too early
            return False
        if self.hold_started is None:
            if not combo:
                return False
            self.hold_started = now
        held = now - self.hold_started
        if combo and self.phase is GesturePhase.ARMED and held >= s.hold_confirm - TIME_EPS:
            self.phase = GesturePhase.HOLDING
        if combo and held >= s.hold_seconds - TIME_EPS:
            self.phase, self.hold_started = GesturePhase.WAIT_RELEASE, None
            return True
        return False
