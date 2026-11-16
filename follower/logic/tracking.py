"""Follow state machine: which person we follow, which hands are theirs, and
the three gestures - stop, resume (locking onto the gesturer) and show / hide
the pose diagram.

Pure Python - the vision layer feeds it landmarks each frame and renders the
``FrameResult`` it returns.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .config import DEFAULT_SETTINGS, Settings
from .gestures import Gesture, GesturePhase, GestureToggle, HandTracker, classify_hand, dist, \
    gesture_pair, hand_size

# MediaPipe pose landmark indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16

# How far the person who made the resume gesture may extend to the left / right
# of their two gesturing hands, in hand sizes (wrist -> middle knuckle, ~1/4 shoulder width).
# Vertically their region is the full frame height: people stand side by side,
# and the head (which the pose detector needs) may be far above low-held hands.
GESTURER_SIDE_MARGIN = 4.0


class Status(Enum):
    FOLLOWING = "following"     # following and the person is in view
    SEARCHING = "searching"     # following, but no person detected right now
    STOPPED = "stopped"         # stopped; waiting for the resume gesture


@dataclass(frozen=True)
class Region:
    """Axis-aligned box in normalised image coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x, y) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class ResumeTarget:
    """The person who made the resume gesture: their hands and body region."""
    hands: Tuple[Tuple[float, float], ...]   # wrist positions of the two gesturing hands
    region: Region                           # where their head + body should be


@dataclass(frozen=True)
class GestureOption:
    """A two-handed gesture the user can currently perform, and its progress.

    ``action`` is ``"stop"``, ``"resume"`` or ``"toggle_drawing"``.
    """
    action: str
    pair: Tuple[Gesture, Gesture]
    phase: GesturePhase
    hold_elapsed: float


@dataclass
class FrameResult:
    following: bool
    status: Status
    show_skeleton: bool = True                 # whether the pose diagram is drawn on the body
    toggled: bool = False                       # following flipped (stopped or resumed) now
    toggled_stop: bool = False
    toggled_resume: bool = False
    toggled_draw: bool = False                  # show_skeleton flipped now
    # (index into the frame's hand list, gesture) for hands that were counted
    hand_gestures: List[Tuple[int, Gesture]] = field(default_factory=list)
    flash_text: Optional[str] = None            # transient banner, if active
    # the gesture(s) the user can currently perform (stop + toggle-drawing while
    # following; resume while stopped), with their hold progress
    options: List[GestureOption] = field(default_factory=list)
    # on the frame following resumes: who made the gesture (to lock onto them)
    resume_target: Optional[ResumeTarget] = None

    @property
    def gestures(self) -> List[Gesture]:
        return [g for _, g in self.hand_gestures]


def shoulder_width(pose_lm, settings: Settings = DEFAULT_SETTINGS, aspect=1.0):
    """Shoulder-to-shoulder distance, or None if the shoulders aren't visible."""
    l, r = pose_lm[LEFT_SHOULDER], pose_lm[RIGHT_SHOULDER]
    if min(l.visibility, r.visibility) < settings.min_visibility:
        return None
    return dist(l, r, aspect)


def hands_of_person(hands: Sequence, pose_lm, shoulder_w,
                    settings: Settings = DEFAULT_SETTINGS, aspect=1.0, labels=None
                    ) -> List[Tuple[int, Gesture]]:
    """The hands that belong to the tracked person, with their gestures.

    ``hands`` is a list of 21-landmark hands. With a person (``pose_lm``), a
    hand counts only if its wrist is within ``hand_near_body`` body-scale units
    of one of the person's pose wrists. The body scale is the shoulder width,
    or - if the shoulders aren't visible - an estimate from the hand's own size.
    With no person (``pose_lm is None``) every visible hand counts.

    ``labels`` - (index, gesture) pairs from a HandTracker (smoothed, duplicates
    removed). Without them each hand is classified on its own.
    Returns ``(hand_index, gesture)`` pairs for the hands that count.
    """
    if labels is None:
        labels = [(i, classify_hand(lm, aspect)) for i, lm in enumerate(hands)]
    out = []
    for i, gesture in labels:
        lm = hands[i]
        if pose_lm is not None:
            scale = shoulder_w or settings.shoulders_per_hand * hand_size(lm, aspect)
            wrists = [pose_lm[LEFT_WRIST], pose_lm[RIGHT_WRIST]]
            if min(dist(lm[0], w, aspect) for w in wrists) > settings.hand_near_body * scale:
                continue
        out.append((i, gesture))
    return out


def gesturer_target(hands: Sequence, hand_gestures, aspect=1.0,
                    pair: Tuple[Gesture, Gesture] = (Gesture.PALM, Gesture.SCISSORS)
                    ) -> Optional[ResumeTarget]:
    """Locate the person who is showing the resume gesture ``pair``.

    Picks the closest pair of hands showing the two gestures among
    ``hand_gestures`` ((index, Gesture) into ``hands``) and returns the region
    where that person should be: a full-height band around the two hands.
    """
    g1, g2 = pair
    first = [hands[i] for i, g in hand_gestures if g is g1]
    second = [hands[i] for i, g in hand_gestures if g is g2]
    if not first or not second:
        return None
    a, b = min(((x, y) for x in first for y in second),
               key=lambda ab: dist(ab[0][0], ab[1][0], aspect))
    hs = (hand_size(a, aspect) + hand_size(b, aspect)) / 2 or 0.05
    xs = (a[0].x, b[0].x)
    side = GESTURER_SIDE_MARGIN * hs / aspect          # height units -> normalised x
    region = Region(max(0.0, min(xs) - side), 0.0, min(1.0, max(xs) + side), 1.0)
    return ResumeTarget(hands=((a[0].x, a[0].y), (b[0].x, b[0].y)), region=region)


def pose_matches_target(pose_lm, target: ResumeTarget) -> bool:
    """Is this pose the person who made the gesture? (shoulder midpoint in their region)"""
    l, r = pose_lm[LEFT_SHOULDER], pose_lm[RIGHT_SHOULDER]
    return target.region.contains((l.x + r.x) / 2, (l.y + r.y) / 2)


class PoseFollower:
    """Tracks whether we are following, whether the pose diagram is drawn, and
    reacts to three two-handed gestures (one hand shows each gesture):

    - ``settings.stop_gesture`` (default open palm + fist): while following, stop.
    - ``settings.resume_gesture`` (default open palm + scissors): while stopped,
      resume - locking onto the person who made the gesture, not whoever is most
      prominent.
    - ``settings.draw_toggle_gesture`` (default fist + scissors): while following,
      show or hide the pose diagram on the body. Tracking itself keeps running.

    Per frame the caller runs pose + hand detection and calls
    ``update(pose_lm, hands, now, aspect)``. When the result says ``toggled``
    (stopped or resumed), the caller should reset the pose tracker so the old
    person is dropped, and on a resume lock onto ``result.resume_target``.

    While stopped, pose landmarks are ignored: any visible hand may resume, and
    the stop / toggle-drawing gestures are not listened for (only resume is).
    """

    def __init__(self, settings: Settings = DEFAULT_SETTINGS):
        self.settings = settings
        self.following = True
        self.show_skeleton = True
        self.hand_tracker = HandTracker(settings)
        self.stop_toggle = GestureToggle(gesture_pair(settings, "stop_gesture"), settings)
        self.resume_toggle = GestureToggle(gesture_pair(settings, "resume_gesture"), settings)
        self.draw_toggle = GestureToggle(gesture_pair(settings, "draw_toggle_gesture"), settings)
        self._flash_until = 0.0
        self._flash_text = ""

    def update(self, pose_lm, hands: Sequence, now: float, aspect: float = 1.0) -> FrameResult:
        """Advance one frame.

        pose_lm: the 33 pose landmarks of the detected person, or None.
        hands:   list of 21-landmark hands visible in the frame.
        now:     timestamp in seconds (session time for live, video time for files).
        aspect:  frame width / height (landmarks are normalised per axis).
        """
        owner_pose = pose_lm if self.following else None  # while stopped, any hand may resume
        sw = shoulder_width(owner_pose, self.settings, aspect) if owner_pose else None
        tracked = self.hand_tracker.update(hands, now, aspect)   # smoothed, de-duplicated
        hand_gestures = hands_of_person(hands, owner_pose, sw, self.settings, aspect,
                                        [(i, shown) for i, shown, _ in tracked])
        owned = {i for i, _ in hand_gestures}
        confirmed = [(i, c) for i, _, c in tracked if i in owned]
        confirmed_gestures = [g for _, g in confirmed]

        target = None
        if self.following:
            toggled_stop = self.stop_toggle.update(confirmed_gestures, now)
            toggled_draw = self.draw_toggle.update(confirmed_gestures, now)
            self.resume_toggle.reset()           # not listened for while following
            toggled_resume = False
        else:
            toggled_resume = self.resume_toggle.update(confirmed_gestures, now)
            self.stop_toggle.reset()
            self.draw_toggle.reset()             # nothing to draw while stopped
            toggled_stop = toggled_draw = False

        if toggled_stop:
            self.following = False
        if toggled_resume:
            self.following = True
            target = gesturer_target(hands, confirmed, aspect, self.resume_toggle.pair)
        if toggled_draw:
            self.show_skeleton = not self.show_skeleton

        for fired, text in ((toggled_stop, "FOLLOWING STOPPED"),
                            (toggled_resume, "FOLLOWING RESUMED"),
                            (toggled_draw, "POSE DRAWING " +
                             ("ON" if self.show_skeleton else "OFF"))):
            if fired:
                self._flash_text, self._flash_until = text, now + self.settings.flash_seconds

        status = Status.STOPPED if not self.following else \
            (Status.FOLLOWING if owner_pose else Status.SEARCHING)
        options = [GestureOption("stop", self.stop_toggle.pair, self.stop_toggle.phase,
                                 self.stop_toggle.hold_elapsed(now)),
                  GestureOption("toggle_drawing", self.draw_toggle.pair, self.draw_toggle.phase,
                                self.draw_toggle.hold_elapsed(now))] if self.following else \
            [GestureOption("resume", self.resume_toggle.pair, self.resume_toggle.phase,
                          self.resume_toggle.hold_elapsed(now))]
        return FrameResult(
            following=self.following,
            status=status,
            show_skeleton=self.show_skeleton,
            toggled=toggled_stop or toggled_resume,
            toggled_stop=toggled_stop, toggled_resume=toggled_resume, toggled_draw=toggled_draw,
            hand_gestures=hand_gestures,
            flash_text=self._flash_text if now < self._flash_until else None,
            options=options,
            resume_target=target,
        )
