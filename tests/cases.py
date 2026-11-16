"""Synthetic test scenarios (test-only code).

Each case is a frame-by-frame sequence of MediaPipe-shaped landmarks with the
expected outcome. They are used by tests/test_cases.py and written out - with
their action logs and rendered videos - by tests/generate_cases.py.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from follower.logic import DEFAULT_SETTINGS, ActionKind, ActionLog, Gesture, \
    LandmarkFileWriter, PoseFollower, Settings
from tests.landmarks import hands_at_wrists, make_hand, make_pose

FPS = 30.0
WIDTH, HEIGHT = 640, 480
ASPECT = WIDTH / HEIGHT
F, P, S, N = Gesture.FIST, Gesture.PALM, Gesture.SCISSORS, Gesture.NONE
K = ActionKind

Frame = Tuple[Optional[list], list]   # (pose landmarks or None, list of hands)


def hold(seconds: float, pose=None, hands=()) -> List[Frame]:
    """The same observation for ``seconds`` worth of frames."""
    return [(pose, list(hands))] * round(seconds * FPS)


# People used in the scenarios (4:3 frame)
def person(x, arms_up=False, **kw):
    return make_pose(x=x, y=0.35, shoulder_w=0.2, arms_up=arms_up, aspect=ASPECT, **kw)


A, A_UP = person(0.35), person(0.35, arms_up=True)
B, B_UP = person(0.70), person(0.70, arms_up=True)
STOP_A = hands_at_wrists(A_UP, F, P, aspect=ASPECT)       # open palm + fist
RESUME_B = hands_at_wrists(B_UP, P, S, aspect=ASPECT)     # open palm + scissors
DRAW_A = hands_at_wrists(A_UP, F, S, aspect=ASPECT)       # fist + scissors
PALM_ONLY, FIST_ONLY = STOP_A[1:], STOP_A[:1]
STOP_A_LOOSE_PALM = hands_at_wrists(A_UP, F, N, aspect=ASPECT)  # palm mis-read as NONE
BYSTANDER_STOP = [make_hand(F, 0.85, 0.35, size=0.12, aspect=ASPECT),
                  make_hand(P, 0.95, 0.35, size=0.12, aspect=ASPECT)]


@dataclass
class Case:
    name: str
    description: str
    build: Callable[[], List[Frame]]
    expected_toggles: List[bool]               # following state after each stop/resume
    expected_actions: List[ActionKind] = field(default_factory=list)

    def frames(self) -> List[Frame]:
        return self.build()


CASES = [
    Case("stop_and_resume",
         "Person A is followed, holds open palm + fist for 3.5 s -> stop (at 3 s). A lowers "
         "hands and leaves. Person B holds open palm + scissors -> resume, then B is followed.",
         lambda: hold(1, A) + hold(3.5, A_UP, STOP_A) + hold(1, A) + hold(0.5) +
         hold(3.5, B_UP, RESUME_B) + hold(2, B),
         [False, True],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.STOPPED, K.GESTURE_STARTED, K.RESUMED,
          K.PERSON_FOUND, K.END]),

    Case("gesture_too_short",
         "Open palm + fist held for 2 s, then 1.5 s - both shorter than 3 s, nothing toggles.",
         lambda: hold(1, A) + hold(2, A_UP, STOP_A) + hold(1, A) + hold(1.5, A_UP, STOP_A) +
         hold(1, A),
         [],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.GESTURE_TOO_SHORT,
          K.GESTURE_STARTED, K.GESTURE_TOO_SHORT, K.END]),

    Case("brief_dropout_tolerated",
         "During the hold the hand detector loses both hands for 0.2 s; the hold continues "
         "and following stops 3 s after it began.",
         lambda: hold(1, A) + hold(1.5, A_UP, STOP_A) + hold(0.2, A_UP) +
         hold(2, A_UP, STOP_A) + hold(1, A),
         [False],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.STOPPED, K.END]),

    Case("long_dropout_restarts_hold",
         "Hands lost for 0.6 s in the middle of the hold: the hold restarts, and neither "
         "part is long enough.",
         lambda: hold(1, A) + hold(1.5, A_UP, STOP_A) + hold(0.6, A_UP) +
         hold(2, A_UP, STOP_A) + hold(1, A),
         [],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.GESTURE_TOO_SHORT,
          K.GESTURE_STARTED, K.GESTURE_TOO_SHORT, K.END]),

    Case("palm_then_fist_separately",
         "An open palm for 2 s, then a fist for 2 s (never both at once) - not the gesture.",
         lambda: hold(1, A) + hold(2, A_UP, PALM_ONLY) + hold(2, A_UP, FIST_ONLY) + hold(1, A),
         [],
         [K.START, K.PERSON_FOUND, K.END]),

    Case("bystander_ignored",
         "A is followed; a bystander on the far right holds open palm + fist for 4 s. Only "
         "the followed person's hands count, so nothing changes.",
         lambda: hold(1, A) + hold(4, A, BYSTANDER_STOP) + hold(1, A),
         [],
         [K.START, K.PERSON_FOUND, K.END]),

    Case("bystander_ignored_shoulders_hidden",
         "Same as bystander_ignored, but the followed person's shoulders aren't visible. "
         "(Without that fallback every hand would count, and the bystander would stop it.)",
         lambda: (hold(1, person(0.35, shoulder_visibility=0.2)) +
                  hold(4, person(0.35, shoulder_visibility=0.2), BYSTANDER_STOP) +
                  hold(1, person(0.35, shoulder_visibility=0.2))),
         [],
         [K.START, K.PERSON_FOUND, K.END]),

    Case("resume_needs_scissors_not_the_stop_gesture",
         "After stopping, showing open palm + fist again (the stop gesture) does not resume; "
         "open palm + scissors does.",
         lambda: (hold(1, A) + hold(3.2, A_UP, STOP_A) + hold(1, A) +
                  hold(3.2, None, hands_at_wrists(A_UP, F, P)) + hold(1, None) +
                  hold(3.2, None, hands_at_wrists(A_UP, P, S)) + hold(1, A)),
         [False, True],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.STOPPED, K.GESTURE_STARTED, K.RESUMED,
          K.PERSON_FOUND, K.END]),

    Case("palm_flicker_tolerated",
         "While open palm + fist is held, the palm is mis-read as NONE on every other frame "
         "(the reported flicker); the hold still completes and following stops.",
         lambda: (hold(1, A) + [f for _ in range(50) for f in (hold(1 / FPS, A_UP, STOP_A) +
                                                               hold(1 / FPS, A_UP, STOP_A_LOOSE_PALM))]
                  + hold(1, A)),
         [False],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.STOPPED, K.END]),

    Case("momentary_pair_ignored",
         "Open palm + fist appears for only 0.3 s (e.g. while the hands change shape): no "
         "hold is started or shown at all.",
         lambda: hold(1, A) + hold(0.3, A_UP, STOP_A) + hold(2, A),
         [],
         [K.START, K.PERSON_FOUND, K.END]),

    Case("draw_toggle_shows_and_hides_the_skeleton",
         "While following, fist + scissors held 3.5 s hides the pose diagram; held again "
         "later, it's shown again. Following never stops.",
         lambda: hold(1, A) + hold(3.5, A_UP, DRAW_A) + hold(1, A) + hold(3.5, A_UP, DRAW_A) +
         hold(1, A),
         [],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.DRAWING_HIDDEN, K.HANDS_LOWERED,
          K.GESTURE_STARTED, K.DRAWING_SHOWN, K.HANDS_LOWERED, K.END]),

    Case("draw_toggle_does_not_flip_twice_while_held",
         "Fist + scissors held for 7 s: the pose diagram hides once at 3 s and must NOT "
         "show again while the hands stay up. Lowering the hands re-arms it.",
         lambda: hold(1, A) + hold(7, A_UP, DRAW_A) + hold(1, A) + hold(3.2, A_UP, DRAW_A) +
         hold(1, A),
         [],
         [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.DRAWING_HIDDEN, K.HANDS_LOWERED,
          K.GESTURE_STARTED, K.DRAWING_SHOWN, K.HANDS_LOWERED, K.END]),

    Case("person_leaves_and_returns",
         "A is followed, drops out of detection for a single frame (not reported), leaves "
         "for 1.5 s, then B walks in and is followed.",
         lambda: hold(2, A) + hold(1 / FPS) + hold(1, A) + hold(1.5) + hold(2, B),
         [],
         [K.START, K.PERSON_FOUND, K.PERSON_LOST, K.PERSON_FOUND, K.END]),
]

CASES_BY_NAME = {c.name: c for c in CASES}


def simulate(case: Case, settings: Settings = DEFAULT_SETTINGS):
    """Run a case through the pure logic. Returns (toggles, ActionLog)."""
    follower, log, toggles = PoseFollower(settings), ActionLog(settings), []
    t = 0.0
    for i, (pose, hands) in enumerate(case.frames()):
        t = i / FPS
        r = follower.update(pose, hands, t, aspect=ASPECT)
        log.observe(r, t)
        if r.toggled:
            toggles.append((t, r.following))
    log.finish(t)
    return toggles, log


def write_landmarks(case: Case, path) -> None:
    """Write a case as a ``.landmarks.jsonl`` recording (replayable by the app)."""
    with LandmarkFileWriter(path, source=f"synthetic:{case.name}", kind="synthetic",
                            width=WIDTH, height=HEIGHT, fps=FPS,
                            description=case.description) as w:
        for i, (pose, hands) in enumerate(case.frames()):
            w.write(i, i / FPS, pose, hands)
