"""follower.logic.gestures - classifier, per-hand smoothing, and the hold-to-toggle
detector (pure logic)."""
import pytest

from follower.logic import Gesture, GesturePhase, GestureToggle, HandTracker, Settings, \
    classify_hand, gesture_pair, pair_name
from follower.logic.gestures import finger_bends, thumb_tuck
from tests.landmarks import LM, make_hand

F, P, S, N = Gesture.FIST, Gesture.PALM, Gesture.SCISSORS, Gesture.NONE
FPS = 30.0
ALL = [F, P, S, N]


# ---- classify_hand ---------------------------------------------------------
@pytest.mark.parametrize("gesture", ALL)
@pytest.mark.parametrize("angle", [0, 45, 90, 135, 180, -60])
@pytest.mark.parametrize("size", [0.05, 0.3])
def test_classify_hand_is_rotation_and_scale_invariant(gesture, angle, size):
    assert classify_hand(make_hand(gesture, size=size, angle_deg=angle)) is gesture


@pytest.mark.parametrize("aspect", [4 / 3, 16 / 9])
@pytest.mark.parametrize("gesture", ALL)
@pytest.mark.parametrize("angle", range(0, 360, 15))
def test_classify_hand_on_non_square_frames(gesture, angle, aspect):
    lm = make_hand(gesture, size=0.2, angle_deg=angle, aspect=aspect)
    assert classify_hand(lm, aspect=aspect) is gesture


def test_fist_needs_the_thumb_tucked_in():
    assert classify_hand(make_hand(F, thumb="in")) is F
    assert classify_hand(make_hand(F, thumb="out")) is N       # e.g. thumbs-up


def test_palm_needs_the_thumb_out():
    assert classify_hand(make_hand(P, thumb="out")) is P
    assert classify_hand(make_hand(P, thumb="in")) is N        # "four"


def test_scissors_is_two_fingers_with_thumb_tucked():
    assert classify_hand(make_hand(S)) is S
    assert classify_hand(make_hand(S, thumb="out")) is N       # V-sign with thumb out
    assert classify_hand(make_hand(N)) is N                    # only the index out


def test_fingers_pointing_at_the_camera_still_read_correctly():
    """3-D: a straight finger pointing toward the camera is short in 2-D (the old
    classifier called such hands a fist) but its bend angle is still ~0."""
    palm = make_hand(P, size=0.2)
    for tip in (8, 12, 16, 20):                  # push the finger ends toward the camera
        for j in (tip - 2, tip - 1, tip):
            wx, wy = palm[tip - 3].x, palm[tip - 3].y
            k = (j - (tip - 3)) / 3
            palm[j] = LM(wx + (palm[j].x - wx) * 0.3, wy + (palm[j].y - wy) * 0.3, z=-0.09 * k * 3)
    assert max(finger_bends(palm)) < 20 and classify_hand(palm) is P


def test_degenerate_hand_does_not_crash():
    assert classify_hand(make_hand(P, size=0.0)) in ALL
    assert thumb_tuck(make_hand(P, size=0.0)) >= 0


def test_pair_names():
    s = Settings()
    assert gesture_pair(s, "stop_gesture") == (P, F) and pair_name((P, F)) == "open palm + fist"
    assert gesture_pair(s, "resume_gesture") == (P, S)
    assert pair_name((P, S)) == "open palm + scissors"
    assert gesture_pair(s, "draw_toggle_gesture") == (F, S)
    assert pair_name((F, S)) == "fist + scissors"


# ---- HandTracker: smoothing + de-duplication ----------------------------------
def track(frames, settings=Settings(), fps=FPS):
    """frames: list of lists of hands. Returns per frame [(shown, confirmed), ...]."""
    t, out = HandTracker(settings), []
    for i, hands in enumerate(frames):
        out.append([(shown, conf) for _, shown, conf in t.update(hands, i / fps)])
    return out


def test_label_needs_two_frames_and_a_tenth_of_a_second():
    fist, palm = make_hand(F, 0.3, 0.5), make_hand(P, 0.3, 0.5)
    at30 = track([[fist]] * 6)
    assert [r[0][0] for r in at30] == [N, N, N, F, F, F]          # 0.1 s at 30 fps
    at9 = track([[fist]] * 3, fps=9)
    assert [r[0][0] for r in at9] == [N, F, F]                    # 2 frames at 9 fps
    assert track([[palm]] + [[fist]] * 2, fps=9)[-1][0][0] is F


def test_fist_flickering_to_none_stays_a_fist():
    fist, loose = make_hand(F, 0.3, 0.5), make_hand(N, 0.3, 0.5)
    frames = [[fist]] * 3 + [[loose], [fist]] * 6                  # alternates every frame
    shown = [r[0][0] for r in track(frames, fps=9)]
    assert shown[2:] == [F] * (len(frames) - 2)


def test_single_frame_palm_is_never_shown_or_confirmed():
    fist, palm = make_hand(F, 0.3, 0.5), make_hand(P, 0.3, 0.5)
    res = track([[fist]] * 3 + [[palm]] + [[fist]] * 3, fps=9)
    assert all(r[0][0] is not P and r[0][1] is not P for r in res)


def test_stale_label_is_not_confirmed():
    """While a palm closes, the shown label lags but doesn't count as a palm."""
    palm, fist = make_hand(P, 0.3, 0.5), make_hand(F, 0.3, 0.5)
    res = track([[palm]] * 3 + [[fist]], fps=9)
    assert res[-1][0] == (P, N)                                   # shown PALM, confirmed NONE


def test_a_changing_hand_drops_to_none():
    palm, fist, loose = make_hand(P, 0.3, 0.5), make_hand(F, 0.3, 0.5), make_hand(N, 0.3, 0.5)
    res = track([[palm]] * 3 + [[fist], [loose]], fps=9)
    assert res[-1][0][0] is N


def test_duplicate_detections_of_one_hand_are_dropped():
    a = make_hand(F, 0.30, 0.5, size=0.12)
    dup = make_hand(F, 0.305, 0.505, size=0.10)
    other = make_hand(P, 0.70, 0.5, size=0.12)
    tr = HandTracker()
    assert [i for i, _, _ in tr.update([a, dup, other], 0.0)] == [0, 2]


def test_hands_keep_their_labels_when_they_swap_order_or_briefly_vanish():
    fist, palm = make_hand(F, 0.3, 0.5), make_hand(P, 0.7, 0.5)
    tr = HandTracker()
    for i in range(6):
        tr.update([fist, palm], i / FPS)
    assert {i: g for i, g, _ in tr.update([palm, fist], 6 / FPS)} == {0: P, 1: F}
    tr.update([palm], 7 / FPS)                                     # fist missing for a frame
    assert {i: g for i, g, _ in tr.update([fist, palm], 8 / FPS)} == {0: F, 1: P}


# ---- GestureToggle ---------------------------------------------------------
PAIR = (S, P)   # the pair these tests hold; GestureToggle itself is pair-agnostic


def seconds(s, gestures=()):
    return [list(gestures)] * round(s * FPS)


def feed(toggle, frames, start=0.0):
    """Feed per-frame gesture lists at 30 fps; return the times it toggled."""
    fired = []
    for i, gestures in enumerate(frames):
        t = start + i / FPS
        if toggle.update(gestures, t):
            fired.append(round(t, 6))
    return fired


def test_holding_the_pair_for_3_seconds_toggles_at_3_seconds():
    assert feed(GestureToggle(PAIR), seconds(1) + seconds(5, [S, P])) == [4.0]


def test_any_pair_works_the_same_way():
    """GestureToggle doesn't care which two gestures it's told to watch for."""
    assert feed(GestureToggle((F, P)), seconds(5, [F, P])) == [3.0]
    assert feed(GestureToggle((F, S)), seconds(5, [F, S])) == [3.0]


def test_holding_for_less_than_3_seconds_does_nothing():
    assert feed(GestureToggle(PAIR), seconds(2.9, [S, P]) + seconds(2)) == []


@pytest.mark.parametrize("hold_seconds", [3.1, 3.5, 6.0, 20.0])
def test_keeping_the_hands_up_never_toggles_twice(hold_seconds):
    assert len(feed(GestureToggle(PAIR), seconds(hold_seconds, [S, P]) + seconds(2))) == 1


def test_hands_must_go_down_before_the_next_gesture_counts():
    frames = (seconds(3.5, [S, P]) + seconds(0.4) + seconds(4, [S, P]) + seconds(0.6) +
              seconds(3.2, [S, P]))
    fired = feed(GestureToggle(PAIR), frames)
    assert len(fired) == 2
    assert fired[1] == pytest.approx(3.5 + 0.4 + 4 + 0.6 + 3.0)


def test_reset_clears_any_in_progress_hold():
    toggle = GestureToggle(PAIR)
    toggle.update([S, P], 1.0)
    toggle.update([S, P], 1.5)
    assert toggle.phase is GesturePhase.HOLDING
    toggle.reset()
    assert toggle.phase is GesturePhase.ARMED and toggle.hold_elapsed(1.5) == 0.0
    assert feed(toggle, seconds(3.2, [S, P]), start=1.5) == [pytest.approx(1.5 + 3.0)]


def test_hold_is_only_reported_after_hold_confirm():
    toggle = GestureToggle(PAIR)
    toggle.update([S, P], 10.0)
    assert toggle.phase is GesturePhase.ARMED and toggle.hold_elapsed(10.0) == 0.0
    toggle.update([S, P], 10.3)
    assert toggle.phase is GesturePhase.ARMED
    toggle.update([S, P], 10.5)
    assert toggle.phase is GesturePhase.HOLDING and toggle.hold_elapsed(10.5) == 0.5
    assert toggle.update([S, P], 13.0)                    # still 3 s from the first frame
    assert toggle.phase is GesturePhase.WAIT_RELEASE
    toggle.update([], 13.6)
    assert toggle.phase is GesturePhase.ARMED


def test_momentary_pair_is_never_reported():
    toggle, phases = GestureToggle(PAIR), []
    for i, g in enumerate(seconds(0.3, [S, P]) + seconds(1)):
        toggle.update(g, i / FPS)
        phases.append(toggle.phase)
    assert GesturePhase.HOLDING not in phases


def test_short_dropouts_do_not_restart_the_hold():
    assert feed(GestureToggle(PAIR), seconds(1.5, [S, P]) + seconds(0.2) + seconds(2, [S, P])) \
        == [3.0]


def test_long_dropouts_restart_the_hold():
    assert feed(GestureToggle(PAIR), seconds(1.5, [S, P]) + seconds(0.5) + seconds(2.9, [S, P])) \
        == []


def test_both_gestures_must_be_shown_together():
    frames = seconds(2, [S]) + seconds(2, [P]) + seconds(2, [S]) + seconds(2, [P])
    assert feed(GestureToggle(PAIR), frames) == []


def test_other_combinations_never_toggle():
    for combo in ([F, P], [F, S], [P, P], [S, S], [N, N], [F]):
        assert feed(GestureToggle(PAIR), seconds(10, combo)) == [], combo


def test_extra_hands_do_not_matter():
    assert feed(GestureToggle(PAIR), seconds(3.5, [S, P, N])) == [3.0]


def test_settings_are_respected():
    s = Settings(hold_seconds=1.0, release_seconds=0.2, hold_confirm=0.0)
    assert feed(GestureToggle(PAIR, s), seconds(1.5, [S, P]) + seconds(0.3) + seconds(1.5, [S, P])) \
        == [1.0, pytest.approx(2.8)]


def test_frame_rate_does_not_change_timing():
    for fps in (8, 15, 60):
        toggle, fired = GestureToggle(PAIR), []
        for i in range(int(5 * fps)):
            if toggle.update([S, P], i / fps):
                fired.append(i / fps)
        assert fired == [pytest.approx(3.0)]
