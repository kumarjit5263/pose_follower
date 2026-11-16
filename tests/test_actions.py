"""follower.logic.actions - the action timeline and the .actions.txt report."""
from follower.logic import ActionKind as K, ActionLog, FrameResult, Gesture, GestureOption, \
    GesturePhase, Settings, Status, format_report

DT = 1 / 30
STOP_PAIR = (Gesture.PALM, Gesture.FIST)
RESUME_PAIR = (Gesture.PALM, Gesture.SCISSORS)
DRAW_PAIR = (Gesture.FIST, Gesture.SCISSORS)


def opt(action, phase, hold_elapsed=0.0, pair=None):
    pair = pair or {"stop": STOP_PAIR, "resume": RESUME_PAIR,
                    "toggle_drawing": DRAW_PAIR}[action]
    return GestureOption(action, pair, phase, hold_elapsed)


def feed(log, spec, t0=0.0):
    """spec: list of (seconds, FrameResult kwargs)."""
    t = t0
    for seconds, kw in spec:
        for _ in range(round(seconds / DT)):
            log.observe(FrameResult(**kw), t)
            t += DT
    return t


FOLLOW = dict(following=True, status=Status.FOLLOWING)
SEARCH = dict(following=True, status=Status.SEARCHING)


def test_person_found_and_lost_are_debounced_and_backdated():
    log = ActionLog(Settings(event_debounce=0.3))
    t = feed(log, [(0.5, SEARCH), (1.0, FOLLOW), (0.1, SEARCH), (1.0, FOLLOW), (1.0, SEARCH)])
    log.finish(t)
    assert log.kinds() == [K.START, K.PERSON_FOUND, K.PERSON_LOST, K.END]   # 0.1 s blip ignored
    found, lost = log.actions[1], log.actions[2]
    assert abs(found.t - 0.5) < 1e-6            # stamped when the person appeared
    assert abs(lost.t - 2.6) < 1e-6


def test_pending_change_is_flushed_at_the_end():
    log = ActionLog(Settings(event_debounce=0.3))
    t = feed(log, [(0.5, SEARCH), (0.4, FOLLOW)])
    log.finish(t)
    assert K.PERSON_FOUND in log.kinds()


def test_stop_gesture_actions():
    """GESTURE_STARTED / TOO_SHORT come from the phase transitions of the "stop"
    option; STOPPED comes from ``toggled_stop`` - needed because by the time this
    FrameResult is observed, ``following`` (and the option list) has already
    flipped, so "stop" is no longer listed."""
    log = ActionLog()
    armed = dict(FOLLOW, options=[opt("stop", GesturePhase.ARMED)])
    holding = dict(FOLLOW, options=[opt("stop", GesturePhase.HOLDING, 1.0)])
    t = feed(log, [(1, armed), (1.2, holding), (1, armed)])    # too short: released after 1.2 s
    t = feed(log, [(0.1, holding)], t)                          # starts again
    log.observe(FrameResult(following=False, status=Status.STOPPED, toggled=True,
                            toggled_stop=True, options=[opt("resume", GesturePhase.ARMED)]), t)
    t += DT
    t = feed(log, [(1, dict(following=False, status=Status.STOPPED,
                            options=[opt("resume", GesturePhase.ARMED)]))], t)
    log.finish(t)
    assert log.kinds() == [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.GESTURE_TOO_SHORT,
                           K.GESTURE_STARTED, K.STOPPED, K.END]            # no HANDS_LOWERED
    assert "stop following" in log.actions[2].message
    assert log.summary()["stops"] == 1 and log.summary()["resumes"] == 0


def test_switching_to_resume_does_not_fabricate_hands_lowered():
    """"stop" disappearing from the options (because we stopped) must not look like
    its hold was released - that's not something the user did."""
    log = ActionLog()
    t = feed(log, [(1, dict(FOLLOW, options=[opt("stop", GesturePhase.HOLDING, 1.0)]))])
    log.observe(FrameResult(following=False, status=Status.STOPPED, toggled=True,
                            toggled_stop=True, options=[opt("resume", GesturePhase.ARMED)]), t)
    log.finish(t + DT)
    assert K.HANDS_LOWERED not in log.kinds()


def test_resume_gesture_actions_and_locking_message():
    log = ActionLog()
    t = feed(log, [(1, dict(following=False, status=Status.STOPPED,
                            options=[opt("resume", GesturePhase.HOLDING, 2.9)]))])
    log.observe(FrameResult(following=True, status=Status.SEARCHING, toggled=True,
                            toggled_resume=True, options=[opt("stop", GesturePhase.ARMED),
                                                          opt("toggle_drawing",
                                                              GesturePhase.ARMED)]), t)
    log.finish(t + DT)
    assert log.kinds() == [K.START, K.GESTURE_STARTED, K.RESUMED, K.END]
    assert "locking onto the person" in log.actions[-2].message
    assert log.summary()["resumes"] == 1


def test_drawing_toggle_actions_do_include_hands_lowered():
    """Unlike stop/resume, "toggle_drawing" stays listenable after it fires (following
    doesn't change), so its WAIT_RELEASE is visible and HANDS_LOWERED is logged."""
    log = ActionLog()
    holding = dict(FOLLOW, options=[opt("toggle_drawing", GesturePhase.HOLDING, 2.9)])
    t = feed(log, [(1, FOLLOW), (0.1, holding)])
    log.observe(FrameResult(following=True, status=Status.FOLLOWING, toggled_draw=True,
                            show_skeleton=False,
                            options=[opt("toggle_drawing", GesturePhase.WAIT_RELEASE)]), t)
    t += DT
    t = feed(log, [(0.6, dict(FOLLOW, options=[opt("toggle_drawing", GesturePhase.WAIT_RELEASE)])),
                   (0.1, dict(FOLLOW, options=[opt("toggle_drawing", GesturePhase.ARMED)]))], t)
    log.finish(t)
    assert log.kinds() == [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.DRAWING_HIDDEN,
                           K.HANDS_LOWERED, K.END]
    assert "hide the pose diagram" in log.actions[2].message
    assert log.summary()["drawing_toggles"] == 1


def test_drawing_shown_event_when_turning_it_back_on():
    log = ActionLog()
    log.observe(FrameResult(following=True, status=Status.FOLLOWING, toggled_draw=True,
                            show_skeleton=True, options=[]), 0.0)
    log.finish(0.0)
    assert log.kinds() == [K.START, K.DRAWING_SHOWN, K.END]


def test_report_lists_actions_and_summary():
    log = ActionLog()
    t = feed(log, [(1, SEARCH), (2, FOLLOW)])
    log.finish(t)
    text = format_report(log, {"source": "clip.mp4", "input": "video file"})
    assert "clip.mp4" in text and "video file" in text
    assert "00:01.000  PERSON_FOUND" in text
    assert "person followed     67% of frames" in text
    assert text.index("START") < text.index("PERSON_FOUND") < text.index("END")


def test_empty_session():
    log = ActionLog()
    log.finish()
    assert log.kinds() == [K.END] and log.summary()["frames"] == 0
    assert "0 frames" in format_report(log, {})
