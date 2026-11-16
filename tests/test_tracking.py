"""follower.logic.tracking - who is followed, the three gestures, and the pose-
diagram toggle (pure logic)."""
from follower.logic import Gesture, GesturePhase, PoseFollower, Settings, Status, \
    gesturer_target, hands_of_person, pose_matches_target, shoulder_width
from tests.landmarks import hands_at_wrists, make_hand, make_pose

F, P, S, N = Gesture.FIST, Gesture.PALM, Gesture.SCISSORS, Gesture.NONE
DT = 1 / 30
# toggle on a single frame: no hold time and no label smoothing
INSTANT = dict(hold_seconds=0.0, hold_confirm=0.0, label_confirm_frames=1,
               label_confirm_seconds=0.0, release_seconds=0.0)


def run(follower, pose, hands, seconds, t0, aspect=1.0):
    """Feed the same observation for ``seconds``; return (results, next_t)."""
    results, n = [], round(seconds / DT)
    for i in range(n):
        results.append(follower.update(pose, hands, t0 + i * DT, aspect))
    return results, t0 + n * DT


def option(result, action):
    return next(o for o in result.options if o.action == action)


# ---- helpers ---------------------------------------------------------------
def test_shoulder_width():
    assert abs(shoulder_width(make_pose(shoulder_w=0.25)) - 0.25) < 1e-9
    assert shoulder_width(make_pose(visibility=0.2)) is None
    wide = make_pose(shoulder_w=0.2, aspect=16 / 9)       # 0.2 of the height, in pixels
    assert abs(shoulder_width(wide, aspect=16 / 9) - 0.2) < 1e-9


def test_hands_near_the_person_count_and_far_hands_are_ignored():
    pose = make_pose(x=0.3)
    own = hands_at_wrists(pose, F, P)
    stranger = make_hand(P, x=0.95, y=0.1)       # > 3 shoulder widths from both wrists
    out = hands_of_person([stranger] + own, pose, shoulder_width(pose))
    assert out == [(1, F), (2, P)]                # indices refer to the input list


def test_without_a_person_every_hand_counts():
    hands = [make_hand(F, 0.1, 0.1), make_hand(P, 0.9, 0.9)]
    assert hands_of_person(hands, None, None) == [(0, F), (1, P)]


def test_hidden_shoulders_fall_back_to_hand_size():
    """Regression: with the shoulders hidden the old code counted every hand,
    so a bystander could stop following."""
    pose = make_pose(x=0.3, shoulder_visibility=0.1)
    assert shoulder_width(pose) is None
    own = hands_at_wrists(pose, F, P, size=0.1)
    bystander = [make_hand(F, 0.9, 0.2, size=0.1), make_hand(P, 0.95, 0.2, size=0.1)]
    assert hands_of_person(bystander + own, pose, None) == [(2, F), (3, P)]


def test_hand_distance_uses_pixel_geometry():
    """On a 16:9 frame a hand 2.5 shoulder widths *below* the wrist is within reach
    (3 widths). Without the aspect correction it measured as 4.4 widths."""
    aspect = 16 / 9
    pose = make_pose(x=0.5, y=0.2, shoulder_w=0.1, aspect=aspect)
    wx, wy = pose[15].x, pose[15].y
    hand = make_hand(P, wx, wy + 0.25, size=0.05, aspect=aspect)
    sw = shoulder_width(pose, aspect=aspect)
    assert hands_of_person([hand], pose, sw, aspect=aspect) == [(0, P)]
    assert hands_of_person([hand], pose, shoulder_width(pose)) == []   # uncorrected


# ---- PoseFollower: following ------------------------------------------------------
def test_starts_following_and_reports_searching_until_someone_appears():
    f = PoseFollower()
    assert f.following and f.show_skeleton
    assert f.update(None, [], 0.0).status is Status.SEARCHING
    assert f.update(make_pose(), [], DT).status is Status.FOLLOWING


def test_options_depend_on_following_state():
    f = PoseFollower()
    while_following = f.update(make_pose(), [], 0.0)
    assert {o.action for o in while_following.options} == {"stop", "toggle_drawing"}
    f.following = False
    while_stopped = f.update(None, [], DT)
    assert {o.action for o in while_stopped.options} == {"resume"}


def test_full_cycle_stop_then_another_person_resumes():
    f = PoseFollower()
    a, a_up = make_pose(x=0.3), make_pose(x=0.3, arms_up=True)
    results, t = run(f, a, [], 1.0, 0.0)
    assert all(r.status is Status.FOLLOWING and not r.toggled for r in results)

    # followed person holds open palm + fist: progress is reported, stops after 3 s
    results, t = run(f, a_up, hands_at_wrists(a_up, F, P), 3.5, t)
    assert [r.toggled for r in results].count(True) == 1
    stop = next(r for r in results if r.toggled)
    # 3.0 s after the hands' gestures are confirmed (0.1 s = 3 frames of label smoothing)
    assert results.index(stop) == 93
    assert option(results[48], "stop").phase is GesturePhase.HOLDING
    assert 1.4 < option(results[48], "stop").hold_elapsed < 1.6
    assert stop.toggled_stop and not stop.toggled_resume and not stop.toggled_draw
    assert not stop.following and stop.status is Status.STOPPED
    assert stop.flash_text == "FOLLOWING STOPPED"

    # showing the STOP gesture again while stopped does nothing (only resume is listened for)
    results, t = run(f, a, hands_at_wrists(a_up, F, P), 3.5, t)
    assert all(not r.toggled for r in results) and all(r.status is Status.STOPPED for r in results)
    assert [o.action for o in results[-1].options] == ["resume"]

    # a different person elsewhere in the frame holds open palm + scissors -> resumes
    b_up = make_pose(x=0.8, arms_up=True)
    results, t = run(f, b_up, hands_at_wrists(b_up, P, S), 3.2, t)
    resume = next(r for r in results if r.toggled)
    assert resume.toggled_resume and resume.following and resume.flash_text == "FOLLOWING RESUMED"
    assert results[-1].status is Status.FOLLOWING         # B is now followed


def test_bystander_gestures_do_not_stop_following():
    f = PoseFollower()
    bystander = [make_hand(F, 0.95, 0.2), make_hand(P, 0.95, 0.3)]
    results, _ = run(f, make_pose(x=0.2), bystander, 5.0, 0.0)
    assert all(r.following and not r.toggled and r.hand_gestures == [] for r in results)


def test_pose_passed_while_stopped_is_ignored():
    f = PoseFollower(Settings(**INSTANT))
    pose = make_pose(arms_up=True)
    assert f.update(pose, hands_at_wrists(pose, F, P), 0.0).toggled_stop
    assert f.update(make_pose(), [], 3.0).status is Status.STOPPED


def test_flash_lasts_flash_seconds():
    f = PoseFollower(Settings(**INSTANT, flash_seconds=1.5))
    pose = make_pose(arms_up=True)
    assert f.update(pose, hands_at_wrists(pose, F, P), 10.0).flash_text == "FOLLOWING STOPPED"
    assert f.update(None, [], 11.4).flash_text == "FOLLOWING STOPPED"
    assert f.update(None, [], 11.5).flash_text is None


# ---- the pose-diagram toggle --------------------------------------------------------
def test_drawing_toggle_does_not_affect_following():
    f = PoseFollower(Settings(**INSTANT))
    pose = make_pose(arms_up=True)
    r = f.update(pose, hands_at_wrists(pose, F, S), 0.0)
    assert r.toggled_draw and not r.toggled and not r.show_skeleton
    assert r.following and r.status is Status.FOLLOWING
    assert r.flash_text == "POSE DRAWING OFF"
    r2 = f.update(make_pose(), [], 1.0)
    assert r2.show_skeleton is False                    # stays off until toggled again


def test_drawing_toggle_is_not_listened_for_while_stopped():
    f = PoseFollower(Settings(**INSTANT))
    f.following = False
    pose = make_pose(arms_up=True)
    r = f.update(pose, hands_at_wrists(pose, F, S), 0.0)
    assert not r.toggled_draw and f.show_skeleton is True


def test_resume_does_not_re_enable_drawing_by_itself():
    f = PoseFollower(Settings(**INSTANT))
    pose = make_pose(arms_up=True)
    f.update(pose, hands_at_wrists(pose, F, S), 0.0)     # hide the diagram
    f.update(pose, hands_at_wrists(pose, F, P), 1.0)     # stop
    r = f.update(pose, hands_at_wrists(pose, P, S), 2.0)  # resume
    assert r.toggled_resume and r.show_skeleton is False


# ---- stop / resume / toggle-drawing never interfere with each other --------------------
def test_each_gesture_only_fires_its_own_action():
    for pair, action in (((F, P), "stop"), ((F, S), "toggle_drawing")):
        f = PoseFollower(Settings(**INSTANT))
        pose = make_pose(arms_up=True)
        r = f.update(pose, hands_at_wrists(pose, *pair), 0.0)
        fired = {"stop": r.toggled_stop, "toggle_drawing": r.toggled_draw}
        assert fired.pop(action)
        assert not any(fired.values())


# ---- resuming follows the person who made the gesture ---------------------------
def test_gesturer_target_surrounds_the_gesturer_and_not_others():
    a, b_up = make_pose(x=0.25), make_pose(x=0.75, arms_up=True)
    hands = [make_hand(P, 0.2, 0.9)] + hands_at_wrists(b_up, P, S)   # a stray palm + B's combo
    target = gesturer_target(hands, [(0, P), (1, P), (2, S)], pair=(P, S))
    assert target is not None
    assert pose_matches_target(b_up, target) and not pose_matches_target(a, target)
    assert target.region.y1 == 1.0 and 0 <= target.region.x0 < target.region.x1 <= 1
    assert set(target.hands) == {(hands[1][0].x, hands[1][0].y),   # picked B's pair
                                 (hands[2][0].x, hands[2][0].y)}
    head = b_up[0]
    assert target.region.contains(head.x, head.y)                  # head is in the region


def test_gesturer_target_needs_the_right_pair():
    hands = [make_hand(S), make_hand(F, 0.7)]
    assert gesturer_target(hands, [(0, S), (1, F)], pair=(P, S)) is None  # needs palm, not fist


def test_resume_reports_who_to_lock_onto():
    f = PoseFollower(Settings(**INSTANT))
    a_up = make_pose(x=0.25, arms_up=True)
    stop = f.update(a_up, hands_at_wrists(a_up, F, P), 0.0)
    assert stop.toggled_stop and stop.resume_target is None        # only set on resume
    f.update(make_pose(x=0.25), [], 0.1)                           # hands down
    b_up = make_pose(x=0.75, arms_up=True)
    resume = f.update(make_pose(x=0.25), hands_at_wrists(b_up, P, S), 0.2)  # A still prominent
    assert resume.toggled_resume and resume.following
    assert pose_matches_target(b_up, resume.resume_target)
    assert not pose_matches_target(make_pose(x=0.25), resume.resume_target)
