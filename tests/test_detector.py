"""follower.vision.detector - pose-guided hand recovery and locking onto the gesturer.

Hand recovery is tested with a stand-in for the hand model (there is no real hand
footage in the repo); locking is tested with real MediaPipe on two people.
"""
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from mediapipe.framework.formats import landmark_pb2  # noqa: E402

from follower.logic import Region, ResumeTarget  # noqa: E402
from follower.vision import detector as D  # noqa: E402
from tests.landmarks import make_hand, make_pose  # noqa: E402

pytestmark = pytest.mark.vision
W, H = 1280, 720
ASPECT = W / H


def proto(lms):
    return landmark_pb2.NormalizedLandmarkList(landmark=[
        landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z, visibility=p.visibility)
        for p in lms])


def raised_pose(**kw):
    """Hands up at chest height, like the live recording (1280x720)."""
    return make_pose(x=0.5, y=0.55, shoulder_w=0.27, arms_up=True, aspect=ASPECT, **kw)


# ---- geometry ----------------------------------------------------------------------
def test_hand_roi_is_around_the_wrist_and_sized_from_the_body():
    pose = raised_pose()
    cx, cy, size = D.hand_roi(pose, 1, W, H)          # person's right wrist
    wx, wy = pose[16].x * W, pose[16].y * H
    assert abs(cx - wx) < 0.5 * size and abs(cy - wy) < 0.5 * size
    shoulders = abs(pose[11].x - pose[12].x) * W
    assert size == pytest.approx(1.2 * shoulders, rel=0.05)


def test_hand_roi_skips_unusable_wrists():
    pose = raised_pose()
    pose[16].visibility = 0.2
    assert D.hand_roi(pose, 1, W, H) is None           # not visible
    down = make_pose(x=0.5, y=0.3, shoulder_w=0.2, aspect=ASPECT)
    down[23].y = down[24].y = 0.7                       # hips in view, wrists below them
    down[15].y = 0.8
    assert D.hand_roi(down, 0, W, H) is None           # hand is down
    edge = raised_pose()
    edge[15].y = 0.99
    assert D.hand_roi(edge, 0, W, H) is None           # at the frame edge


def test_crop_square_pads_outside_the_image():
    img = np.full((100, 200, 3), 200, np.uint8)
    crop, x0, y0, s = D.crop_square(img, 5, 5, 40, out_size=40)
    assert (x0, y0, s) == (-15, -15, 40)
    assert crop[0, 0].tolist() == [0, 0, 0] and crop[-1, -1].tolist() == [200, 200, 200]


# ---- pose-guided hand recovery --------------------------------------------------------
class FakeRoiHands:
    """Pretends the hand model finds a fist whose wrist is at (u, v) of the crop."""

    def __init__(self, u=0.45, v=0.55, found=True):
        self.u, self.v, self.found, self.calls = u, v, found, 0

    def process(self, crop):
        self.calls += 1
        assert crop.shape == (D.ROI_SIZE, D.ROI_SIZE, 3)
        if not self.found:
            return SimpleNamespace(multi_hand_landmarks=None)
        lm = make_hand(None, self.u, self.v, size=0.3)
        return SimpleNamespace(multi_hand_landmarks=[proto(lm)])

    def close(self):
        pass


@pytest.fixture
def det():
    d = D.MediaPipeDetector()
    yield d
    d.close()


def test_missing_hand_is_recovered_from_a_crop_at_the_wrist(det):
    pose = raised_pose()
    left = proto(make_hand(None, pose[15].x, pose[15].y, size=0.1, aspect=ASPECT))
    det._roi_hands = fake = FakeRoiHands()
    rgb = np.zeros((H, W, 3), np.uint8)
    found = det._recover_hands(rgb, pose, [left])      # MediaPipe found only the left hand
    assert fake.calls == 1 and len(found) == 1         # right one recovered, left not redone
    cx, cy, size = D.hand_roi(pose, 1, W, H)
    wx = (cx - size / 2 + 0.45 * size) / W             # crop coords mapped back to the frame
    wy = (cy - size / 2 + 0.55 * size) / H
    assert found[0].landmark[0].x == pytest.approx(wx, abs=2 / W)
    assert found[0].landmark[0].y == pytest.approx(wy, abs=2 / H)


def test_recovery_rejects_a_hand_far_from_the_wrist(det):
    det._roi_hands = FakeRoiHands(u=0.02, v=0.02)      # corner of the crop: another hand
    assert det._recover_hands(np.zeros((H, W, 3), np.uint8), raised_pose(), []) == []


def test_recovery_does_nothing_when_both_hands_are_known_or_down(det):
    det._roi_hands = fake = FakeRoiHands()
    pose = raised_pose()
    both = [proto(make_hand(None, pose[i].x, pose[i].y, size=0.1, aspect=ASPECT))
            for i in (15, 16)]
    assert det._recover_hands(np.zeros((H, W, 3), np.uint8), pose, both) == []
    for i in (15, 16):
        pose[i].visibility = 0.1
    assert det._recover_hands(np.zeros((H, W, 3), np.uint8), pose, []) == []
    assert fake.calls == 0


def test_process_merges_recovered_hands(det, monkeypatch):
    pose = raised_pose()
    monkeypatch.setattr(det._pose, "process", lambda rgb: SimpleNamespace(pose_landmarks=proto(pose)))
    monkeypatch.setattr(det._hands, "process", lambda rgb: SimpleNamespace(multi_hand_landmarks=None))
    det._roi_hands = FakeRoiHands()
    d = det.process(np.zeros((H, W, 3), np.uint8))
    assert len(d.hands) == 2 and d.recovered_hands == 2


# ---- locking onto the gesturer ------------------------------------------------------------
def test_lock_gives_up_after_the_timeout_and_rejects_other_people(det, monkeypatch):
    other = make_pose(x=0.2)
    monkeypatch.setattr(det._pose, "process", lambda rgb: SimpleNamespace(pose_landmarks=proto(other)))
    monkeypatch.setattr(det._hands, "process", lambda rgb: SimpleNamespace(multi_hand_landmarks=None))
    det.lock_on(ResumeTarget(hands=((0.7, 0.6), (0.8, 0.6)), region=Region(0.6, 0.3, 1.0, 1.0)))
    results = [det.process(np.zeros((H, W, 3), np.uint8)) for _ in range(D.LOCK_TIMEOUT_FRAMES + 1)]
    assert all(r.pose_lm is None and r.lock_region for r in results[:-1])   # wrong person ignored
    assert results[-1].pose_lm is not None and results[-1].lock_region is None


def two_people_frame():
    from skimage import data
    ast = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    frame = np.full((H, W, 3), 60, np.uint8)
    frame[80:720, 20:660] = cv2.resize(ast, (640, 640))        # left: large, most prominent
    frame[300:720, 800:1220] = cv2.resize(ast, (420, 420))     # right: smaller
    return frame


def shoulder_x(d):
    return None if d.pose_lm is None else (d.pose_lm[11].x + d.pose_lm[12].x) / 2


def test_real_mediapipe_locks_onto_the_gesturer_and_keeps_following_them():
    pytest.importorskip("skimage")
    frame = two_people_frame()
    with D.MediaPipeDetector() as det:
        assert shoulder_x(det.process(frame)) < 0.5               # default: the prominent one
        det.lock_on(ResumeTarget(hands=((0.72, 0.75), (0.86, 0.75)),
                                 region=Region(0.6, 0.35, 1.0, 1.0)))
        xs = [shoulder_x(det.process(frame)) for _ in range(20)]
    assert all(x is not None and x > 0.6 for x in xs)              # right person, and stays


def test_end_to_end_resume_follows_the_gesturer(tmp_path):
    """Whole app loop, real MediaPipe pose on two people, scripted hands: the left
    person stops following, the right person resumes it -> the right person is followed."""
    pytest.importorskip("skimage")
    from tests.generate_cases import run_two_people
    s, det = run_two_people(tmp_path, "two", annotate=False)
    assert [f for _, f in s.toggles] == [False, True]
    resume_frame = round(s.toggles[-1][0] * 30)
    before = [x for x in det.followed_x[:30] if x is not None]
    after = [x for x in det.followed_x[resume_frame + 1:] if x is not None]
    assert before and all(x < 0.5 for x in before)
    assert after and all(x > 0.6 for x in after)
