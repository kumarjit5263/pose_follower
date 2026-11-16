"""follower.vision - capture, MediaPipe wrapper, drawing, outputs and the main loop.

Needs cv2 + mediapipe (skipped otherwise). Uses generated videos, scripted fake
detectors and fake cameras, so no camera and no real footage are required.
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from mediapipe.framework.formats import landmark_pb2  # noqa: E402

from follower.logic import ActionKind as K, FrameResult, Gesture, GestureOption, \
    GesturePhase, LandmarkFile, Status  # noqa: E402
from follower.vision import app, capture, drawing  # noqa: E402
from follower.vision.detector import Detection, MediaPipeDetector  # noqa: E402
from follower.vision.outputs import VideoOut  # noqa: E402
from tests.landmarks import hands_at_wrists, make_hand, make_pose  # noqa: E402

pytestmark = pytest.mark.vision
F, P, S = Gesture.FIST, Gesture.PALM, Gesture.SCISSORS


def write_video(path, n_frames=12, fps=10.0, size=(160, 120)):
    w, h = size
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert out.isOpened()
    for i in range(n_frames):
        out.write(np.full((h, w, 3), (i * 20) % 255, np.uint8))
    out.release()
    return path


def to_proto(lms):
    return landmark_pb2.NormalizedLandmarkList(landmark=[
        landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z, visibility=p.visibility)
        for p in lms])


def fake_camera(n_frames, fps=30.0, size=(160, 120)):
    """A live FrameSource whose 'camera' delivers n frames at roughly ``fps``."""
    w, h = size
    frames = [np.full((h, w, 3), 30 + i % 200, np.uint8) for i in range(n_frames)]
    frames.reverse()

    def read():
        time.sleep(1 / fps)
        return (True, frames.pop()) if frames else (False, None)

    cap = SimpleNamespace(get=lambda prop: {cv2.CAP_PROP_FPS: 30.0}.get(prop, 0),
                          read=read, release=lambda: None)
    src = capture.FrameSource(cap, "camera 0", live=True, mirror=True)
    src.width, src.height = w, h
    return src


class ScriptedDetector:
    """Stands in for MediaPipeDetector: returns pre-scripted landmarks per frame."""

    def __init__(self, script):
        self.script = iter(script)
        self.resets = 0
        self.locks = []

    def process(self, frame):
        pose, hands = next(self.script, (None, []))
        return Detection(pose_landmarks=to_proto(pose) if pose else None,
                         hand_landmarks=[to_proto(h) for h in hands])

    def reset_pose(self):
        self.resets += 1

    def lock_on(self, target):
        self.locks.append(target)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


# ---- capture -----------------------------------------------------------------
def test_video_timestamps_come_from_the_video_not_the_wall_clock(tmp_path):
    with capture.open_video(write_video(tmp_path / "v.avi", n_frames=12, fps=10)) as src:
        assert not src.live and src.fps == pytest.approx(10)
        assert (src.width, src.height) == (160, 120)
        stamps = [t for _, t in src.frames()]
    assert stamps == pytest.approx([i / 10 for i in range(12)])


def test_live_timestamps_start_at_zero():
    stamps = [t for _, t in fake_camera(5).frames()]
    assert stamps[0] == 0.0 and stamps == sorted(stamps) and stamps[-1] > 0.05


def test_open_video_rejects_unreadable_files(tmp_path):
    with pytest.raises(IOError):
        capture.open_video(tmp_path / "missing.mp4")
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    with pytest.raises(IOError):
        capture.open_video(tmp_path / "junk.mp4")
    (tmp_path / "junk.landmarks.jsonl").write_text("nope\n")
    with pytest.raises(IOError):
        capture.open_replay(tmp_path / "junk.landmarks.jsonl")


def test_missing_camera_returns_none():
    assert capture.open_camera(97) is None


# ---- detector ----------------------------------------------------------------
def test_detector_on_empty_frame_finds_nothing():
    with MediaPipeDetector() as det:
        d = det.process(np.zeros((240, 320, 3), np.uint8))
        assert d.pose_lm is None and d.hands == []
        det.reset_pose()


def test_detection_from_record_round_trips():
    from follower.logic import FrameRecord
    rec = FrameRecord(0, 0.0, make_pose(), [make_hand(F)])
    d = Detection.from_record(rec)
    assert len(d.pose_lm) == 33 and len(d.hands) == 1 and len(d.hands[0]) == 21
    assert d.pose_lm[11].x == pytest.approx(rec.pose[11].x)


# ---- drawing -----------------------------------------------------------------
STOP_PAIR = (P, F)
RESUME_PAIR = (P, S)
DRAW_PAIR = (F, S)


def stop_option(phase=GesturePhase.ARMED, hold_elapsed=0.0):
    return GestureOption("stop", STOP_PAIR, phase, hold_elapsed)


def draw_option(phase=GesturePhase.ARMED, hold_elapsed=0.0):
    return GestureOption("toggle_drawing", DRAW_PAIR, phase, hold_elapsed)


def resume_option(phase=GesturePhase.ARMED, hold_elapsed=0.0):
    return GestureOption("resume", RESUME_PAIR, phase, hold_elapsed)


def test_render_draws_pose_labels_status_and_header():
    pose = make_pose(arms_up=True)
    det = Detection(pose_landmarks=to_proto(pose),
                    hand_landmarks=[to_proto(h) for h in hands_at_wrists(pose, F, P)])
    result = FrameResult(following=True, status=Status.FOLLOWING,
                         hand_gestures=[(0, F), (1, P)], flash_text="FOLLOWING RESUMED",
                         options=[stop_option(), draw_option()])
    blank = np.zeros((240, 320, 3), np.uint8)
    img = drawing.render(blank.copy(), det, result)
    assert (img != blank).any(axis=2).sum() > 500
    assert ((img == (255, 255, 0)).all(axis=2)).any()   # pose connections colour (BGR)
    assert drawing.header_lines(result) == ["Hold open palm + fist to STOP tracking",
                                             "Hold fist + scissors to HIDE the body pose drawing"]


def test_header_reflects_resume_state_and_whether_drawing_is_shown():
    stopped = FrameResult(following=False, status=Status.STOPPED, options=[resume_option()])
    assert drawing.header_lines(stopped) == \
        ["Hold open palm + scissors to RESUME tracking (on whoever shows it)"]
    hidden = FrameResult(following=True, status=Status.FOLLOWING, show_skeleton=False,
                         options=[draw_option()])
    assert drawing.header_lines(hidden) == ["Hold fist + scissors to SHOW the body pose drawing"]


def test_pose_is_not_drawn_while_stopped():
    det = Detection(pose_landmarks=to_proto(make_pose()))
    img = drawing.render(np.zeros((240, 320, 3), np.uint8), det,
                         FrameResult(following=False, status=Status.STOPPED,
                                    options=[resume_option()]))
    assert not ((img == (255, 255, 0)).all(axis=2)).any()


def test_pose_is_not_drawn_while_hidden_by_the_toggle():
    det = Detection(pose_landmarks=to_proto(make_pose()))
    img = drawing.render(np.zeros((240, 320, 3), np.uint8), det,
                         FrameResult(following=True, status=Status.FOLLOWING,
                                    show_skeleton=False, options=[stop_option(), draw_option()]))
    assert not ((img == (255, 255, 0)).all(axis=2)).any()


def test_hold_progress_bar_and_release_prompt():
    det = Detection()
    base = FrameResult(following=True, status=Status.SEARCHING,
                       options=[stop_option(), draw_option()])
    plain = drawing.render(np.zeros((240, 320, 3), np.uint8), det, base)
    holding = drawing.render(np.zeros((240, 320, 3), np.uint8), det, FrameResult(
        following=True, status=Status.SEARCHING,
        options=[stop_option(GesturePhase.HOLDING, 1.5), draw_option()]))
    bar = holding[240 - 28:240 - 14, 12:312]
    assert ((bar == drawing.ORANGE).all(axis=2)).mean() > 0.4      # about half full
    release = drawing.render(np.zeros((240, 320, 3), np.uint8), det, FrameResult(
        following=True, status=Status.FOLLOWING,
        options=[stop_option(), draw_option(GesturePhase.WAIT_RELEASE)]))
    assert (holding[200:] != plain[200:]).any() and (release[200:] != plain[200:]).any()


def test_every_status_has_a_label():
    assert set(drawing.STATUS_TEXT) == set(Status)


# ---- outputs -------------------------------------------------------------------
def test_video_out_retimes_to_the_real_frame_rate(tmp_path):
    out = VideoOut(tmp_path / "r.mp4", fps=30.0)
    for i in range(20):                        # frames actually arrived at 10 fps
        out.write(np.zeros((64, 64, 3), np.uint8), i / 10)
    path = out.close()
    cap = cv2.VideoCapture(str(path))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10, abs=0.1)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    assert not (tmp_path / "r.partial.mp4").exists()


# ---- main loop -----------------------------------------------------------------
def scripted_session():
    """Person followed 1 s; holds open palm + fist 3.5 s (stop @ 4.1 s); lowers hands and
    leaves; someone else holds open palm + scissors 3.2 s (resume @ 9.1 s); followed again."""
    a, a_up = make_pose(x=0.3), make_pose(x=0.3, arms_up=True)
    b, b_up = make_pose(x=0.75), make_pose(x=0.75, arms_up=True)
    return ([(a, [])] * 30 + [(a_up, hands_at_wrists(a_up, F, P))] * 105 + [(a, [])] * 30 +
            [(None, [])] * 15 + [(b_up, hands_at_wrists(b_up, P, S))] * 96 + [(b, [])] * 30)


def test_run_source_toggles_resets_tracker_and_writes_outputs(tmp_path):
    from follower.vision.outputs import SessionOutputs
    script = scripted_session()
    video = write_video(tmp_path / "scripted.avi", n_frames=len(script), fps=30)
    fake = ScriptedDetector(script)
    with capture.open_video(video) as src:
        outs = SessionOutputs(tmp_path / "out", "scripted", src, landmarks=True, annotated=True)
        s = app.run_source(src, display=False, outputs=outs, detector_factory=lambda: fake)
    assert s.frames == len(script)
    # 3 s hold + 0.1 s for the hands' gestures to be confirmed (label smoothing)
    assert s.toggles == [(pytest.approx(4.1), False), (pytest.approx(9.1), True)]
    assert fake.resets == 2                           # tracker reset at every toggle
    from follower.logic import pose_matches_target
    stop_lock, resume_lock = fake.locks               # resume locks onto the gesturer (B)
    assert stop_lock is None
    assert pose_matches_target(make_pose(x=0.75), resume_lock)
    assert not pose_matches_target(make_pose(x=0.3), resume_lock)
    assert s.log.kinds() == [K.START, K.PERSON_FOUND, K.GESTURE_STARTED, K.STOPPED,
                             K.GESTURE_STARTED, K.RESUMED, K.PERSON_FOUND, K.END]
    names = sorted(p.name for p in s.files)
    assert names == ["scripted.actions.txt", "scripted.landmarks.jsonl",
                     "scripted_annotated.mp4"]
    report = (tmp_path / "out" / "scripted.actions.txt").read_text()
    assert "FOLLOWING_STOPPED" in report and "00:04.100" in report

    # the landmarks written from the video replay to exactly the same actions
    with capture.open_replay(tmp_path / "out" / "scripted.landmarks.jsonl") as rep:
        r = app.run_source(rep, display=False)
    assert r.toggles == s.toggles
    assert [(a.t, a.kind) for a in r.log.actions] == [(a.t, a.kind) for a in s.log.actions]


def test_main_with_no_camera_and_no_recordings_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app, "open_camera", lambda index: None)
    assert app.main(["--scenarios", str(tmp_path), "--no-display"]) == 1
    assert "No camera and no recordings" in capsys.readouterr().err


def test_main_falls_back_to_scenarios_and_writes_outputs(tmp_path, monkeypatch, capsys):
    from tests.cases import CASES_BY_NAME, write_landmarks
    monkeypatch.setattr(app, "open_camera", lambda index: None)
    scen, outs = tmp_path / "scen", tmp_path / "outputs"
    scen.mkdir()
    write_video(scen / "one.avi", n_frames=5)
    write_landmarks(CASES_BY_NAME["stop_and_resume"], scen / "sr.landmarks.jsonl")
    assert app.main(["--scenarios", str(scen), "--outputs", str(outs), "--no-display"]) == 0
    out = capsys.readouterr().out
    assert "No camera available" in out and "one.avi" in out and "sr.landmarks.jsonl" in out
    assert "1 stop(s), 1 resume(s)" in out
    assert sorted(p.relative_to(outs).as_posix() for p in outs.rglob("*")) == [
        "one", "one/one.actions.txt", "one/one.landmarks.jsonl",
        "sr", "sr/sr.replay.actions.txt"]


def test_main_no_save_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "open_camera", lambda index: None)
    write_video(tmp_path / "one.avi", n_frames=3)
    outs = tmp_path / "outputs"
    assert app.main(["--video", str(tmp_path / "one.avi"), "--outputs", str(outs),
                     "--no-display", "--no-save"]) == 0
    assert not outs.exists()


def test_main_prefers_the_camera_when_available(monkeypatch, capsys):
    monkeypatch.setattr(app, "open_camera", lambda index: fake_camera(3))
    monkeypatch.setattr(app, "find_scenarios", lambda *a: pytest.fail("should not look"))
    assert app.main(["--no-display"]) == 0
    assert "Using live camera 0" in capsys.readouterr().out


def test_record_session(tmp_path, monkeypatch, capsys):
    script = scripted_session()[:45]
    monkeypatch.setattr(app, "open_camera", lambda index: fake_camera(len(script)))
    monkeypatch.setattr(app, "MediaPipeDetector", lambda: ScriptedDetector(script))
    assert app.main(["--record", "sess", "--outputs", str(tmp_path), "--no-display"]) == 0
    folder = tmp_path / "sess"
    assert sorted(p.name for p in folder.iterdir()) == \
        ["sess.actions.txt", "sess.landmarks.jsonl", "sess.mp4"]
    rec = LandmarkFile(folder / "sess.landmarks.jsonl")
    frames = list(rec.frames())
    assert rec.header["kind"] == "camera" and rec.header["mirrored"] is True
    assert len(frames) == 45 and frames[0].t == 0.0 and frames[0].pose is not None
    cap = cv2.VideoCapture(str(folder / "sess.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 45
    measured = 44 / frames[-1].t                      # the video plays at the real rate
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(measured, rel=0.06)
    assert "live camera" in (folder / "sess.actions.txt").read_text()
    assert "tests/scenarios" in capsys.readouterr().out

    # refuses to overwrite an existing recording
    monkeypatch.setattr(app, "open_camera", lambda index: fake_camera(3))
    assert app.main(["--record", "sess", "--outputs", str(tmp_path), "--no-display"]) == 1


def test_record_without_camera_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app, "open_camera", lambda index: None)
    assert app.main(["--record", "--outputs", str(tmp_path), "--no-display"]) == 1
    assert "needs a live camera" in capsys.readouterr().err


def test_duration_limits_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "open_camera", lambda index: None)
    write_video(tmp_path / "v.avi", n_frames=30, fps=10)
    with capture.open_video(tmp_path / "v.avi") as src:
        s = app.run_source(src, display=False, duration=1.0,
                           detector_factory=lambda: ScriptedDetector([]))
    assert s.frames == 11                              # t = 0.0 .. 1.0


def test_headless_linux_is_detected(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not app.display_available()
    monkeypatch.setenv("DISPLAY", ":0")
    assert app.display_available()


# ---- real MediaPipe on a real photo ------------------------------------------------
def test_astronaut_video_with_real_mediapipe(tmp_path):
    """Person in view, leaves for 1 s, comes back: found / lost / found; and the
    landmarks file written from the video replays to the same actions."""
    pytest.importorskip("skimage")
    from tests.generate_cases import make_astronaut_video
    video = make_astronaut_video(tmp_path / "astro.mp4", seconds_in=1.5, seconds_out=1.0,
                                 seconds_back=1.5)
    assert app.main(["--video", str(video), "--outputs", str(tmp_path / "o"),
                     "--no-display"]) == 0
    report = (tmp_path / "o" / "astro" / "astro.actions.txt").read_text()
    assert report.count("PERSON_FOUND") == 2 and report.count("PERSON_LOST") == 1, report
    assert "FOLLOWING_STOPPED" not in report
    with capture.open_replay(tmp_path / "o" / "astro" / "astro.landmarks.jsonl") as rep:
        r = app.run_source(rep, display=False)
    assert r.log.kinds() == [K.START, K.PERSON_FOUND, K.PERSON_LOST, K.PERSON_FOUND, K.END]


def test_pose_is_not_drawn_while_searching():
    """On the frame following resumes, the pose still belongs to whoever was tracked
    while stopped - it must not be drawn as if they were followed."""
    det = Detection(pose_landmarks=to_proto(make_pose()))
    img = drawing.render(np.zeros((240, 320, 3), np.uint8), det,
                         FrameResult(following=True, status=Status.SEARCHING, toggled=True))
    assert not ((img == (255, 255, 0)).all(axis=2)).any()


def test_long_labels_fit_inside_the_frame():
    img = np.zeros((120, 320, 3), np.uint8)
    drawing.draw_label(img, "STOPPED - hold open palm + scissors to resume", drawing.RED)
    cols = np.where(img.any(axis=(0, 2)))[0]
    assert cols.max() < 320 - 4
