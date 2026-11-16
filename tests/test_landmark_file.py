"""follower.logic.landmark_file - the .landmarks.jsonl recording format."""
import json
import struct

import pytest

from follower.logic import LandmarkFile, LandmarkFileWriter, is_landmark_file
from tests.landmarks import make_hand, make_pose


def f32(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def test_round_trip(tmp_path):
    path = tmp_path / "s.landmarks.jsonl"
    pose = make_pose()
    for lm in pose:                                  # MediaPipe values are float32
        lm.x, lm.y, lm.z, lm.visibility = f32(lm.x), f32(lm.y), f32(lm.z), f32(lm.visibility)
    hand = make_hand(None)
    for lm in hand:
        lm.x, lm.y = f32(lm.x), f32(lm.y)
    with LandmarkFileWriter(path, source="cam", kind="camera", width=640, height=480,
                            fps=30, mirrored=True, note="x") as w:
        w.write(0, 0.0, pose, [hand, hand])
        w.write(1, 1 / 30, None, [])
    f = LandmarkFile(path)
    assert (f.width, f.height, f.fps) == (640, 480, 30.0)
    assert f.header["mirrored"] is True and f.header["note"] == "x"
    frames = list(f.frames())
    assert [fr.index for fr in frames] == [0, 1]
    assert frames[1].t == 1 / 30                     # exact
    assert frames[1].pose is None and frames[1].hands == []
    # float32 values come back bit-identical -> replay makes identical decisions
    assert [(p.x, p.y, p.z, p.visibility) for p in frames[0].pose] == \
        [(p.x, p.y, p.z, p.visibility) for p in pose]
    assert [(p.x, p.y) for p in frames[0].hands[1]] == [(p.x, p.y) for p in hand]


def test_rejects_other_files(tmp_path):
    bad = tmp_path / "x.landmarks.jsonl"
    bad.write_text('{"format": "something else"}\n')
    with pytest.raises(ValueError):
        LandmarkFile(bad)
    bad.write_text("not json\n")
    with pytest.raises(ValueError):
        LandmarkFile(bad)


def test_reports_corrupt_frames(tmp_path):
    path = tmp_path / "c.landmarks.jsonl"
    with LandmarkFileWriter(path, source="s", kind="video", width=1, height=1, fps=1):
        pass
    with path.open("a") as f:
        f.write(json.dumps({"i": 0}) + "\n")
    with pytest.raises(ValueError, match=":2:"):
        list(LandmarkFile(path).frames())


def test_is_landmark_file():
    assert is_landmark_file("a/b.landmarks.jsonl") and is_landmark_file("B.LANDMARKS.JSONL")
    assert not is_landmark_file("b.jsonl") and not is_landmark_file("b.mp4")


def test_header_fps_is_corrected_to_the_real_rate(tmp_path):
    """Cameras can report nonsense (the live recording's header said 1 fps at ~10 fps)."""
    path = tmp_path / "cam.landmarks.jsonl"
    with LandmarkFileWriter(path, source="cam", kind="camera", width=2, height=2, fps=1.0) as w:
        for i in range(21):
            w.write(i, i / 10, None, [])
    f = LandmarkFile(path)
    assert f.fps == pytest.approx(10) and f.header["nominal_fps"] == 1.0
    assert len(list(f.frames())) == 21


def test_header_fps_kept_when_right(tmp_path):
    path = tmp_path / "v.landmarks.jsonl"
    with LandmarkFileWriter(path, source="v", kind="video", width=2, height=2, fps=30) as w:
        for i in range(30):
            w.write(i, i / 30, None, [])
    assert LandmarkFile(path).fps == 30 and "nominal_fps" not in LandmarkFile(path).header
