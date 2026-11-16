"""Runs the full pipeline (no window) on every recording in tests/scenarios/:
videos (real MediaPipe) and .landmarks.jsonl files (replayed).

Optional ``<name>.expected.json`` (see tests/scenarios/README.md) holds the
expectations, and may override any of ``stop_gesture`` / ``resume_gesture`` /
``draw_toggle_gesture``.

Optional ``<name>.live_actions.txt`` is the action log the app wrote while the
recording was being captured live (``--record``); replaying the recording must
reproduce it exactly - every action, in order, to the millisecond.
"""
import json
import re

import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from follower.logic import DEFAULT_SETTINGS, Settings, find_scenarios, is_landmark_file, \
    scenario_stem  # noqa: E402
from follower.vision.app import run_source  # noqa: E402
from follower.vision.capture import open_replay, open_video  # noqa: E402

FILES = find_scenarios()
OVERRIDABLE = ("stop_gesture", "resume_gesture", "draw_toggle_gesture")
LIVE_LOGS = [f for f in FILES if f.with_name(f"{scenario_stem(f)}.live_actions.txt").exists()]
_ACTION_LINE = re.compile(r"^(\d\d):(\d\d\.\d{3})\s+([A-Z_]+)\s")


def read_action_log(path):
    """[(seconds since start, ACTION), ...] from an ``.actions.txt`` report."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ACTION_LINE.match(line)
        if m:
            out.append((round(int(m[1]) * 60 + float(m[2]), 3), m[3]))
    return out


def expected_for(path):
    f = path.with_name(f"{scenario_stem(path)}.expected.json")
    return json.loads(f.read_text()) if f.exists() else {}


def settings_for(expected) -> Settings:
    overrides = {k: tuple(expected[k]) for k in OVERRIDABLE if k in expected}
    return Settings(**overrides) if overrides else DEFAULT_SETTINGS


@pytest.mark.scenario
@pytest.mark.parametrize("path", FILES, ids=[f.name for f in FILES])
def test_scenario_file(path):
    expected = expected_for(path)
    with (open_replay(path) if is_landmark_file(path) else open_video(path)) as src:
        s = run_source(src, display=False, settings=settings_for(expected))
    assert s.frames > 0, "recording produced no frames"
    if "toggles" in expected:
        assert [following for _, following in s.toggles] == expected["toggles"], s.toggles
    if "actions" in expected:
        assert [k.value for k in s.log.kinds()] == expected["actions"]
    if "min_followed_fraction" in expected:
        assert s.person_frames / s.frames >= expected["min_followed_fraction"]


@pytest.mark.scenario
@pytest.mark.parametrize("path", LIVE_LOGS, ids=[f.name for f in LIVE_LOGS])
def test_replay_reproduces_the_live_session_exactly(path):
    live = read_action_log(path.with_name(f"{scenario_stem(path)}.live_actions.txt"))
    assert live, "no actions found in the live log"
    with open_replay(path) as src:
        s = run_source(src, display=False)
    start = s.log.start_t
    replay = [(round(a.t - start, 3), a.kind.value) for a in s.log.actions]
    assert replay == live, ("replay differs from the live session - if the gestures or "
                            "their settings changed since recording, re-record it")
