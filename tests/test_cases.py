"""The synthetic scenarios in tests/cases.py: expected toggles and action timelines,
both through the pure logic and replayed from a .landmarks.jsonl file by the app."""
import pytest

from tests.cases import CASES, simulate, write_landmarks


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_case_logic(case):
    toggles, log = simulate(case)
    assert [following for _, following in toggles] == case.expected_toggles
    assert log.kinds() == case.expected_actions


@pytest.mark.vision
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_case_replayed_from_file(case, tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from follower.vision.app import run_source
    from follower.vision.capture import open_replay

    path = tmp_path / f"{case.name}.landmarks.jsonl"
    write_landmarks(case, path)
    with open_replay(path) as src:
        s = run_source(src, display=False)
    assert [following for _, following in s.toggles] == case.expected_toggles
    assert s.log.kinds() == case.expected_actions
