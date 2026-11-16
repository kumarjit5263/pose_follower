"""follower.logic.scenarios - discovering recordings (pure filesystem)."""
from follower.logic import DEFAULT_SCENARIO_DIR, find_scenarios, scenario_stem


def test_default_dir_is_tests_scenarios():
    assert DEFAULT_SCENARIO_DIR.parts[-2:] == ("tests", "scenarios")
    assert DEFAULT_SCENARIO_DIR.is_dir()


def test_finds_videos_and_landmark_files_recursively_sorted(tmp_path):
    for rel in ["b.mp4", "a.AVI", "sub/c.mov", "sub/deep/d.webm", "e.landmarks.jsonl",
                "notes.txt", "b.expected.json", "f.jsonl", "b.actions.txt"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    names = [p.relative_to(tmp_path).as_posix() for p in find_scenarios(tmp_path)]
    assert names == ["a.AVI", "b.mp4", "e.landmarks.jsonl", "sub/c.mov", "sub/deep/d.webm"]


def test_accepts_files_and_folders_and_ignores_missing(tmp_path):
    (tmp_path / "x.mkv").write_bytes(b"")
    (tmp_path / "y.txt").write_bytes(b"")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "z.mp4").write_bytes(b"")
    found = find_scenarios([tmp_path / "x.mkv", tmp_path / "y.txt", tmp_path / "dir",
                            tmp_path / "missing"])
    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["dir/z.mp4", "x.mkv"]
    assert find_scenarios(tmp_path / "nope") == []


def test_scenario_stem():
    assert scenario_stem("a/walk.mp4") == "walk"
    assert scenario_stem("walk.landmarks.jsonl") == "walk"
    assert scenario_stem("my.clip.v2.mov") == "my.clip.v2"
